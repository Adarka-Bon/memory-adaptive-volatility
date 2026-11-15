"""Attention-mechanism comparison under time-varying memory regimes.

Reuses the diploma attention implementations (Standard, ProbSparse,
MultiFeature) and evaluates them separately on high-memory and low-memory
test regimes, defined by the local GPH estimate d(t).

This is the core comparative-statistics experiment for the first semester.
"""
import math
import warnings
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

SRC_DATA = r"C:\diploma\realization\source"
OUT = r"C:\diploma\magistracy\results"
FIG = r"C:\diploma\magistracy\figures"

SEED = 42
INPUT_WINDOW = 60
BATCH_SIZE = 64
EPOCHS = 8
LR = 1e-3
D_MODEL = 32
NHEAD = 4
NUM_LAYERS = 2
DROPOUT = 0.1
TEST_SIZE = 0.30
MAX_ROWS = 60_000
MEM_WINDOW = 1500
MEM_STEP = 250
VAL_FRACTION = 0.15

np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cpu")


# --------------------------------------------------------------------------
# Data preparation (following the diploma pipeline, reduced feature set)
# --------------------------------------------------------------------------
def load_data(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["datetime"] = df.apply(
        lambda r: pd.Timestamp(r["date"]) + pd.Timedelta(hours=int(r["minute"].split(":")[0]),
                                                          minutes=int(r["minute"].split(":")[1])),
        axis=1)
    df = df.set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def add_features(df):
    df = df.copy()
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"]).diff()
    df["range_hl"] = df["high"] - df["low"]
    df["range_co"] = df["close"] - df["open"]
    df["volatility_20"] = df["returns"].rolling(20).std()
    df["ma_10"] = df["close"].rolling(10).mean()
    df["ma_20"] = df["close"].rolling(20).mean()
    df["ma_ratio"] = df["ma_10"] / (df["ma_20"] + 1e-9)
    df["volume_ma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_ma_20"] + 1e-9)
    for lag in (1, 2, 3, 5):
        df[f"close_lag_{lag}"] = df["close"].shift(lag)
        df[f"returns_lag_{lag}"] = df["returns"].shift(lag)
    df = df.bfill().ffill()
    return df


FEATURES = ["open", "high", "low", "close", "volume",
            "returns", "log_returns", "range_hl", "range_co", "volatility_20",
            "ma_10", "ma_20", "ma_ratio", "volume_ratio",
            "close_lag_1", "close_lag_2", "close_lag_3", "close_lag_5",
            "returns_lag_1", "returns_lag_2", "returns_lag_3", "returns_lag_5"]
TARGET = "close"


# --------------------------------------------------------------------------
# Attention architectures (from the diploma implementation)
# --------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class StandardTransformer(nn.Module):
    """Full self-attention, O(T^2)."""
    def __init__(self, input_dim, d_model=D_MODEL, nhead=NHEAD,
                 num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x):
        x = self.pos(self.input_proj(x))
        return self.head(self.encoder(x)[:, -1, :]).squeeze(-1)


class ProbSparseAttention(nn.Module):
    """Top-k sparse attention, O(T log T)."""
    def __init__(self, d_model, n_heads=NHEAD, sparsity=0.3):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.sparsity = sparsity
        self.scale = math.sqrt(self.d_head)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head
        split = lambda t: t.view(B, T, H, Dh).transpose(1, 2)
        Q, K, V = split(self.q(x)), split(self.k(x)), split(self.v(x))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        k_top = max(1, int(T * self.sparsity))
        tv, ti = torch.topk(scores, k_top, dim=-1)
        mask = torch.full_like(scores, -1e9)
        mask.scatter_(-1, ti, tv)
        out = torch.matmul(torch.softmax(mask, dim=-1), V)
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, D))


class ProbSparseBlock(nn.Module):
    def __init__(self, d_model, n_heads=NHEAD, dropout=DROPOUT):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = ProbSparseAttention(d_model, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ff(self.norm2(x))


class ProbSparseTransformer(nn.Module):
    def __init__(self, input_dim, d_model=D_MODEL, n_heads=NHEAD,
                 num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList([ProbSparseBlock(d_model, n_heads, dropout)
                                     for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.pos(self.input_proj(x))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x[:, -1, :])).squeeze(-1)


class MultiFeatureTransformer(nn.Module):
    """Two-path transformer: temporal + variate attention with a gate."""
    def __init__(self, input_dim, seq_len=INPUT_WINDOW, d_model=D_MODEL,
                 nhead=NHEAD, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.t_proj = nn.Linear(input_dim, d_model)
        self.t_pos = PositionalEncoding(d_model)
        t_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.t_encoder = nn.TransformerEncoder(t_layer, num_layers=num_layers)

        self.v_proj = nn.Linear(seq_len, d_model)
        v_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.v_encoder = nn.TransformerEncoder(v_layer, num_layers=num_layers)

        self.gate = nn.Parameter(torch.tensor(0.5))
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout),
                                  nn.Linear(d_model, 1))

    def forward(self, x):
        xt = self.t_pos(self.t_proj(x))
        xt = self.t_encoder(xt)[:, -1, :]
        xv = self.v_proj(x.transpose(1, 2))
        xv = self.v_encoder(xv).mean(dim=1)
        g = torch.sigmoid(self.gate)
        return self.head(g * xt + (1 - g) * xv).squeeze(-1)


# --------------------------------------------------------------------------
# Training utilities
# --------------------------------------------------------------------------
class TSDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = torch.FloatTensor(X), torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def make_sequences(X, y, window):
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i:i + window])
        ys.append(y[i + window])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def train_model(model, train_loader, val_loader, epochs=EPOCHS):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2, factor=0.5)
    loss_fn = nn.MSELoss()
    best_val, best_state, patience = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        vals = []
        with torch.no_grad():
            for xb, yb in val_loader:
                vals.append(loss_fn(model(xb), yb).item())
        v = np.mean(vals)
        sched.step(v)
        if v < best_val:
            best_val, best_state, patience = v, {k: p.clone() for k, p in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 3:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


def predict(model, X):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 1024):
            preds.append(model(torch.FloatTensor(X[i:i + 1024])).numpy())
    return np.concatenate(preds)


# --------------------------------------------------------------------------
# Local memory estimate d(t) on the test range
# --------------------------------------------------------------------------
def gph_d(x, power=0.5):
    n = len(x)
    m = int(n ** power)
    if m < 5:
        return np.nan
    x = x - x.mean()
    freqs = 2 * np.pi * np.arange(1, n // 2 + 1) / n
    fft = np.fft.rfft(x)
    per = (np.abs(fft) ** 2) / (2 * np.pi * n)
    per = per[1:n // 2 + 1]
    lam = freqs[:m]
    per = per[:m]
    mask = per > 0
    lam, per = lam[mask], per[mask]
    if len(lam) < 5:
        return np.nan
    xreg = np.log(4 * np.sin(lam / 2) ** 2)
    slope = np.polyfit(xreg, np.log(per), 1)[0]
    return -slope / 2.0


def rolling_d(series, window, step):
    v = series.to_numpy(float)
    rows = []
    for s in range(0, len(v) - window + 1, step):
        seg = v[s:s + window]
        if np.isnan(seg).any():
            continue
        rows.append({"start": s, "end": s + window - 1, "d": gph_d(seg)})
    return pd.DataFrame(rows)


def memory_of_test_points(returns_test, window=MEM_WINDOW, step=MEM_STEP):
    """Assign each test index a local d based on a window ending at that index."""
    roll = rolling_d(returns_test, window, step)
    if roll.empty:
        return np.full(len(returns_test), np.nan)
    # map each point to the d of the nearest window whose end <= point
    dmap = np.full(len(returns_test), np.nan)
    for _, r in roll.iterrows():
        end = int(r["end"])
        nxt = min(end + step, len(returns_test))
        dmap[end:nxt] = r["d"]
    return dmap


# --------------------------------------------------------------------------
# Main experiment
# --------------------------------------------------------------------------
def main():
    print("Loading data ...")
    df = load_data(rf"{SRC_DATA}\facebook.csv")
    if len(df) > MAX_ROWS:
        df = df.iloc[-MAX_ROWS:].copy()
    df = add_features(df)
    print(f"rows: {len(df)}  range: {df.index.min()} .. {df.index.max()}")

    split = int(len(df) * (1 - TEST_SIZE))
    train_df, test_df = df.iloc[:split].copy(), df.iloc[split:].copy()

    fs, ts = [], []
    for c in FEATURES:
        sc = RobustScaler()
        fs.append(sc.fit_transform(train_df[[c]]).flatten())
        ts.append(sc.transform(test_df[[c]]).flatten())
    train_m, test_m = np.column_stack(fs), np.column_stack(ts)

    tsc = MinMaxScaler()
    y_tr = tsc.fit_transform(train_df[TARGET].values.reshape(-1, 1)).flatten()
    y_te = tsc.transform(test_df[TARGET].values.reshape(-1, 1)).flatten()
    inv = lambda a: tsc.inverse_transform(np.array(a).reshape(-1, 1)).flatten()

    Xtr, ytr = make_sequences(train_m, y_tr, INPUT_WINDOW)
    Xte, yte = make_sequences(test_m, y_te, INPUT_WINDOW)
    print(f"X_train {Xtr.shape}  X_test {Xte.shape}")

    n_val = int(len(Xtr) * VAL_FRACTION)
    Xva, yva = Xtr[-n_val:], ytr[-n_val:]
    Xtr2, ytr2 = Xtr[:-n_val], ytr[:-n_val]
    tr_loader = DataLoader(TSDataset(Xtr2, ytr2), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    va_loader = DataLoader(TSDataset(Xva, yva), batch_size=BATCH_SIZE, shuffle=False)

    models = {
        "Standard": StandardTransformer(len(FEATURES)),
        "ProbSparse": ProbSparseTransformer(len(FEATURES)),
        "MultiFeature": MultiFeatureTransformer(len(FEATURES)),
    }

    preds = {}
    for name, model in models.items():
        print(f"Training {name} ...")
        model = train_model(model, tr_loader, va_loader)
        preds[name] = inv(predict(model, Xte))

    y_true = inv(yte)

    # local memory on the test range (based on returns)
    d_series = test_df["returns"].reset_index(drop=True)
    d_vals = memory_of_test_points(d_series)
    # align with test sequence targets: sequence j predicts point window+j
    d_seq = d_vals[INPUT_WINDOW:]
    q25, q75 = np.nanpercentile(d_seq, [25, 75])
    high = d_seq >= q75
    low = d_seq <= q25
    print(f"d(t): q25={q25:.3f}  median={np.nanmedian(d_seq):.3f}  q75={q75:.3f}")
    print(f"high-memory points={high.sum()}  low-memory points={low.sum()}")

    def metrics(yt, yp):
        rmse = np.sqrt(mean_squared_error(yt, yp))
        mae = mean_absolute_error(yt, yp)
        naive = np.sqrt(np.mean((yt[1:] - yt[:-1]) ** 2)) if len(yt) > 1 else np.nan
        return {"RMSE": rmse, "MAE": mae,
                "TheilU2": rmse / (naive + 1e-9),
                "R2": r2_score(yt, yp)}

    rows = []
    for name, yp in preds.items():
        all_m = metrics(y_true, yp)
        rows.append({"mechanism": name, "regime": "all", **all_m})
        for lab, mask in (("high-memory", high), ("low-memory", low)):
            yt_m, yp_m = y_true[mask], yp[mask]
            rows.append({"mechanism": name, "regime": lab, **metrics(yt_m, yp_m)})

    out = pd.DataFrame(rows)
    out.to_csv(rf"{OUT}\attention_regimes.csv", index=False)
    print("\n=== Attention mechanisms by memory regime ===")
    print(out.to_string(index=False))

    # pairwise gap: does ranking change between regimes?
    print("\n=== RMSE degradation from low- to high-memory regime ===")
    piv = out.pivot(index="mechanism", columns="regime", values="RMSE")
    piv["degradation_%"] = 100 * (piv["high-memory"] - piv["low-memory"]) / piv["low-memory"]
    print(piv.to_string())

    np.save(rf"{OUT}\attention_preds.npy",
            {"y_true": y_true, "d_seq": d_seq, **preds}, allow_pickle=True)

    make_figure(out, d_seq)


def make_figure(out, d_seq):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mechs = out["mechanism"].unique()
    regimes = ["low-memory", "all", "high-memory"]
    x = np.arange(len(mechs))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    for i, reg in enumerate(regimes):
        vals = [out[(out.mechanism == m) & (out.regime == reg)]["RMSE"].values[0]
                for m in mechs]
        axes[0].bar(x + i * width, vals, width, label=reg)
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(mechs)
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("RMSE механизмов внимания по режимам памяти")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].plot(d_seq, lw=0.8, color="black")
    axes[1].axhline(np.nanpercentile(d_seq, 75), color="red", ls="--", lw=0.8)
    axes[1].axhline(np.nanpercentile(d_seq, 25), color="blue", ls="--", lw=0.8)
    axes[1].set_title(r"Локальная память $\hat d(t)$ на тестовой выборке")
    axes[1].set_xlabel("наблюдение")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(rf"{FIG}\attention_regimes.png", dpi=150)
    print(rf"saved {FIG}\attention_regimes.png")


if __name__ == "__main__":
    main()
