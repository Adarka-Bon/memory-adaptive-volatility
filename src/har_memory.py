"""Does local (time-varying) memory improve HAR-RV volatility forecasts?

HAR-RV baseline:
    RV_{t+1} = b0 + b1*RV_t + b2*RV_t^(5) + b3*RV_t^(22)

Extended:
    + b4*d_hat_t + b5*delta_d_t   (local memory level and its change)

Walk-forward (expanding) OLS with log-RV target, evaluated by
MSE and QLIKE on the original RV scale.
"""
import numpy as np
import pandas as pd

from config import RES_DIR, ROLL_WINDOW
from data_utils import all_assets_processed
from memory_estimators import rolling_memory


def har_features(rv: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame(index=rv.index)
    df["rv_d"] = rv
    df["rv_w"] = rv.rolling(5).mean()
    df["rv_m"] = rv.rolling(22).mean()
    return df


def qlike(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.clip(y_pred, 1e-12, None)
    ratio = y_true / y_pred
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def walk_forward(X: np.ndarray, y: np.ndarray, min_train: int = 300):
    preds = np.full(len(y), np.nan)
    for t in range(min_train, len(y)):
        Xtr, ytr = X[:t], y[:t]
        Xte = X[t:t + 1]
        beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
        preds[t] = float(Xte @ beta)
    return preds


def align_memory(asset_df: pd.DataFrame, rv: pd.Series,
                 window=ROLL_WINDOW) -> pd.DataFrame:
    """Map rolling memory windows onto the daily timeline (as-of end date)."""
    roll = rolling_memory(rv, window=window, step=5)
    roll = roll.set_index(pd.to_datetime(roll["end_date"]))
    mem = roll[["d"]].rename(columns={"d": "d_local"})
    mem = mem.reindex(asset_df.set_index("date").index, method="ffill")
    mem["delta_d"] = mem["d_local"].diff()
    return mem


def run_asset(name, group_df, window=ROLL_WINDOW):
    df = group_df.set_index("date").copy()
    rv = df["rv"]
    feats = har_features(rv)
    mem = align_memory(group_df, rv, window)
    data = feats.join(mem)

    y = np.log(rv.shift(-1))
    data["y"] = y
    data = data.dropna()
    if len(data) < 400:
        return None

    base_cols = ["rv_d", "rv_w", "rv_m"]
    ext_cols = base_cols + ["d_local", "delta_d"]

    def build(cols):
        X = np.column_stack([np.ones(len(data))] + [data[c].to_numpy() for c in cols])
        return X

    yv = data["y"].to_numpy()
    min_train = 300

    Xb = build(base_cols)
    Xe = build(ext_cols)
    pb = walk_forward(Xb, yv, min_train)
    pe = walk_forward(Xe, yv, min_train)

    mask = ~np.isnan(pb) & ~np.isnan(pe)
    y_true_rv = np.exp(yv[mask])
    rv_prev = data["rv_d"].to_numpy()[mask]

    # forecasts in log-space -> exp; naive = previous RV
    pred_b_rv = np.exp(pb[mask])
    pred_e_rv = np.exp(pe[mask])

    out = {
        "asset": name, "n_test": int(mask.sum()),
        "mse_base": float(np.mean((pred_b_rv - y_true_rv) ** 2)),
        "mse_ext": float(np.mean((pred_e_rv - y_true_rv) ** 2)),
        "qlike_base": qlike(y_true_rv, pred_b_rv),
        "qlike_ext": qlike(y_true_rv, pred_e_rv),
        "qlike_naive": qlike(y_true_rv, rv_prev),
    }
    out["mse_gain_%"] = 100 * (out["mse_base"] - out["mse_ext"]) / out["mse_base"]
    out["qlike_gain_%"] = 100 * (out["qlike_base"] - out["qlike_ext"]) / out["qlike_base"]
    return out


def main():
    processed = all_assets_processed()
    from config import ASSETS
    groups = {n: g for g, m in ASSETS.items() for n in m.values()}

    rows = []
    for name, df in processed.items():
        try:
            r = run_asset(name, df)
            if r:
                rows.append(r)
        except Exception as e:
            print(f"!! {name}: {e}")

    out = pd.DataFrame(rows)
    out.to_csv(RES_DIR / "har_memory_comparison.csv", index=False)
    print(out.to_string(index=False))
    print("\n--- Summary ---")
    print(f"Mean MSE gain:   {out['mse_gain_%'].mean():+.2f}%")
    print(f"Mean QLIKE gain: {out['qlike_gain_%'].mean():+.2f}%")
    print(f"Assets improved (QLIKE): {(out['qlike_gain_%'] > 0).sum()}/{len(out)}")
    print(f"Assets improved (MSE):   {(out['mse_gain_%'] > 0).sum()}/{len(out)}")


if __name__ == "__main__":
    main()
