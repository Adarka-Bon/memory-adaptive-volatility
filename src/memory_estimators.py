"""Rolling long-memory estimators: R/S (Hurst) and GPH (fractional d).

Applied to log-returns and volatility proxies (|r|, r^2, RV) with a sliding
window, producing time-varying H(t) and d(t) curves.
"""
import numpy as np
import pandas as pd

from config import ROLL_WINDOW, ROLL_STEP


def hurst_rs(x: np.ndarray, min_chunk: int = 10) -> float:
    """Hurst exponent via classical rescaled range R/S analysis."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2 * min_chunk:
        return np.nan

    sizes = []
    m = min_chunk
    while m <= n // 2:
        sizes.append(m)
        m = int(m * 1.5) if int(m * 1.5) > m + 1 else m + 1
    sizes = sorted(set(sizes))
    if len(sizes) < 3:
        return np.nan

    rs_means = []
    for m in sizes:
        num_seg = n // m
        vals = []
        for i in range(num_seg):
            seg = x[i * m:(i + 1) * m]
            seg = seg - seg.mean()
            s = np.sqrt(np.mean(seg ** 2))
            if s == 0:
                continue
            z = np.cumsum(seg)
            r = z.max() - z.min()
            vals.append(r / s)
        if vals:
            rs_means.append((m, np.mean(vals)))

    if len(rs_means) < 3:
        return np.nan

    logs = np.log([v[0] for v in rs_means])
    logrs = np.log([v[1] for v in rs_means])
    slope = np.polyfit(logs, logrs, 1)[0]
    return float(slope)


def periodogram(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    freqs = 2.0 * np.pi * np.arange(1, n // 2 + 1) / n
    fft = np.fft.rfft(x)
    per = (np.abs(fft) ** 2) / (2.0 * np.pi * n)
    return freqs, per[1:n // 2 + 1]


def gph_d(x: np.ndarray, power: float = 0.5) -> tuple:
    """GPH estimate of fractional integration parameter d.

    Regresses log periodogram on log(4 sin^2(lambda/2)) over m = n^power
    low-frequency ordinates. Returns (d, std_err).
    """
    n = len(x)
    m = int(n ** power)
    if m < 5:
        return np.nan, np.nan
    freqs, per = periodogram(x)
    lam = freqs[:m]
    per = per[:m]
    per = np.where(per <= 0, np.nan, per)
    mask = ~np.isnan(per)
    lam, per = lam[mask], per[mask]
    if len(lam) < 5:
        return np.nan, np.nan

    y = np.log(per)
    xreg = np.log(4.0 * np.sin(lam / 2.0) ** 2)
    slope, intercept = np.polyfit(xreg, y, 1)
    d = -slope / 2.0

    resid = y - (slope * xreg + intercept)
    dof = len(lam) - 2
    if dof <= 0:
        return d, np.nan
    s2 = np.sum(resid ** 2) / dof
    xvar = np.sum((xreg - xreg.mean()) ** 2)
    se_slope = np.sqrt(s2 / xvar) if xvar > 0 else np.nan
    se_d = se_slope / 2.0
    return float(d), float(se_d)


def rolling_memory(series: pd.Series, window: int = ROLL_WINDOW,
                   step: int = ROLL_STEP) -> pd.DataFrame:
    vals = series.to_numpy(dtype=float)
    n = len(vals)
    rows = []
    for start in range(0, n - window + 1, step):
        seg = vals[start:start + window]
        if np.isnan(seg).any():
            continue
        h = hurst_rs(seg)
        d, se = gph_d(seg)
        end_idx = start + window - 1
        rows.append({
            "start_idx": start,
            "end_idx": end_idx,
            "start_date": series.index[start],
            "end_date": series.index[end_idx],
            "H": h,
            "d": d,
            "d_se": se,
            "t": d / se if se and se > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def compute_all(processed: dict, window: int = ROLL_WINDOW,
                step: int = ROLL_STEP) -> dict:
    out = {}
    for name, df in processed.items():
        s = df.set_index("date")
        res = {}
        res["log_ret"] = rolling_memory(s["log_ret"], window, step)
        res["abs_ret"] = rolling_memory(s["abs_ret"], window, step)
        res["rv"] = rolling_memory(s["rv"], window, step)
        out[name] = res
    return out


if __name__ == "__main__":
    from data_utils import all_assets_processed

    processed = all_assets_processed()
    results = compute_all(processed)
    for name, res in results.items():
        r = res["rv"]
        print(f"{name:8s} rv: n_windows={len(r):3d} "
              f"d_mean={r['d'].mean():+.3f} d_std={r['d'].std():.3f} "
              f"H_mean={r['H'].mean():.3f}")
