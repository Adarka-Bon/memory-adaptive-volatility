"""Structural-break analysis of the rolling memory curve d(t).

Tests used:
  - CUSUM test on the rolling d(t) series (structural change detection);
  - difference-in-means between crisis and non-crisis windows;
  - Welch t-test and Mann-Whitney U with bootstrap CI.

Everything is run on the RV proxy, where long memory is strongest.
"""
import numpy as np
import pandas as pd
from scipy import stats

from config import RES_DIR, CRISES
from data_utils import all_assets_processed
from memory_estimators import compute_all

RNG = np.random.default_rng(42)


def cusum_test(x: np.ndarray) -> dict:
    """CUSUM of standardized residuals around the mean.

    Returns the max absolute deviation and its approximate location.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = x.mean()
    sd = x.std(ddof=1)
    if sd == 0:
        return {"cusum_max": 0.0, "cusum_loc": np.nan,
                "p_value": 1.0}
    resid = (x - mean) / sd
    s = np.cumsum(resid) / np.sqrt(n)
    k = np.max(np.abs(s))
    loc = int(np.argmax(np.abs(s)))
    # Brownian bridge approximation for p-value
    crit = 1.358
    p = 2 * np.exp(-2 * k ** 2) if k > 0 else 1.0
    p = min(max(p, 0.0), 1.0)
    return {"cusum_max": float(k), "cusum_loc": loc,
            "p_value": float(p), "significant": k > crit}


def bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 5000,
                      alpha: float = 0.05) -> tuple:
    diff = a.mean() - b.mean()
    boots = np.empty(n_boot)
    for i in range(n_boot):
        sa = RNG.choice(a, size=len(a), replace=True)
        sb = RNG.choice(b, size=len(b), replace=True)
        boots[i] = sa.mean() - sb.mean()
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return diff, lo, hi


def analyze_asset(name: str, df: pd.DataFrame) -> list:
    rows = []
    d = df["d"].to_numpy(dtype=float)
    dates = pd.to_datetime(df["end_date"]).to_numpy()

    cs = cusum_test(d)

    mask_crisis = np.zeros(len(dates), dtype=bool)
    for _, (a, b) in CRISES.items():
        mask_crisis |= (dates >= np.datetime64(a)) & (dates <= np.datetime64(b))
    mask_calm = ~mask_crisis

    if mask_crisis.sum() >= 3 and mask_calm.sum() >= 3:
        dc, dk = d[mask_crisis], d[mask_calm]
        t_stat, t_p = stats.ttest_ind(dc, dk, equal_var=False)
        u_stat, u_p = stats.mannwhitneyu(dc, dk, alternative="two-sided")
        diff, lo, hi = bootstrap_diff_ci(dc, dk)
        rows.append({
            "asset": name, "cusum_max": cs["cusum_max"],
            "cusum_p": cs["p_value"], "cusum_sig": cs["significant"],
            "n_crisis": int(mask_crisis.sum()),
            "n_calm": int(mask_calm.sum()),
            "d_crisis_mean": dc.mean(), "d_calm_mean": dk.mean(),
            "diff": diff, "ci_lo": lo, "ci_hi": hi,
            "t_p": t_p, "u_p": u_p,
        })
    return rows


def main():
    processed = all_assets_processed()
    results = compute_all(processed)

    all_rows = []
    for name, res in results.items():
        rv = res["rv"]
        if rv.empty:
            continue
        all_rows.extend(analyze_asset(name, rv))

    out = pd.DataFrame(all_rows)
    out.to_csv(RES_DIR / "structural_breaks.csv", index=False)
    print(out.to_string(index=False))

    print("\n--- Summary ---")
    print(f"Assets with significant CUSUM break: "
          f"{int(out['cusum_sig'].sum())}/{len(out)}")
    sig_diff = out[(out['ci_lo'] > 0) | (out['ci_hi'] < 0)]
    print(f"Assets with significant crisis vs calm difference: "
          f"{len(sig_diff)}/{len(out)}")
    print(f"Mean d in crisis: {out['d_crisis_mean'].mean():.3f}, "
          f"calm: {out['d_calm_mean'].mean():.3f}")


if __name__ == "__main__":
    main()
