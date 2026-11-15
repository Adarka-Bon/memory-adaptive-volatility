"""Build compact figure and summary tables for the ITS paper (2 pages)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from config import FIG_DIR, RES_DIR, CRISES
from data_utils import all_assets_processed
from memory_estimators import compute_all


def paper_figure():
    processed = all_assets_processed()
    results = compute_all(processed)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))

    themes = {
        "stocks": (["META", "BA", "JPM"], ["#1f77b4", "#ff7f0e", "#2ca02c"]),
        "crypto": (["BTC", "ETH"], ["#d62728", "#9467bd"]),
        "fx": (["EURUSD", "USDJPY", "GBPUSD"], ["#8c564b", "#e377c2", "#7f7f7f"]),
    }

    for name, color in [(n, c) for _, (ns, cs) in themes.items()
                        for n, c in zip(ns, cs)]:
        r = results[name]["rv"]
        if r.empty:
            continue
        x = pd.to_datetime(r["end_date"])
        axes[0].plot(x, r["d"], color=color, lw=1, label=name,
                     marker="o", ms=1.8)
        axes[1].plot(x, r["H"], color=color, lw=1, label=name,
                     marker="o", ms=1.8)

    for ax, ttl in zip(axes, [r"GPH $\hat d(t)$", r"Hurst $H(t)$"]):
        for _, (a, b) in CRISES.items():
            ax.axvspan(pd.Timestamp(a), pd.Timestamp(b),
                       color="red", alpha=0.10, lw=0)
        ax.set_title(ttl, fontsize=10)
        ax.grid(alpha=0.3, lw=0.4)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(labelsize=8)

    axes[0].axhline(0.5, color="gray", ls="--", lw=0.7)
    axes[1].axhline(0.5, color="gray", ls="--", lw=0.7)
    axes[0].legend(fontsize=6, ncol=2, loc="lower left")
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    out = FIG_DIR / "paper_memory.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def summary_tables():
    s = pd.read_csv(RES_DIR / "rolling_summary.csv")
    rv = s[s["proxy"] == "rv"].copy()
    rv = rv[["asset", "d_mean", "d_std", "d_min", "d_max", "H_mean"]]
    rv.to_csv(RES_DIR / "table_memory.csv", index=False)

    h = pd.read_csv(RES_DIR / "har_memory_comparison.csv")
    h = h[["asset", "qlike_base", "qlike_ext", "qlike_gain_%",
           "mse_gain_%"]]
    h.to_csv(RES_DIR / "table_har.csv", index=False)

    b = pd.read_csv(RES_DIR / "structural_breaks.csv")
    b = b[["asset", "cusum_p", "d_crisis_mean", "d_calm_mean", "t_p"]]
    b.to_csv(RES_DIR / "table_breaks.csv", index=False)

    print("\n=== Table memory (RV) ===")
    print(rv.to_string(index=False))
    print("\n=== Table HAR ===")
    print(h.to_string(index=False))
    print("\n=== Table breaks ===")
    print(b.to_string(index=False))


if __name__ == "__main__":
    paper_figure()
    summary_tables()
