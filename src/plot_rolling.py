"""Plot rolling d(t), H(t) curves with crisis shading."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from config import FIG_DIR, RES_DIR, CRISES
from data_utils import all_assets_processed
from memory_estimators import compute_all

GROUPS = {
    "stocks": ["META", "BA", "JPM", "AAPL", "MSFT"],
    "crypto": ["BTC", "ETH"],
    "fx": ["EURUSD", "USDJPY", "GBPUSD"],
}


def shade_crises(ax):
    for label, (a, b) in CRISES.items():
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="red",
                   alpha=0.08, lw=0)


def plot_group(names, results, group):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for name in names:
        r = results[name]["rv"]
        if r.empty:
            continue
        x = pd.to_datetime(r["end_date"])
        axes[0].plot(x, r["d"], marker="o", ms=2.5, lw=1, label=name)
        axes[1].plot(x, r["H"], marker="o", ms=2.5, lw=1, label=name)
    for ax, ttl in zip(axes, ["GPH $\\hat d(t)$ (RV)", "Hurst $H(t)$ (RV)"]):
        shade_crises(ax)
        ax.set_title(ttl, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=len(names))
    axes[0].axhline(0.5, color="gray", ls="--", lw=0.8)
    axes[1].axhline(0.5, color="gray", ls="--", lw=0.8)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    out = FIG_DIR / f"rolling_memory_{group}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def main():
    processed = all_assets_processed()
    results = compute_all(processed)

    for name, res in results.items():
        for proxy, r in res.items():
            if not r.empty:
                r.to_csv(RES_DIR / f"rolling_{name}_{proxy}.csv", index=False)

    for group, names in GROUPS.items():
        plot_group(names, results, group)

    summary = []
    for name, res in results.items():
        for proxy, r in res.items():
            if r.empty:
                continue
            summary.append({
                "asset": name, "proxy": proxy, "n": len(r),
                "d_mean": r["d"].mean(), "d_std": r["d"].std(),
                "d_min": r["d"].min(), "d_max": r["d"].max(),
                "H_mean": r["H"].mean(), "H_std": r["H"].std(),
            })
    s = pd.DataFrame(summary)
    s.to_csv(RES_DIR / "rolling_summary.csv", index=False)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
