# -*- coding: utf-8 -*-
"""Regenerate the paper figures.

Two language variants are produced from the same results/*.csv:
  * Russian labels -> written to figures/ and paper/ (used by the Russian referat
    and by the Russian ITS paper bondarenko.tex);
  * English labels -> written to paper/ with the _en suffix (used by the English
    IEEE paper bondarenko_ieee.tex).

Usage:  python make_figures_en.py [ru|en]
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

ROOT = r"C:\diploma\magistracy"
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
PAPER = os.path.join(ROOT, "paper")
os.makedirs(FIG, exist_ok=True)

ASSETS = ["META", "BA", "JPM", "AAPL", "MSFT", "BTC", "ETH", "EURUSD", "USDJPY", "GBPUSD"]
CRISES = {
    "COVID-19": ("2020-02-19", "2020-04-30"),
    "Rate hikes 2022": ("2022-01-03", "2022-10-31"),
    "Banking stress 2023": ("2023-03-01", "2023-05-31"),
}
C_LOG, C_ABS, C_RV = "#4FC3F7", "#7E57C2", "#26A69A"
C_CALM, C_CRISIS = "#8FA8B8", "#E8746A"
C_LOW, C_HIGH = "#3DBE95", "#F2736B"

L = {}

def set_lang(lang):
    if lang == "ru":
        L.update(
            loc_title="Локализация длинной памяти: доходность и волатильность",
            loc_leg=["доходность", "доходность|", "волатильность RV"],
            loc_yl="оценка $d$ (GPH)",
            cri_title="Сила памяти в кризисные и спокойные периоды (* — значимый сдвиг)",
            cri_leg=["спокойные периоды", "кризисные периоды"],
            cri_yl="среднее $\\hat d$",
            har_title="Улучшение прогноза HAR-RV при добавлении локальной памяти",
            har_yl="прирост, %",
            att_title="Деградация механизмов внимания при усилении памяти",
            att_leg=["низкая память", "высокая память"],
            sup=("Скользящие оценки $\\hat d(t)$ и $H(t)$ для реализованной волатильности "
                 "(красным выделены кризисные периоды)"),
            ttl1="GPH $\\hat d(t)$ (RV)", ttl2="Hurst $H(t)$ (RV)", xlab="Дата",
        )
    else:
        L.update(
            loc_title="Localisation of long memory: returns versus volatility",
            loc_leg=["Logarithmic returns", "Absolute returns", "Realised volatility"],
            loc_yl="GPH estimate $d$",
            cri_title="Local memory in calm and crisis periods (* significant shift)",
            cri_leg=["Calm periods", "Crisis periods"],
            cri_yl="Mean local memory $\\hat d$",
            har_title="Improvement of HAR-RV after adding the local memory features",
            har_yl="Forecast improvement, %",
            att_title="Degradation of attention mechanisms when memory strengthens",
            att_leg=["Low-memory regime", "High-memory regime"],
            sup=("Rolling memory estimates for realised volatility "
                 "(shaded: COVID-19, 2022 rate hikes, 2023 banking stress)"),
            ttl1="GPH $\\hat d(t)$ (realised volatility)",
            ttl2="Hurst $H(t)$ (realised volatility)", xlab="Date",
        )

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 160, "savefig.dpi": 160})


def save(fig, name, targets):
    for d in targets:
        fig.savefig(os.path.join(d, name), bbox_inches="tight")
    plt.close(fig)
    print("  saved", name, "->", ", ".join(os.path.basename(t) for t in targets))


def build(targets, suffix):
    # ---------------------------------------------------------- 1. localisation
    rs = pd.read_csv(os.path.join(RES, "rolling_summary.csv"))
    piv = rs.pivot_table(index="asset", columns="proxy", values="d_mean")
    piv = piv.reindex([a for a in ASSETS if a in piv.index])
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    w, x = 0.26, np.arange(len(piv))
    for i, (key, col) in enumerate([("log_ret", C_LOG), ("abs_ret", C_ABS), ("rv", C_RV)]):
        vals = [piv.loc[a, key] if key in piv.columns and a in piv.index else np.nan
                for a in piv.index]
        ax.bar(x + (i - 1) * w, vals, w, label=L["loc_leg"][i], color=col,
               edgecolor="white", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(piv.index, rotation=45, ha="right")
    ax.set_ylabel(L["loc_yl"]); ax.set_title(L["loc_title"])
    ax.legend(fontsize=8, ncol=3, framealpha=0.9); ax.axhline(0, color="gray", lw=0.8)
    save(fig, "fig_memory_localization%s.png" % suffix, targets)

    # ---------------------------------------------------------- 2. crisis vs calm
    tb = pd.read_csv(os.path.join(RES, "table_breaks.csv")).set_index("asset")
    tb = tb.reindex([a for a in ASSETS if a in tb.index])
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(len(tb))
    ax.bar(x - 0.2, tb["d_calm_mean"], 0.4, label=L["cri_leg"][0], color=C_CALM,
           edgecolor="white", linewidth=0.4)
    ax.bar(x + 0.2, tb["d_crisis_mean"], 0.4, label=L["cri_leg"][1], color=C_CRISIS,
           edgecolor="white", linewidth=0.4)
    for i, (a, r) in enumerate(tb.iterrows()):
        if r["t_p"] < 0.05:
            ax.text(i, max(r["d_calm_mean"], r["d_crisis_mean"]) + 0.012, "*",
                    ha="center", va="bottom", fontsize=13)
    ax.set_xticks(x); ax.set_xticklabels(tb.index, rotation=45, ha="right")
    ax.set_ylabel(L["cri_yl"]); ax.set_title(L["cri_title"])
    ax.set_ylim(0, tb[["d_calm_mean", "d_crisis_mean"]].to_numpy().max() * 1.18)
    ax.legend(fontsize=8, framealpha=0.95, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig_crisis%s.png" % suffix, targets)

    # ---------------------------------------------------------- 3. HAR-RV gains
    th = pd.read_csv(os.path.join(RES, "table_har.csv")).set_index("asset")
    th = th.reindex([a for a in ASSETS if a in th.index])
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(len(th))
    ax.bar(x - 0.2, th["mse_gain_%"], 0.4, label="MSE", color=C_LOG,
           edgecolor="white", linewidth=0.4)
    ax.bar(x + 0.2, th["qlike_gain_%"], 0.4, label="QLIKE", color=C_RV,
           edgecolor="white", linewidth=0.4)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(th.index, rotation=45, ha="right")
    ax.set_ylabel(L["har_yl"]); ax.set_title(L["har_title"])
    ax.legend(fontsize=8, framealpha=0.9)
    save(fig, "fig_har_improvement%s.png" % suffix, targets)

    # ---------------------------------------------------------- 4. attention regimes
    ar = pd.read_csv(os.path.join(RES, "attention_regimes.csv"))
    order = ["Standard", "ProbSparse", "MultiFeature"]
    low = {r["mechanism"]: r["RMSE"] for _, r in ar.iterrows() if r["regime"] == "low-memory"}
    high = {r["mechanism"]: r["RMSE"] for _, r in ar.iterrows() if r["regime"] == "high-memory"}
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(len(order))
    ax.bar(x - 0.2, [low[m] for m in order], 0.4, label=L["att_leg"][0], color=C_LOW,
           edgecolor="white", linewidth=0.4)
    ax.bar(x + 0.2, [high[m] for m in order], 0.4, label=L["att_leg"][1], color=C_HIGH,
           edgecolor="white", linewidth=0.4)
    for i, m in enumerate(order):
        g = (high[m] / low[m] - 1) * 100
        ax.text(i, max(low[m], high[m]) + 0.04, "+%d%%" % round(g), ha="center",
                va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(order)
    ax.set_ylabel("RMSE"); ax.set_title(L["att_title"])
    ax.set_ylim(0, max(max(low.values()), max(high.values())) * 1.22)
    ax.legend(fontsize=8, framealpha=0.9)
    save(fig, "fig_attention_deg%s.png" % suffix, targets)

    # ---------------------------------------------------------- 5. rolling d(t), H(t)
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.0), sharex=True)
    for a in ASSETS:
        f = os.path.join(RES, "rolling_%s_rv.csv" % a)
        if not os.path.exists(f):
            continue
        r = pd.read_csv(f)
        if r.empty:
            continue
        xd = pd.to_datetime(r["end_date"])
        axes[0].plot(xd, r["d"], marker="o", ms=2.5, lw=1.0, label=a)
        axes[1].plot(xd, r["H"], marker="o", ms=2.5, lw=1.0, label=a)
    for ax, ttl in zip(axes, [L["ttl1"], L["ttl2"]]):
        for _, (a, b) in CRISES.items():
            ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="red", alpha=0.10, lw=0)
        ax.set_title(ttl, fontsize=10); ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5, ncol=5, loc="upper left")
        ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right")
    axes[1].set_xlabel(L["xlab"])
    fig.suptitle(L["sup"], fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "paper_memory%s.png" % suffix, targets)


if __name__ == "__main__":
    lang = (sys.argv[1] if len(sys.argv) > 1 else "en").lower()
    set_lang(lang)
    if lang == "ru":
        build([FIG, PAPER], "")          # Russian versions keep the original names
    else:
        build([PAPER], "_en")            # English versions for the IEEE paper only
    print("done", lang)
