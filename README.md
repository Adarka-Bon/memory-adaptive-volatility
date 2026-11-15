# Adaptive Long Memory for Volatility Forecasting

**Transformer architectures with an adaptive long-memory mechanism for forecasting financial
time-series volatility and assessing market risk.**

Master's thesis, Department of Probability Theory and Mathematical Statistics,
Faculty of Applied Mathematics and Informatics, Belarusian State University (BSU).
Supervisor: S. N. Stalevskaya, PhD in Physics and Mathematics.

| | |
|---|---|
| **Thesis text** | [`docs/thesis_Bondarenko.docx`](docs/thesis_Bondarenko.docx) · [PDF](docs/thesis_Bondarenko.pdf) |
| **Slides** | [`docs/slides_referat.pptx`](docs/slides_referat.pptx) · [PDF](docs/slides_referat.pdf) |
| **Interactive site** | **<https://adarka-bon.github.io/>** |
| **Notebook with results** | [`notebooks/01_results_and_interpretation.ipynb`](notebooks/01_results_and_interpretation.ipynb) |
| **Conference paper** | [`paper/`](paper) (CTDA'2026, pp. 462–465) |

## Research question

Volatility of financial series shows **long memory**: autocorrelations decay as a power law, so
the influence of the past survives over long horizons. The central finding of this work is that
this memory is **non-stationary** — it "breathes" over time and collapses during crises. A local
estimate of the memory strength `d̂(t)` therefore carries independent predictive value and can
steer the attention of a transformer — the proposed **Memory-Adaptive Attention**.

## Key results

| Result | Value |
|---|---|
| Memory in realized volatility | d ∈ [0.42; 0.56], H ≈ 0.90–0.95 |
| Memory in returns | d ≈ 0 (absent) |
| Structural breaks in d(t) | 7 of 10 assets (CUSUM test) |
| Memory during crises | d: 0.52 → 0.45 |
| HAR-RV + local memory | MSE +7.1 % (10/10 assets), QLIKE +1.8 % (8/10) |
| ProbSparse degradation | RMSE +143 % in the high-memory regime |

Data: daily quotes of 10 assets over 2019–2023 — stocks (META, BA, JPM, AAPL, MSFT),
crypto (BTC, ETH) and FX pairs (EURUSD, USDJPY, GBPUSD), covering three crisis episodes
(COVID-19 2020, rate-hike cycle 2022, banking stress 2023).

## The notebook

`notebooks/01_results_and_interpretation.ipynb` rebuilds **every figure and table of the thesis
from the committed CSV results** and explains what each number means — from the localization of
long memory up to the VaR/ES part. It runs in seconds and needs no heavy computation:

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_results_and_interpretation.ipynb
```

## Repository layout

```
memory-adaptive-volatility/
├── notebooks/             # executed notebook: results + interpretation
├── src/                    # analysis pipeline
│   ├── config.py               # assets, crisis windows, estimator settings
│   ├── download_data.py        # daily quotes via yfinance
│   ├── data_utils.py           # returns and realized volatility
│   ├── memory_estimators.py    # rolling R/S and GPH estimators
│   ├── structural_breaks.py    # CUSUM test, Welch test, bootstrap
│   ├── har_memory.py           # HAR-RV vs HAR-RV with local memory
│   ├── make_figures_en.py      # figures and report tables
│   └── make_paper_assets.py    # summary figures and tables
├── src_attention/          # attention mechanism regimes (Standard, ProbSparse, MultiFeature)
├── data/                   # daily quotes of the 10 assets, 2019–2023
├── results/                # CSV results of every experiment
├── figures/                # report and notebook figures
├── site/                   # personal site (interactive demos)
├── docs/                   # thesis text and slide deck
└── paper/                  # conference materials, CTDA'2026
```

## Reproducing the results

```bash
pip install -r requirements.txt

python src/download_data.py                 # download quotes
python src/plot_rolling.py                  # rolling d(t), H(t) + summary
python src/structural_breaks.py             # CUSUM diagnostics and regimes
python src/har_memory.py                    # HAR-RV experiment
python src_attention/attention_regimes.py   # attention regimes
python src/make_paper_assets.py             # figures and tables
```

Open `site/index.html` locally for the interactive demos: attention-head maps with a memory-strength
slider, a seven-model comparison across four metrics, and an interactive VaR/ES calculator.

## Publication

1. Bondarenko, D. D. *Transformer architectures for forecasting time series with long memory* /
   D. D. Bondarenko // Computer Technologies and Data Analysis (CTDA'2026) : proceedings of the V
   International Scientific and Practical Conference, Minsk, 23–24 Apr. 2026 / BSU ; ed. by
   V. V. Skakun (ed.-in-chief), N. N. Yatskov, V. V. Grinyov. – Minsk : BSU, 2026. – P. 462–465.

## License

MIT — see [LICENSE](LICENSE).
