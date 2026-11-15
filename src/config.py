from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"

DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

START = "2019-01-01"
END = "2023-12-31"

ASSETS = {
    "stocks": {
        "META": "META",
        "BA": "BA",
        "JPM": "JPM",
        "AAPL": "AAPL",
        "MSFT": "MSFT",
    },
    "crypto": {
        "BTC-USD": "BTC",
        "ETH-USD": "ETH",
    },
    "fx": {
        "EURUSD=X": "EURUSD",
        "USDJPY=X": "USDJPY",
        "GBPUSD=X": "GBPUSD",
    },
}

CRISES = {
    "COVID-19": ("2020-02-19", "2020-04-30"),
    "Rate hikes 2022": ("2022-01-03", "2022-10-31"),
    "Banking stress 2023": ("2023-03-01", "2023-05-31"),
}

ROLL_WINDOW = 500
ROLL_STEP = 25
