"""Common utilities: load prices, build log-returns and realized volatility."""
import numpy as np
import pandas as pd

from config import DATA_DIR


def load_asset(group: str, name: str) -> pd.DataFrame:
    path = DATA_DIR / "raw" / group / f"{name}.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_price"] = np.log(df["close"])
    df["log_ret"] = df["log_price"].diff()
    df["ret"] = df["close"].pct_change()
    df["abs_ret"] = df["log_ret"].abs()
    df["sq_ret"] = df["log_ret"] ** 2
    return df


def add_rv(df: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Realized volatility proxy from daily squared returns (rolling sum)."""
    df = df.copy()
    df["rv"] = df["sq_ret"].rolling(window, min_periods=window).sum()
    df["rv"] = np.sqrt(df["rv"])
    df["rv_log"] = np.log(df["rv"].replace(0, np.nan))
    return df


def load_processed(group: str, name: str, rv_window: int = 21) -> pd.DataFrame:
    df = load_asset(group, name)
    df = add_returns(df)
    df = add_rv(df, window=rv_window)
    return df.dropna(subset=["rv"]).reset_index(drop=True)


def all_assets_processed(rv_window: int = 21) -> dict:
    from config import ASSETS

    out = {}
    for group, mapping in ASSETS.items():
        for key, name in mapping.items():
            try:
                out[name] = load_processed(group, name, rv_window=rv_window)
            except FileNotFoundError:
                print(f"missing: {group}/{name}")
    return out


if __name__ == "__main__":
    data = all_assets_processed()
    for name, df in data.items():
        print(f"{name:8s} n={len(df):5d} "
              f"rv_mean={df['rv'].mean():.5f} "
              f"ret_mean={df['log_ret'].mean():.5f}")
