"""Download daily price data for stocks, crypto and FX (2019-2023)."""
import sys
import pandas as pd
import yfinance as yf

from config import DATA_DIR, START, END, ASSETS


def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if c[0] else c[1] for c in df.columns]
    return df


def download(ticker, name, group):
    df = yf.download(ticker, start=START, end=END, interval="1d",
                     auto_adjust=False, progress=False)
    if df is None or df.empty:
        print(f"  !! empty: {ticker}")
        return None
    df = flatten_columns(df)
    if "Close" not in df.columns:
        print(f"  !! no close: {ticker} cols={list(df.columns)}")
        return None
    out = pd.DataFrame(index=df.index)
    out["close"] = df["Close"]
    out["open"] = df.get("Open")
    out["high"] = df.get("High")
    out["low"] = df.get("Low")
    out["volume"] = df.get("Volume")
    out["asset"] = name
    out["group"] = group
    out.index.name = "date"
    out = out.dropna(subset=["close"])
    print(f"  ok: {ticker} -> {name} ({len(out)} rows)")
    return out


def main():
    frames = []
    for group, mapping in ASSETS.items():
        print(f"[{group}]")
        for ticker, name in mapping.items():
            try:
                df = download(ticker, name, group)
                if df is not None:
                    frames.append(df)
            except Exception as e:
                print(f"  !! error {ticker}: {e}")

    if not frames:
        print("No data downloaded.")
        sys.exit(1)

    all_df = pd.concat(frames)
    all_df = all_df.reset_index()
    all_df["date"] = pd.to_datetime(all_df["date"]).dt.tz_localize(None)
    all_df = all_df.sort_values(["asset", "date"])

    for (grp, name), sub in all_df.groupby(["group", "asset"]):
        path = DATA_DIR / "raw" / grp
        path.mkdir(parents=True, exist_ok=True)
        sub = sub.sort_values("date")
        sub.to_csv(path / f"{name}.csv", index=False)
        print(f"saved {grp}/{name}.csv ({len(sub)})")


if __name__ == "__main__":
    main()
