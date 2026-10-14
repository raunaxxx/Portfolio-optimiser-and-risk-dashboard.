import os
import pandas as pd
import yfinance as yf

# Default stocks: 5 large NSE companies from different sectors
DEFAULT_TICKERS = [
    "RELIANCE.NS",   # Energy / Conglomerate
    "TCS.NS",        # IT
    "HDFCBANK.NS",   # Banking
    "INFY.NS",       # IT
    "ITC.NS",        # FMCG
]

# Nifty 50 index, used as our benchmark
BENCHMARK = "^NSEI"

# Path to data/raw (works no matter where you run the code from)
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def download_prices(tickers=None, start="2019-01-01", end="2025-01-01", save=True):
    """
    Download daily closing prices for the given stocks.
    Returns a DataFrame: rows = dates, columns = tickers.
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS
    if isinstance(tickers, str):
        tickers = [tickers]

    # Download adjusted closing prices from Yahoo Finance
    data = yf.download(tickers, start=start, end=end,
                       auto_adjust=True, progress=False)["Close"]

    # Clean: drop empty rows, fill small gaps, drop anything still missing
    data = data.dropna(how="all").ffill().dropna()

    if save:
        os.makedirs(RAW_DIR, exist_ok=True)
        data.to_csv(os.path.join(RAW_DIR, "prices.csv"))

    return data


def load_saved_prices():
    """Load prices from the saved CSV so we don't download every time."""
    path = os.path.join(RAW_DIR, "prices.csv")
    return pd.read_csv(path, index_col=0, parse_dates=True)


if __name__ == "__main__":
    prices = download_prices()
    print(prices.head())
    print("\nShape:", prices.shape)
    print("Missing values:", prices.isna().sum().sum())