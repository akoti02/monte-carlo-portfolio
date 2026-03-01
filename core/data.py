"""Data fetching and preprocessing module using yfinance."""

import os
import tempfile

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Redirect yfinance timezone cache to a temp directory to avoid
# SQLite "database is locked" errors from concurrent Streamlit reruns.
_cache_dir = os.path.join(tempfile.gettempdir(), "yf_tz_cache")
os.makedirs(_cache_dir, exist_ok=True)
yf.set_tz_cache_location(_cache_dir)


def fetch_stock_data(
    tickers: list[str],
    years: int = 5,
    end_date: datetime | None = None,
) -> pd.DataFrame:
    """Fetch adjusted close prices for given tickers.

    Returns a DataFrame with dates as index and tickers as columns.
    Raises ValueError if no valid data is returned.
    """
    if end_date is None:
        end_date = datetime.today()
    start_date = end_date - timedelta(days=years * 365)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    # Download each ticker individually to avoid partial-failure issues
    # where yfinance's bulk download silently drops tickers on cache lock.
    frames = {}
    failed = []
    for ticker in tickers:
        try:
            data = yf.download(
                ticker,
                start=start_str,
                end=end_str,
                auto_adjust=True,
                progress=False,
            )
            if data.empty:
                failed.append(ticker)
                continue
            if isinstance(data.columns, pd.MultiIndex):
                close = data["Close"].iloc[:, 0]
            else:
                close = data["Close"]
            frames[ticker] = close
        except Exception:
            failed.append(ticker)

    if not frames:
        raise ValueError("No data returned for any ticker. Check tickers and date range.")
    if failed:
        raise ValueError(f"No data for tickers: {failed}")

    prices = pd.DataFrame(frames)
    prices = prices.dropna(how="all").ffill().bfill()

    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns from price data."""
    log_ret = np.log(prices / prices.shift(1))
    # Drop only the first row (NaN from shift), keep everything else
    return log_ret.iloc[1:].fillna(0.0)


def fetch_benchmark(years: int = 5, end_date: datetime | None = None) -> pd.Series:
    """Fetch S&P 500 adjusted close prices as benchmark."""
    if end_date is None:
        end_date = datetime.today()
    start_date = end_date - timedelta(days=years * 365)

    data = yf.download(
        "^GSPC",
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )
    if data.empty:
        raise ValueError("Could not fetch S&P 500 data.")

    if isinstance(data.columns, pd.MultiIndex):
        return data["Close"]["^GSPC"].dropna()
    return data["Close"].dropna()
