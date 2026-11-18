"""Loading OHLCV price data."""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED = ["open", "high", "low", "close", "volume"]


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case columns, prefer adjusted close, sort by date."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    if "adj_close" in df.columns:
        ratio = df["adj_close"] / df["close"]
        for col in ("open", "high", "low"):
            df[col] = df[col] * ratio
        df["close"] = df["adj_close"]
        df = df.drop(columns="adj_close")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df[REQUIRED].sort_index()


def load_csv(path: str, date_col: str | None = None) -> pd.DataFrame:
    """Load a CSV with a date column and OHLCV columns (e.g. a Yahoo/Stooq export)."""
    raw = pd.read_csv(path)
    date_col = date_col or next(c for c in raw.columns if c.lower() in ("date", "datetime", "timestamp"))
    raw[date_col] = pd.to_datetime(raw[date_col])
    return normalize(raw.set_index(date_col))


def load_yahoo(ticker: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Download daily bars via yfinance (`pip install yfinance`)."""
    import yfinance as yf

    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return normalize(raw)


def synthetic(n: int = 1000, seed: int = 0, start_price: float = 100.0) -> pd.DataFrame:
    """Geometric random walk OHLCV data for testing and demos."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.015, n)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start_price], close[:-1]]) * (1 + rng.normal(0, 0.002, n))
    spread = np.abs(rng.normal(0, 0.008, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.integers(500_000, 5_000_000, n).astype(float)
    idx = pd.bdate_range("2020-01-01", periods=n, name="date")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx
    )
