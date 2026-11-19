"""Standard-deviation band entries around the 50-day average, in a 200-day uptrend.

    pullback:  close crosses below SMA50 - k * std50   (buy the dip)
    breakout:  close crosses above SMA50 + k * std50   (buy the strength)

Both require close > SMA200 on the signal day; optionally the market index
above its own 200-day average. Exits via `momentum.simulate` (stop or hold).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import indicators as ind
from .momentum import simulate
from .swing import market_ok


@dataclass
class BandParams:
    side: str = "pullback"   # or "breakout"
    k: float = 2.0
    ma: int = 50
    trend_ma: int = 200
    market_ma: int = 200
    stop_pct: float = 0.10
    hold: int = 40
    cost_bps: float = 10.0


def signals(df: pd.DataFrame, p: BandParams, market: pd.DataFrame | None = None) -> pd.Series:
    c = df["close"]
    mid = ind.sma(c, p.ma)
    sd = c.rolling(p.ma, min_periods=p.ma).std(ddof=0)
    if p.side == "pullback":
        band = mid - p.k * sd
        hit = (c < band) & (c.shift(1) >= band.shift(1))
    elif p.side == "breakout":
        band = mid + p.k * sd
        hit = (c > band) & (c.shift(1) <= band.shift(1))
    else:
        raise ValueError(p.side)
    ok = hit & (c > ind.sma(c, p.trend_ma))
    if market is not None:
        ok &= pd.Series(market_ok(market, df.index, p.market_ma), index=df.index)
    return ok.fillna(False)


def run(df: pd.DataFrame, p: BandParams | None = None, market: pd.DataFrame | None = None,
        ticker: str = "") -> pd.DataFrame:
    p = p or BandParams()
    return simulate(df, signals(df, p, market).to_numpy(), p.stop_pct, p.hold, p.cost_bps, ticker)
