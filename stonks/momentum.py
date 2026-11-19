"""Earnings-growth momentum strategy (long-only, 6-8 week holds).

Entry, all on the same day:
  * EPS rose quarter over quarter in each of the last `qoq_streak` reports
    (point in time: only reports already public count);
  * RSI(14) crosses above `rsi_level` (60) - momentum turning up;
  * optional: price above its 200-day average (`trend_filter`);
  * optional: the stock beat the market index over the last `rs_lookback`
    days (`rs_filter`, relative strength);
  * optional: the market index is above its 200-day average (pass `market`).

Exit: a stop `stop_pct` below entry, or after `hold` trading days.
Stops fill at the stop price, or at the open when the stock gaps through it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import earnings as er
from . import indicators as ind
from .swing import market_ok


@dataclass
class MomentumParams:
    qoq_streak: int = 2            # consecutive quarter-over-quarter EPS increases
    rsi_period: int = 14
    rsi_level: float = 60.0
    trend_filter: bool = False
    rs_filter: bool = False
    rs_lookback: int = 63          # ~3 months
    market_ma: int = 200
    stop_pct: float = 0.10
    hold: int = 40                 # trading days (~8 weeks)
    cost_bps: float = 10.0


def qoq_streak(prices: pd.DataFrame, earnings: pd.DataFrame) -> pd.Series:
    """Point-in-time count of consecutive quarter-over-quarter EPS increases."""
    e = er.normalize(earnings)
    e = e[e["eps_actual"].notna()].copy()
    up = (e["eps_actual"] > e["eps_actual"].shift(1)).astype(int)
    # streak length ending at each report
    streak = up.groupby((up == 0).cumsum()).cumsum()
    idx = prices.index
    side = np.where(e["timing"] == "bmo", "left", "right")
    react = np.array([idx.searchsorted(d, side=s) for d, s in zip(e["date"], side)], dtype=int)
    out = pd.Series(np.nan, index=idx)
    for r, s in zip(react, streak.to_numpy()):
        if r < len(idx):
            out.iloc[r] = s
    return out.ffill().fillna(0)


def signals(df: pd.DataFrame, earnings: pd.DataFrame, p: MomentumParams,
            market: pd.DataFrame | None = None) -> pd.Series:
    c = df["close"]
    r = ind.rsi(c, p.rsi_period)
    ok = (r > p.rsi_level) & (r.shift(1) <= p.rsi_level)
    ok &= qoq_streak(df, earnings) >= p.qoq_streak
    if p.trend_filter:
        ok &= c > ind.sma(c, 200)
    if market is not None:
        ok &= pd.Series(market_ok(market, df.index, p.market_ma), index=df.index)
        if p.rs_filter:
            mc = market["close"].reindex(df.index, method="ffill")
            ok &= c.pct_change(p.rs_lookback) > mc.pct_change(p.rs_lookback)
    return ok.fillna(False)


def run(df: pd.DataFrame, earnings: pd.DataFrame, p: MomentumParams | None = None,
        market: pd.DataFrame | None = None, ticker: str = "") -> pd.DataFrame:
    """Simulate; returns one row per trade (reason 'open' if still held)."""
    p = p or MomentumParams()
    sig = signals(df, earnings, p, market).to_numpy()
    o, l, c = (df[k].to_numpy(float) for k in ("open", "low", "close"))
    idx, cost = df.index, p.cost_bps / 10_000
    trades, i, n = [], 0, len(df)
    while i < n:
        if not sig[i]:
            i += 1
            continue
        entry, stop = c[i], c[i] * (1 - p.stop_pct)
        exit_i, price, reason = None, None, None
        for j in range(i + 1, min(i + p.hold, n - 1) + 1):
            if l[j] <= stop:
                exit_i, price, reason = j, min(o[j], stop), "stop"
                break
            if j - i >= p.hold:
                exit_i, price, reason = j, c[j], "time"
                break
        if exit_i is None:
            exit_i, price, reason = n - 1, c[-1], "open"
        trades.append(dict(ticker=ticker, entry_date=idx[i], entry_price=entry,
                           exit_date=idx[exit_i], exit_price=price, reason=reason,
                           bars_held=exit_i - i,
                           ret=price * (1 - cost) / (entry * (1 + cost)) - 1))
        i = exit_i + 1
    t = pd.DataFrame(trades)
    return t.rename(columns={"ret": "return"}) if not t.empty else t
