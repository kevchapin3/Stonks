"""Candlestick swing-trading system for 6-8 week holds.

Setup rules (long-only, the default for mid caps):

1. **Trend filter**: close above the 200-day SMA and 50-day SMA above the
   200-day. Bullish reversal candles are taken as *pullback entries inside a
   larger uptrend*, not as bottom-picking in a downtrend.
2. **Pattern**: one of the bullish reversal patterns in `candlesticks.BULLISH`
   fires after a short-term pullback.
3. **Volume**: pattern-bar volume >= `vol_mult` x its 20-day average. Mid caps
   have thinner books than large caps, so volume confirmation filters out
   many false patterns.
4. **Confirmation**: within `confirm_bars` bars a close above the pattern
   bar's high (Nison's "wait for confirmation"). Entry is at that close.

Exits, whichever comes first:

* Initial stop: pattern low minus `stop_atr` x ATR (capped at `max_risk_atr`
  ATRs below entry).
  Or, with `stop_pct` set, a fixed stop that far below entry (0.10 = 10%).
* Chandelier trailing stop: highest close since entry minus `trail_atr` x ATR
  (set `trail_atr=None` to turn it off).
  It only tightens.
* Bearish reversal pattern after `min_hold` bars.
* Time stop at `max_hold` bars (40 trading days is about 8 weeks).

Earnings rule (when an earnings table is passed to `run`, see `earnings.py`):

* Enter only within `post_window` bars after a report that beat the EPS
  estimate (`require_beat`) and grew EPS year over year (`require_growth`).
* Optional (off by default): need `min_bars_to_earnings` bars before the next
  report, and with `exit_before_earnings` sell at the close `exit_buffer`
  bar(s) before it so a trade never holds through earnings.

Stops fill at the stop price, or at the open when the stock gaps through it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import candlesticks as cs
from . import earnings as er
from . import indicators as ind


@dataclass
class SwingParams:
    patterns: tuple[str, ...] = ("hammer", "inverted_hammer")
    trend_filter: bool = True
    vol_mult: float = 1.2
    confirm_bars: int = 3
    atr_period: int = 14
    stop_atr: float = 0.5
    max_risk_atr: float = 3.0
    stop_pct: float | None = 0.10   # fixed stop 10% below entry; None = ATR stop
    trail_atr: float | None = None  # e.g. 3.0 for a chandelier trailing stop
    market_ma: int = 200            # with a market index: buy only above this SMA
    min_hold: int = 30          # ~6 weeks
    max_hold: int = 40          # ~8 weeks
    exit_on_bearish: bool = True
    cost_bps: float = 10.0      # per side; mid-cap spreads + slippage
    # Earnings rule (applies when earnings data is supplied)
    post_window: int = 20       # enter within ~4 weeks of a report
    require_beat: bool = True
    min_surprise_pct: float = 0.0
    require_growth: bool = True
    require_positive_reaction: bool = False
    min_bars_to_earnings: int = 0     # 0 = no minimum
    exit_before_earnings: bool = False
    exit_buffer: int = 1


@dataclass
class SwingResult:
    trades: pd.DataFrame
    position: pd.Series
    equity: pd.Series
    stats: dict = field(default_factory=dict)

    def __str__(self) -> str:
        lines = []
        for k, v in self.stats.items():
            if isinstance(v, float):
                v = f"{v:.2%}" if any(s in k for s in ("return", "rate", "drawdown", "Exposure")) else f"{v:.2f}"
            lines.append(f"{k:<22}{v}")
        return "\n".join(lines)


def market_ok(market: pd.DataFrame, index: pd.Index, ma: int = 200) -> np.ndarray:
    """True on dates when the market index closes above its `ma`-day SMA."""
    mc = market["close"]
    above = (mc > ind.sma(mc, ma)).astype(float)
    return above.reindex(index, method="ffill").fillna(0).astype(bool).to_numpy()


def earnings_ok(feat: pd.DataFrame, p: SwingParams) -> pd.Series:
    """Bars where the earnings rule allows a new entry."""
    ok = (feat["bars_since"] >= 1) & (feat["bars_since"] <= p.post_window)
    if p.min_bars_to_earnings > 0:
        ok &= feat["bars_to_next"] >= p.min_bars_to_earnings
    if p.require_beat:
        ok &= (feat["beat"] == 1) & (feat["surprise_pct"] >= p.min_surprise_pct)
    if p.require_growth:
        ok &= feat["eps_growth"] == 1
    if p.require_positive_reaction:
        ok &= feat["reaction_ret"] > 0
    return ok.fillna(False).astype(bool)


def setups(
    df: pd.DataFrame, p: SwingParams | None = None, earnings: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Per-bar setup table: which pattern fired and whether filters passed."""
    p = p or SwingParams()
    c, v = df["close"], df["volume"]
    pats = pd.DataFrame({name: cs.BULLISH[name](df).astype(bool) for name in p.patterns}, index=df.index)
    any_pat = pats.any(axis=1)
    first_pat = pats.idxmax(axis=1).where(any_pat)

    sma50, sma200 = ind.sma(c, 50), ind.sma(c, 200)
    trend_ok = (c > sma200) & (sma50 > sma200) if p.trend_filter else pd.Series(True, index=df.index)
    vol_ok = v >= p.vol_mult * v.rolling(20).mean().shift(1)

    out = pd.DataFrame(
        {
            "pattern": first_pat,
            "trend_ok": trend_ok.fillna(False),
            "vol_ok": vol_ok.fillna(False),
            "setup": any_pat & trend_ok.fillna(False) & vol_ok.fillna(False),
        },
        index=df.index,
    )
    if earnings is not None:
        out["earnings_ok"] = earnings_ok(er.align(df, earnings), p)
        out["setup"] &= out["earnings_ok"]
    return out


def run(
    df: pd.DataFrame,
    p: SwingParams | None = None,
    ticker: str = "",
    earnings: pd.DataFrame | None = None,
    market: pd.DataFrame | None = None,
) -> SwingResult:
    """Simulate the system bar by bar. Returns trades, daily position and equity.

    Pass `earnings` (see `earnings.py`) to turn on the earnings rule, and
    `market` (an index's OHLC, e.g. the S&P 400) to buy only while that index
    is above its `market_ma`-day average.
    """
    p = p or SwingParams()
    if earnings is not None:
        feat = er.align(df, earnings)
        entry_ok = earnings_ok(feat, p).to_numpy()
        to_next = feat["bars_to_next"].to_numpy(float)
    else:
        entry_ok = np.ones(len(df), bool)
        to_next = np.full(len(df), np.nan)
    if market is not None:
        entry_ok = entry_ok & market_ok(market, df.index, p.market_ma)
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    atr = ind.atr(df["high"], df["low"], df["close"], p.atr_period).to_numpy()
    st = setups(df, p, earnings)
    setup = st["setup"].to_numpy()
    pattern = st["pattern"].to_numpy()
    bearish = (
        pd.concat([fn(df) for fn in cs.BEARISH.values()], axis=1).any(axis=1).to_numpy()
        if p.exit_on_bearish else np.zeros(len(df), bool)
    )
    idx = df.index
    cost = p.cost_bps / 10_000

    trades = []
    position = np.zeros(len(df))
    pending = None          # (pattern_bar, pattern_high, pattern_low, name)
    trade = None

    for i in range(len(df)):
        if trade is not None:
            bars_held = i - trade["entry_i"]
            # Stop check on this bar's range (gap-through fills at the open).
            if l[i] <= trade["stop"]:
                fill = min(o[i], trade["stop"])
                trade.update(exit_i=i, exit_price=fill, reason="stop")
            elif p.exit_before_earnings and to_next[i] <= p.exit_buffer:
                trade.update(exit_i=i, exit_price=c[i], reason="earnings")
            elif p.exit_on_bearish and bars_held >= p.min_hold and bearish[i]:
                trade.update(exit_i=i, exit_price=c[i], reason="bearish_pattern")
            elif bars_held >= p.max_hold:
                trade.update(exit_i=i, exit_price=c[i], reason="time")
            if "exit_i" in trade:
                trades.append(trade)
                trade = None
            else:
                position[i] = 1.0
                trade["high_close"] = max(trade["high_close"], c[i])
                if p.trail_atr is not None and not np.isnan(atr[i]):
                    trail = trade["high_close"] - p.trail_atr * atr[i]
                    trade["stop"] = max(trade["stop"], trail)
                continue

        # Flat: look for confirmation of a pending pattern.
        if pending is not None:
            pi, ph, pl, name = pending
            if i - pi > p.confirm_bars:
                pending = None
            elif i > pi and c[i] > ph and not np.isnan(atr[i]) and entry_ok[i]:
                entry = c[i]
                if p.stop_pct is not None:
                    stop = entry * (1 - p.stop_pct)
                else:
                    stop = max(pl - p.stop_atr * atr[i], entry - p.max_risk_atr * atr[i])
                trade = dict(
                    ticker=ticker, pattern=name, pattern_date=idx[pi], entry_i=i,
                    entry_price=entry, initial_stop=stop, stop=stop, high_close=entry,
                    risk_atr=(entry - stop) / atr[i],
                )
                position[i] = 1.0
                pending = None
                continue
        if setup[i]:
            pending = (i, h[i], l[i], pattern[i])

    if trade is not None:  # still open at the end of the data
        trade.update(exit_i=len(df) - 1, exit_price=c[-1], reason="open")
        trades.append(trade)

    tdf = pd.DataFrame(trades)
    if not tdf.empty:
        tdf["entry_date"] = idx[tdf["entry_i"]]
        tdf["exit_date"] = idx[tdf["exit_i"]]
        tdf["bars_held"] = tdf["exit_i"] - tdf["entry_i"]
        tdf["return"] = (tdf["exit_price"] * (1 - cost)) / (tdf["entry_price"] * (1 + cost)) - 1
        tdf["r_multiple"] = (tdf["exit_price"] - tdf["entry_price"]) / (
            tdf["entry_price"] - tdf["initial_stop"]
        )
        tdf = tdf[
            ["ticker", "pattern", "pattern_date", "entry_date", "entry_price", "initial_stop",
             "exit_date", "exit_price", "reason", "bars_held", "return", "r_multiple"]
        ]

    # Daily equity (close-to-close while in a trade, exact fills on entry/exit days).
    daily = pd.Series(0.0, index=idx)
    for t in trades:
        ei, xi = t["entry_i"], t["exit_i"]
        seg = np.concatenate([[t["entry_price"] * (1 + cost)], c[ei + 1 : xi], [t["exit_price"] * (1 - cost)]])
        if xi > ei:
            daily.iloc[ei + 1 : xi + 1] = seg[1:] / seg[:-1] - 1
    equity = (1 + daily).cumprod()

    return SwingResult(tdf, pd.Series(position, index=idx, name="position"), equity, _stats(tdf, equity, df))


def _stats(tdf: pd.DataFrame, equity: pd.Series, df: pd.DataFrame) -> dict:
    years = len(df) / 252
    total = float(equity.iloc[-1] - 1)
    stats = {
        "Total return": total,
        "CAGR": float((1 + total) ** (1 / years) - 1) if years > 0 else float("nan"),
        "Max drawdown": float((equity / equity.cummax() - 1).min()),
        "Buy & hold return": float(df["close"].iloc[-1] / df["close"].iloc[0] - 1),
        "Trades": int(len(tdf)),
    }
    if len(tdf):
        wins, losses = tdf.loc[tdf["return"] > 0, "return"], tdf.loc[tdf["return"] <= 0, "return"]
        stats.update(
            {
                "Win rate": float((tdf["return"] > 0).mean()),
                "Avg trade return": float(tdf["return"].mean()),
                "Avg win return": float(wins.mean()) if len(wins) else 0.0,
                "Avg loss return": float(losses.mean()) if len(losses) else 0.0,
                "Profit factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf"),
                "Avg R": float(tdf["r_multiple"].mean()),
                "Avg bars held": float(tdf["bars_held"].mean()),
                "Exposure": float(sum(tdf["bars_held"]) / len(df)),
            }
        )
    return stats


def pattern_report(trades: pd.DataFrame) -> pd.DataFrame:
    """Which patterns actually pay: stats grouped by pattern (use on multi-ticker trades)."""
    if trades.empty:
        return pd.DataFrame()
    g = trades.groupby("pattern")
    return pd.DataFrame(
        {
            "trades": g.size(),
            "win_rate": g["return"].apply(lambda r: (r > 0).mean()),
            "avg_return": g["return"].mean(),
            "median_return": g["return"].median(),
            "avg_R": g["r_multiple"].mean(),
            "avg_bars": g["bars_held"].mean(),
        }
    ).sort_values("avg_return", ascending=False)
