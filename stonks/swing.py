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
* Chandelier trailing stop: highest close since entry minus `trail_atr` x ATR.
  It only tightens.
* Bearish reversal pattern after `min_hold` bars.
* Time stop at `max_hold` bars (40 trading days is about 8 weeks).

Stops fill at the stop price, or at the open when the stock gaps through it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import candlesticks as cs
from . import indicators as ind


@dataclass
class SwingParams:
    patterns: tuple[str, ...] = (
        "hammer", "bullish_engulfing", "piercing_line", "morning_star",
        "bullish_harami", "tweezer_bottom", "inverted_hammer", "dragonfly_doji",
    )
    trend_filter: bool = True
    vol_mult: float = 1.2
    confirm_bars: int = 3
    atr_period: int = 14
    stop_atr: float = 0.5
    max_risk_atr: float = 3.0
    trail_atr: float = 3.0
    min_hold: int = 30          # ~6 weeks
    max_hold: int = 40          # ~8 weeks
    exit_on_bearish: bool = True
    cost_bps: float = 10.0      # per side; mid-cap spreads + slippage


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


def setups(df: pd.DataFrame, p: SwingParams | None = None) -> pd.DataFrame:
    """Per-bar setup table: which pattern fired and whether filters passed."""
    p = p or SwingParams()
    c, v = df["close"], df["volume"]
    pats = pd.DataFrame({name: cs.BULLISH[name](df).astype(bool) for name in p.patterns}, index=df.index)
    any_pat = pats.any(axis=1)
    first_pat = pats.idxmax(axis=1).where(any_pat)

    sma50, sma200 = ind.sma(c, 50), ind.sma(c, 200)
    trend_ok = (c > sma200) & (sma50 > sma200) if p.trend_filter else pd.Series(True, index=df.index)
    vol_ok = v >= p.vol_mult * v.rolling(20).mean().shift(1)

    return pd.DataFrame(
        {
            "pattern": first_pat,
            "trend_ok": trend_ok.fillna(False),
            "vol_ok": vol_ok.fillna(False),
            "setup": any_pat & trend_ok.fillna(False) & vol_ok.fillna(False),
        },
        index=df.index,
    )


def run(df: pd.DataFrame, p: SwingParams | None = None, ticker: str = "") -> SwingResult:
    """Simulate the system bar by bar. Returns trades, daily position and equity."""
    p = p or SwingParams()
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    atr = ind.atr(df["high"], df["low"], df["close"], p.atr_period).to_numpy()
    st = setups(df, p)
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
                if not np.isnan(atr[i]):
                    trail = trade["high_close"] - p.trail_atr * atr[i]
                    trade["stop"] = max(trade["stop"], trail)
                continue

        # Flat: look for confirmation of a pending pattern.
        if pending is not None:
            pi, ph, pl, name = pending
            if i - pi > p.confirm_bars:
                pending = None
            elif i > pi and c[i] > ph and not np.isnan(atr[i]):
                entry = c[i]
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
