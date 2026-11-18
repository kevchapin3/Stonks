"""Trading signal generators.

Each strategy returns a position Series aligned to the input index:
1 = long, 0 = flat, -1 = short. The position on day t is decided using
data up to and including day t; the backtester applies it from day t+1.
"""

from __future__ import annotations

import pandas as pd

from . import indicators as ind


def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    """True on the bar where `a` crosses above `b`."""
    return (a > b) & (a.shift(1) <= b.shift(1))


def crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    """True on the bar where `a` crosses below `b`."""
    return (a < b) & (a.shift(1) >= b.shift(1))


def _from_events(entries: pd.Series, exits: pd.Series, allow_short: bool) -> pd.Series:
    """Turn entry/exit event booleans into a held position series."""
    pos = pd.Series(float("nan"), index=entries.index)
    pos[entries] = 1.0
    pos[exits] = -1.0 if allow_short else 0.0
    return pos.ffill().fillna(0.0)


def ma_crossover(
    df: pd.DataFrame, fast: int = 50, slow: int = 200, kind: str = "sma", allow_short: bool = False
) -> pd.Series:
    """Golden cross / death cross."""
    f = ind.ema if kind == "ema" else ind.sma
    fast_ma, slow_ma = f(df["close"], fast), f(df["close"], slow)
    pos = pd.Series(0.0, index=df.index)
    pos[fast_ma > slow_ma] = 1.0
    if allow_short:
        pos[fast_ma < slow_ma] = -1.0
    pos[slow_ma.isna()] = 0.0
    return pos.rename("position")


def rsi_reversion(
    df: pd.DataFrame, period: int = 14, oversold: float = 30, overbought: float = 70,
    allow_short: bool = False,
) -> pd.Series:
    """Buy when RSI crosses back above `oversold`, exit when it crosses below `overbought`."""
    r = ind.rsi(df["close"], period)
    lo = pd.Series(oversold, index=df.index)
    hi = pd.Series(overbought, index=df.index)
    return _from_events(crossover(r, lo), crossunder(r, hi), allow_short).rename("position")


def macd_signal(df: pd.DataFrame, allow_short: bool = False) -> pd.Series:
    """Long when MACD crosses above its signal line, exit on cross below."""
    m = ind.macd(df["close"])
    return _from_events(
        crossover(m["MACD"], m["MACD_signal"]),
        crossunder(m["MACD"], m["MACD_signal"]),
        allow_short,
    ).rename("position")


def bollinger_reversion(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """Buy on close back inside lower band, exit at middle band."""
    bb = ind.bollinger_bands(df["close"], period, num_std)
    entries = crossover(df["close"], bb["BB_lower"])
    exits = crossover(df["close"], bb["BB_mid"])
    return _from_events(entries, exits, allow_short=False).rename("position")


def donchian_breakout(df: pd.DataFrame, entry: int = 20, exit: int = 10) -> pd.Series:
    """Turtle-style breakout: buy new `entry`-day high, exit new `exit`-day low."""
    upper = df["high"].rolling(entry).max().shift(1)
    lower = df["low"].rolling(exit).min().shift(1)
    return _from_events(df["close"] > upper, df["close"] < lower, allow_short=False).rename(
        "position"
    )


STRATEGIES = {
    "ma_cross": ma_crossover,
    "rsi": rsi_reversion,
    "macd": macd_signal,
    "bollinger": bollinger_reversion,
    "donchian": donchian_breakout,
}
