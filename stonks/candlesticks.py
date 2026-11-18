"""Japanese candlestick pattern recognition.

Every detector takes an OHLC DataFrame (lower-case columns) and returns a
boolean Series that is True on the bar that *completes* the pattern.

Reversal patterns only mean something after a move to reverse, so the
bullish reversals require a prior short-term downtrend and the bearish ones
a prior short-term uptrend (see `prior_trend`). This follows Steve Nison's
point that a hammer in the middle of a range is just a candle.
"""

from __future__ import annotations

import pandas as pd

from . import indicators as ind

# --------------------------------------------------------------------------- #
# Candle anatomy
# --------------------------------------------------------------------------- #


def _parts(df: pd.DataFrame):
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    rng = (h - l).where(h > l)
    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    lower = pd.concat([o, c], axis=1).min(axis=1) - l
    return o, h, l, c, body, rng, upper, lower


def avg_body(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """Average real-body size, used to judge 'long' vs 'small' bodies."""
    return (df["close"] - df["open"]).abs().rolling(period, min_periods=1).mean()


def prior_trend(df: pd.DataFrame, lookback: int = 10, ma: int = 10) -> pd.Series:
    """+1 if the market was rising into the previous bar, -1 if falling, 0 otherwise.

    Uses the previous bar's close vs. its short SMA and the close `lookback`
    bars earlier, so the pattern candle itself does not define its own trend.
    """
    c = df["close"]
    prev = c.shift(1)
    sma_prev = ind.sma(c, ma).shift(1)
    past = c.shift(lookback + 1)
    up = (prev > sma_prev) & (prev > past)
    down = (prev < sma_prev) & (prev < past)
    return up.astype(int) - down.astype(int)


# --------------------------------------------------------------------------- #
# Single-candle patterns
# --------------------------------------------------------------------------- #


def doji(df: pd.DataFrame, max_body_pct: float = 0.1) -> pd.Series:
    """Open and close (nearly) equal: indecision."""
    *_, body, rng, _, _ = _parts(df)
    return (body <= max_body_pct * rng).fillna(False)


def dragonfly_doji(df: pd.DataFrame) -> pd.Series:
    *_, rng, upper, lower = _parts(df)
    return (doji(df) & (upper <= 0.1 * rng) & (lower >= 0.6 * rng)).fillna(False)


def gravestone_doji(df: pd.DataFrame) -> pd.Series:
    *_, rng, upper, lower = _parts(df)
    return (doji(df) & (lower <= 0.1 * rng) & (upper >= 0.6 * rng)).fillna(False)


def spinning_top(df: pd.DataFrame) -> pd.Series:
    *_, body, rng, upper, lower = _parts(df)
    return (
        (body > 0.1 * rng) & (body <= 0.3 * rng) & (upper > body) & (lower > body)
    ).fillna(False)


def _hammer_shape(df: pd.DataFrame) -> pd.Series:
    """Small body at the top, lower shadow >= 2x body, little upper shadow."""
    *_, body, rng, upper, lower = _parts(df)
    return (
        (lower >= 2 * body) & (upper <= 0.25 * rng) & (body >= 0.05 * rng) & (body <= 0.35 * rng)
    ).fillna(False)


def _inverted_shape(df: pd.DataFrame) -> pd.Series:
    """Small body at the bottom, upper shadow >= 2x body, little lower shadow."""
    *_, body, rng, upper, lower = _parts(df)
    return (
        (upper >= 2 * body) & (lower <= 0.25 * rng) & (body >= 0.05 * rng) & (body <= 0.35 * rng)
    ).fillna(False)


def hammer(df: pd.DataFrame) -> pd.Series:
    """Bullish: hammer shape after a decline."""
    return _hammer_shape(df) & (prior_trend(df) == -1)


def hanging_man(df: pd.DataFrame) -> pd.Series:
    """Bearish: hammer shape after an advance."""
    return _hammer_shape(df) & (prior_trend(df) == 1)


def inverted_hammer(df: pd.DataFrame) -> pd.Series:
    """Bullish (needs confirmation): inverted shape after a decline."""
    return _inverted_shape(df) & (prior_trend(df) == -1)


def shooting_star(df: pd.DataFrame) -> pd.Series:
    """Bearish: inverted shape after an advance."""
    return _inverted_shape(df) & (prior_trend(df) == 1)


def bullish_marubozu(df: pd.DataFrame) -> pd.Series:
    o, h, l, c, body, rng, upper, lower = _parts(df)
    return ((c > o) & (body >= 0.9 * rng) & (body > 1.5 * avg_body(df).shift(1))).fillna(False)


def bearish_marubozu(df: pd.DataFrame) -> pd.Series:
    o, h, l, c, body, rng, upper, lower = _parts(df)
    return ((c < o) & (body >= 0.9 * rng) & (body > 1.5 * avg_body(df).shift(1))).fillna(False)


# --------------------------------------------------------------------------- #
# Two-candle patterns
# --------------------------------------------------------------------------- #


def bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    """Up candle whose body engulfs the prior down candle's body, after a decline."""
    o, h, l, c, *_ = _parts(df)
    po, pc = o.shift(1), c.shift(1)
    return (
        (pc < po) & (c > o) & (o <= pc) & (c >= po) & ((c - o) > (po - pc))
        & (prior_trend(df) == -1)
    )


def bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    o, h, l, c, *_ = _parts(df)
    po, pc = o.shift(1), c.shift(1)
    return (
        (pc > po) & (c < o) & (o >= pc) & (c <= po) & ((o - c) > (pc - po))
        & (prior_trend(df) == 1)
    )


def bullish_harami(df: pd.DataFrame) -> pd.Series:
    """Small up body inside a long prior down body, after a decline."""
    o, h, l, c, body, *_ = _parts(df)
    po, pc = o.shift(1), c.shift(1)
    long_prev = (po - pc) > avg_body(df).shift(2)
    return (
        (pc < po) & long_prev & (c > o) & (o > pc) & (c < po) & (prior_trend(df) == -1)
    )


def bearish_harami(df: pd.DataFrame) -> pd.Series:
    o, h, l, c, body, *_ = _parts(df)
    po, pc = o.shift(1), c.shift(1)
    long_prev = (pc - po) > avg_body(df).shift(2)
    return (
        (pc > po) & long_prev & (c < o) & (o < pc) & (c > po) & (prior_trend(df) == 1)
    )


def piercing_line(df: pd.DataFrame) -> pd.Series:
    """Opens below the prior low, closes above the midpoint of the prior long down body."""
    o, h, l, c, *_ = _parts(df)
    po, pc, pl = o.shift(1), c.shift(1), l.shift(1)
    long_prev = (po - pc) > avg_body(df).shift(2)
    return (
        (pc < po) & long_prev & (c > o) & (o < pl) & (c > (po + pc) / 2) & (c < po)
        & (prior_trend(df) == -1)
    )


def dark_cloud_cover(df: pd.DataFrame) -> pd.Series:
    o, h, l, c, *_ = _parts(df)
    po, pc, ph = o.shift(1), c.shift(1), h.shift(1)
    long_prev = (pc - po) > avg_body(df).shift(2)
    return (
        (pc > po) & long_prev & (c < o) & (o > ph) & (c < (po + pc) / 2) & (c > po)
        & (prior_trend(df) == 1)
    )


def tweezer_bottom(df: pd.DataFrame, tol: float = 0.002) -> pd.Series:
    """Two consecutive candles with matching lows after a decline; first down, second up."""
    o, h, l, c, *_ = _parts(df)
    return (
        ((l - l.shift(1)).abs() <= tol * c) & (c.shift(1) < o.shift(1)) & (c > o)
        & (prior_trend(df) == -1)
    )


def tweezer_top(df: pd.DataFrame, tol: float = 0.002) -> pd.Series:
    o, h, l, c, *_ = _parts(df)
    return (
        ((h - h.shift(1)).abs() <= tol * c) & (c.shift(1) > o.shift(1)) & (c < o)
        & (prior_trend(df) == 1)
    )


# --------------------------------------------------------------------------- #
# Three-candle patterns
# --------------------------------------------------------------------------- #


def morning_star(df: pd.DataFrame) -> pd.Series:
    """Long down candle, small 'star' gapping lower, long up candle closing into bar 1's body."""
    o, h, l, c, body, *_ = _parts(df)
    ab = avg_body(df).shift(3)
    o1, c1 = o.shift(2), c.shift(2)
    o2, c2, b2 = o.shift(1), c.shift(1), body.shift(1)
    first_long_down = (c1 < o1) & ((o1 - c1) > ab)
    small_star = (b2 < 0.5 * (o1 - c1)) & (pd.concat([o2, c2], axis=1).max(axis=1) < c1)
    third_up = (c > o) & (c > (o1 + c1) / 2)
    trend = prior_trend(df).shift(2) == -1
    return (first_long_down & small_star & third_up & trend).fillna(False)


def evening_star(df: pd.DataFrame) -> pd.Series:
    o, h, l, c, body, *_ = _parts(df)
    ab = avg_body(df).shift(3)
    o1, c1 = o.shift(2), c.shift(2)
    o2, c2, b2 = o.shift(1), c.shift(1), body.shift(1)
    first_long_up = (c1 > o1) & ((c1 - o1) > ab)
    small_star = (b2 < 0.5 * (c1 - o1)) & (pd.concat([o2, c2], axis=1).min(axis=1) > c1)
    third_down = (c < o) & (c < (o1 + c1) / 2)
    trend = prior_trend(df).shift(2) == 1
    return (first_long_up & small_star & third_down & trend).fillna(False)


def three_white_soldiers(df: pd.DataFrame) -> pd.Series:
    """Three long up candles, each opening within the prior body and closing near its high."""
    o, h, l, c, body, rng, upper, lower = _parts(df)
    up = (c > o) & (body > 0.6 * rng) & (upper < 0.25 * rng)
    opens_inside = (o > o.shift(1)) & (o < c.shift(1))
    higher = c > c.shift(1)
    step = up & opens_inside & higher
    return (up.shift(2, fill_value=False) & step.shift(1, fill_value=False) & step).fillna(False)


def three_black_crows(df: pd.DataFrame) -> pd.Series:
    o, h, l, c, body, rng, upper, lower = _parts(df)
    down = (c < o) & (body > 0.6 * rng) & (lower < 0.25 * rng)
    opens_inside = (o < o.shift(1)) & (o > c.shift(1))
    lower_close = c < c.shift(1)
    step = down & opens_inside & lower_close
    return (down.shift(2, fill_value=False) & step.shift(1, fill_value=False) & step).fillna(False)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

BULLISH = {
    "hammer": hammer,
    "inverted_hammer": inverted_hammer,
    "dragonfly_doji": lambda d: dragonfly_doji(d) & (prior_trend(d) == -1),
    "bullish_engulfing": bullish_engulfing,
    "bullish_harami": bullish_harami,
    "piercing_line": piercing_line,
    "tweezer_bottom": tweezer_bottom,
    "morning_star": morning_star,
    "three_white_soldiers": three_white_soldiers,
    "bullish_marubozu": bullish_marubozu,
}

BEARISH = {
    "hanging_man": hanging_man,
    "shooting_star": shooting_star,
    "gravestone_doji": lambda d: gravestone_doji(d) & (prior_trend(d) == 1),
    "bearish_engulfing": bearish_engulfing,
    "bearish_harami": bearish_harami,
    "dark_cloud_cover": dark_cloud_cover,
    "tweezer_top": tweezer_top,
    "evening_star": evening_star,
    "three_black_crows": three_black_crows,
    "bearish_marubozu": bearish_marubozu,
}

NEUTRAL = {"doji": doji, "spinning_top": spinning_top}


def detect_all(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean frame with one column per pattern."""
    cols = {**BULLISH, **BEARISH, **NEUTRAL}
    return pd.DataFrame({name: fn(df).astype(bool) for name, fn in cols.items()}, index=df.index)


def pattern_score(df: pd.DataFrame) -> pd.Series:
    """Net count of bullish minus bearish patterns firing on each bar."""
    bull = sum(fn(df).astype(int) for fn in BULLISH.values())
    bear = sum(fn(df).astype(int) for fn in BEARISH.values())
    return (bull - bear).rename("candle_score")


def resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Daily OHLCV -> weekly (Friday-ending) bars. Weekly candles suit 6-8 week holds."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return df.resample("W-FRI").agg(agg).dropna()
