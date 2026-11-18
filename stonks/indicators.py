"""Technical analysis indicators.

All functions take pandas Series/DataFrames indexed by date and return
Series/DataFrames aligned to the same index. Warm-up periods are NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Moving averages
# --------------------------------------------------------------------------- #


def sma(close: pd.Series, period: int = 20) -> pd.Series:
    """Simple moving average."""
    return close.rolling(window=period, min_periods=period).mean().rename(f"SMA_{period}")


def ema(close: pd.Series, period: int = 20) -> pd.Series:
    """Exponential moving average (alpha = 2 / (period + 1)), seeded after `period` bars."""
    out = close.ewm(span=period, adjust=False, min_periods=period).mean()
    return out.rename(f"EMA_{period}")


def wma(close: pd.Series, period: int = 20) -> pd.Series:
    """Linearly weighted moving average."""
    weights = np.arange(1, period + 1, dtype=float)
    out = close.rolling(period, min_periods=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )
    return out.rename(f"WMA_{period}")


def _wilder(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (RMA), alpha = 1 / period."""
    return series.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


# --------------------------------------------------------------------------- #
# Momentum
# --------------------------------------------------------------------------- #


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder), 0-100."""
    delta = close.diff()
    gain = _wilder(delta.clip(lower=0), period)
    loss = _wilder(-delta.clip(upper=0), period)
    rs = gain / loss
    out = 100 - 100 / (1 + rs)
    # No losses at all -> RSI 100; flat -> 50
    out = out.where(loss != 0, np.where(gain > 0, 100.0, 50.0))
    out[gain.isna()] = np.nan
    return out.rename(f"RSI_{period}")


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """MACD line, signal line and histogram."""
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame(
        {"MACD": macd_line, "MACD_signal": signal_line, "MACD_hist": macd_line - signal_line}
    )


def stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3
) -> pd.DataFrame:
    """Stochastic oscillator %K and %D (0-100)."""
    lowest = low.rolling(k_period, min_periods=k_period).min()
    highest = high.rolling(k_period, min_periods=k_period).max()
    k = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    d = k.rolling(d_period, min_periods=d_period).mean()
    return pd.DataFrame({"STOCH_K": k, "STOCH_D": d})


def roc(close: pd.Series, period: int = 12) -> pd.Series:
    """Rate of change, in percent."""
    return (100 * close.pct_change(period)).rename(f"ROC_{period}")


def williams_r(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Williams %R (-100 to 0)."""
    highest = high.rolling(period, min_periods=period).max()
    lowest = low.rolling(period, min_periods=period).min()
    out = -100 * (highest - close) / (highest - lowest).replace(0, np.nan)
    return out.rename(f"WILLR_{period}")


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    tp = (high + low + close) / 3
    ma = tp.rolling(period, min_periods=period).mean()
    mad = tp.rolling(period, min_periods=period).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    return ((tp - ma) / (0.015 * mad)).rename(f"CCI_{period}")


# --------------------------------------------------------------------------- #
# Volatility
# --------------------------------------------------------------------------- #


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rename("TR")


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range (Wilder)."""
    return _wilder(true_range(high, low, close), period).rename(f"ATR_{period}")


def bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands (population std, as in Bollinger's definition)."""
    mid = sma(close, period)
    std = close.rolling(period, min_periods=period).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return pd.DataFrame(
        {
            "BB_mid": mid,
            "BB_upper": upper,
            "BB_lower": lower,
            "BB_width": (upper - lower) / mid,
            "BB_pctb": (close - lower) / (upper - lower).replace(0, np.nan),
        }
    )


def keltner_channels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
) -> pd.DataFrame:
    mid = ema(close, period)
    rng = atr(high, low, close, atr_period)
    return pd.DataFrame(
        {"KC_mid": mid, "KC_upper": mid + multiplier * rng, "KC_lower": mid - multiplier * rng}
    )


def donchian_channels(high: pd.Series, low: pd.Series, period: int = 20) -> pd.DataFrame:
    upper = high.rolling(period, min_periods=period).max()
    lower = low.rolling(period, min_periods=period).min()
    return pd.DataFrame({"DC_upper": upper, "DC_lower": lower, "DC_mid": (upper + lower) / 2})


# --------------------------------------------------------------------------- #
# Trend
# --------------------------------------------------------------------------- #


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    """Average Directional Index with +DI / -DI."""
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    tr_s = _wilder(true_range(high, low, close), period)
    plus_di = 100 * _wilder(plus_dm, period) / tr_s
    minus_di = 100 * _wilder(minus_dm, period) / tr_s
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return pd.DataFrame({"ADX": _wilder(dx, period), "+DI": plus_di, "-DI": minus_di})


# --------------------------------------------------------------------------- #
# Volume
# --------------------------------------------------------------------------- #


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum().rename("OBV")


def vwap(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> pd.Series:
    """Cumulative VWAP over the whole series (anchor by slicing the input)."""
    tp = (high + low + close) / 3
    return ((tp * volume).cumsum() / volume.cumsum()).rename("VWAP")


def mfi(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14
) -> pd.Series:
    """Money Flow Index (volume-weighted RSI), 0-100."""
    tp = (high + low + close) / 3
    flow = tp * volume
    tp_diff = tp.diff()
    pos = flow.where(tp_diff > 0, 0.0).rolling(period, min_periods=period).sum()
    neg = flow.where(tp_diff < 0, 0.0).rolling(period, min_periods=period).sum()
    out = 100 - 100 / (1 + pos / neg.replace(0, np.nan))
    out = out.where(neg != 0, 100.0)
    out[pos.isna()] = np.nan
    return out.rename(f"MFI_{period}")


# --------------------------------------------------------------------------- #
# Convenience
# --------------------------------------------------------------------------- #


def add_all(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of an OHLCV frame with a standard set of indicators appended.

    Expects columns: open, high, low, close, volume (case-insensitive).
    """
    out = df.copy()
    out.columns = [c.lower() for c in out.columns]
    h, l, c, v = out["high"], out["low"], out["close"], out["volume"]
    parts = [
        sma(c, 20), sma(c, 50), sma(c, 200), ema(c, 12), ema(c, 26),
        rsi(c), macd(c), bollinger_bands(c), atr(h, l, c), adx(h, l, c),
        stochastic(h, l, c), obv(c, v), mfi(h, l, c, v),
    ]
    return pd.concat([out, *parts], axis=1)
