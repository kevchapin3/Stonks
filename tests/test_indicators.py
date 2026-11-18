import numpy as np
import pandas as pd
import pytest

from stonks import backtest, data, indicators as ind, signals


@pytest.fixture
def df():
    return data.synthetic(500, seed=42)


def test_sma_matches_manual():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    out = ind.sma(s, 3)
    assert out.isna().sum() == 2
    assert out.tolist()[2:] == [2.0, 3.0, 4.0]


def test_ema_known_values():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    out = ind.ema(s, 3)  # alpha = 0.5
    assert out.iloc[-1] == pytest.approx(4.0625)


def test_wma():
    s = pd.Series([1, 2, 3], dtype=float)
    assert ind.wma(s, 3).iloc[-1] == pytest.approx((1 + 4 + 9) / 6)


def test_rsi_bounds_and_extremes(df):
    r = ind.rsi(df["close"]).dropna()
    assert r.between(0, 100).all()
    up = pd.Series(np.arange(1, 50, dtype=float))
    assert ind.rsi(up).iloc[-1] == 100
    flat = pd.Series(np.ones(50))
    assert ind.rsi(flat).iloc[-1] == 50


def test_macd_hist(df):
    m = ind.macd(df["close"]).dropna()
    assert np.allclose(m["MACD_hist"], m["MACD"] - m["MACD_signal"])


def test_bollinger_ordering(df):
    bb = ind.bollinger_bands(df["close"]).dropna()
    assert (bb["BB_upper"] >= bb["BB_mid"]).all()
    assert (bb["BB_mid"] >= bb["BB_lower"]).all()


def test_atr_positive(df):
    a = ind.atr(df["high"], df["low"], df["close"]).dropna()
    assert (a > 0).all()


def test_stochastic_and_willr_bounds(df):
    st = ind.stochastic(df["high"], df["low"], df["close"]).dropna()
    assert st["STOCH_K"].between(0, 100).all()
    w = ind.williams_r(df["high"], df["low"], df["close"]).dropna()
    assert w.between(-100, 0).all()


def test_adx_bounds(df):
    a = ind.adx(df["high"], df["low"], df["close"]).dropna()
    assert a["ADX"].between(0, 100).all()


def test_obv_direction():
    close = pd.Series([1.0, 2.0, 1.5, 1.5])
    vol = pd.Series([10.0, 20.0, 30.0, 40.0])
    assert ind.obv(close, vol).tolist() == [0.0, 20.0, -10.0, -10.0]


def test_mfi_bounds(df):
    m = ind.mfi(df["high"], df["low"], df["close"], df["volume"]).dropna()
    assert m.between(0, 100).all()


def test_add_all_shape(df):
    out = ind.add_all(df)
    assert len(out) == len(df)
    assert {"RSI_14", "MACD", "BB_upper", "ATR_14", "ADX", "OBV"} <= set(out.columns)


def test_crossover():
    a = pd.Series([1, 2, 3, 2, 1], dtype=float)
    b = pd.Series([2, 2, 2, 2, 2], dtype=float)
    assert signals.crossover(a, b).tolist() == [False, False, True, False, False]
    assert signals.crossunder(a, b).tolist() == [False, False, False, False, True]


@pytest.mark.parametrize("name", sorted(signals.STRATEGIES))
def test_strategies_valid_positions(df, name):
    pos = signals.STRATEGIES[name](df)
    assert pos.index.equals(df.index)
    assert set(pos.unique()) <= {0.0, 1.0}  # long-only


def test_backtest_no_lookahead():
    # Price doubles on day 2; a signal set on day 2 must not capture it.
    df = pd.DataFrame({"close": [1.0, 1.0, 2.0, 2.0]})
    pos = pd.Series([0.0, 0.0, 1.0, 1.0])
    res = backtest.run(df, pos, cost_bps=0)
    assert res.stats["Total return"] == pytest.approx(0.0)
    pos_early = pd.Series([0.0, 1.0, 1.0, 1.0])
    assert backtest.run(df, pos_early, cost_bps=0).stats["Total return"] == pytest.approx(1.0)


def test_buy_and_hold_matches(df):
    pos = pd.Series(1.0, index=df.index)
    res = backtest.run(df, pos, cost_bps=0)
    expected = df["close"].iloc[-1] / df["close"].iloc[0] - 1
    assert res.stats["Total return"] == pytest.approx(expected)
