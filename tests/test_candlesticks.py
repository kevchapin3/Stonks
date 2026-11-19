import numpy as np
import pandas as pd
import pytest

from stonks import candlesticks as cs, data, swing

# Mechanics tests use every pattern: synthetic candles rarely form hammers.
ALL_PATTERNS = tuple(cs.BULLISH)


def frame(rows):
    """rows: list of (open, high, low, close); prefixed by a 12-bar decline or rally."""
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"],
                        index=pd.bdate_range("2024-01-01", periods=len(rows))).assign(volume=1e6)


def downtrend(n=12, start=100.0):
    rows, p = [], start
    for _ in range(n):
        rows.append((p, p + 0.5, p - 2.5, p - 2))
        p -= 2
    return rows, p


def uptrend(n=12, start=100.0):
    rows, p = [], start
    for _ in range(n):
        rows.append((p, p + 2.5, p - 0.5, p + 2))
        p += 2
    return rows, p


def test_hammer_after_decline():
    rows, p = downtrend()
    rows.append((p, p + 0.2, p - 4, p + 0.5))  # small body at top, long lower shadow
    df = frame(rows)
    assert cs.hammer(df).iloc[-1]
    assert not cs.hanging_man(df).iloc[-1]


def test_hanging_man_after_rally():
    rows, p = uptrend()
    rows.append((p, p + 0.2, p - 4, p + 0.5))
    df = frame(rows)
    assert cs.hanging_man(df).iloc[-1]
    assert not cs.hammer(df).iloc[-1]


def test_shooting_star():
    rows, p = uptrend()
    rows.append((p, p + 4, p - 0.2, p - 0.5))
    assert cs.shooting_star(frame(rows)).iloc[-1]


def test_bullish_engulfing():
    rows, p = downtrend()
    rows.append((p - 1, p + 3, p - 1.5, p + 2.5))  # opens below prior close, closes above prior open
    df = frame(rows)
    assert cs.bullish_engulfing(df).iloc[-1]
    assert not cs.bearish_engulfing(df).iloc[-1]


def test_bearish_engulfing():
    rows, p = uptrend()
    rows.append((p + 1, p + 1.5, p - 3, p - 2.5))
    assert cs.bearish_engulfing(frame(rows)).iloc[-1]


def test_piercing_line():
    rows, p = downtrend()
    rows.append((p, p + 0.3, p - 4.5, p - 4))   # long down candle, midpoint p-2
    p -= 4
    rows.append((p - 1, p + 2.6, p - 1.2, p + 2.5))  # opens below its low, closes above midpoint
    assert cs.piercing_line(frame(rows)).iloc[-1]


def test_morning_star():
    rows, p = downtrend(12)
    rows.append((p, p + 0.2, p - 5, p - 4.8))             # long down
    rows.append((p - 5.5, p - 5.2, p - 6, p - 5.4))         # small star gapping below
    rows.append((p - 5, p - 1.5, p - 5.2, p - 1.8))         # strong up into first body
    assert cs.morning_star(frame(rows)).iloc[-1]


def test_three_white_soldiers():
    rows = [(100, 101, 99, 99.5)] * 5
    rows += [(100, 103.2, 99.8, 103), (102, 106.2, 101.8, 106), (105, 109.2, 104.8, 109)]
    assert cs.three_white_soldiers(frame(rows)).iloc[-1]


def test_doji():
    df = frame([(100, 102, 98, 100.1)])
    assert cs.doji(df).iloc[0]


def test_no_patterns_without_trend_context():
    flat = frame([(100, 101, 99, 100.5)] * 15 + [(100, 100.2, 96, 100.5)])
    assert not cs.hammer(flat).iloc[-1]


def test_detect_all_no_lookahead():
    df = data.synthetic(400, seed=3)
    full = cs.detect_all(df)
    partial = cs.detect_all(df.iloc[:300])
    pd.testing.assert_frame_equal(full.iloc[:300], partial)


def test_weekly_resample():
    df = data.synthetic(100, seed=1)
    w = cs.resample_weekly(df)
    assert (w.index.dayofweek == 4).all()
    assert w["volume"].sum() == df["volume"].sum()
    first_week = df.loc[: w.index[0]]
    assert w["high"].iloc[0] == first_week["high"].max()
    assert w["open"].iloc[0] == first_week["open"].iloc[0]


def test_swing_trades_respect_rules():
    df = data.synthetic(2500, seed=7)
    p = swing.SwingParams(patterns=ALL_PATTERNS, trend_filter=False, vol_mult=0.0)
    res = swing.run(df, p)
    t = res.trades
    assert len(t) > 5
    assert (t["bars_held"] <= p.max_hold).all()
    # Entries happen after the pattern bar, never on it.
    assert (t["entry_date"] > t["pattern_date"]).all()
    # Stops are below entry.
    assert (t["initial_stop"] < t["entry_price"]).all()
    # Trades never overlap.
    assert (t["entry_date"].iloc[1:].to_numpy() > t["exit_date"].iloc[:-1].to_numpy()).all()
    stop_exits = t[t["reason"] == "stop"]
    assert (stop_exits["r_multiple"] <= stop_exits["r_multiple"].max()).all()


def test_swing_equity_matches_trades():
    df = data.synthetic(2500, seed=11)
    res = swing.run(df, swing.SwingParams(patterns=ALL_PATTERNS, trend_filter=False, vol_mult=0.0))
    compounded = (1 + res.trades["return"]).prod() - 1
    assert res.stats["Total return"] == pytest.approx(compounded, rel=1e-9)


def test_midcap_liquidity_filter():
    from stonks import scan
    df = data.synthetic(300, seed=1, start_price=20)
    u = scan.Universe()
    assert scan.liquidity_ok(df, u)[0]
    thin = df.assign(volume=df["volume"] / 1000)
    assert not scan.liquidity_ok(thin, u)[0]
    penny = df.assign(close=df["close"] / 10)
    assert not scan.liquidity_ok(penny, u)[0]


def test_fixed_percent_stop():
    df = data.synthetic(2500, seed=7)
    p = swing.SwingParams(patterns=ALL_PATTERNS, trend_filter=False, vol_mult=0.0, stop_pct=0.10, trail_atr=None)
    t = swing.run(df, p).trades
    assert len(t) > 5
    assert np.allclose(t["initial_stop"], t["entry_price"] * 0.9)
    stopped = t[t["reason"] == "stop"]
    # With no trailing stop, a stop exit fills at the stop or (on a gap) below it.
    assert (stopped["exit_price"] <= stopped["entry_price"] * 0.9 + 1e-9).all()


def test_rsi_filter_only_admits_low_rsi_setups():
    from stonks import indicators as ind
    df = data.synthetic(2500, seed=7)
    p = swing.SwingParams(patterns=ALL_PATTERNS, trend_filter=False, vol_mult=0.0, rsi_max=40)
    st = swing.setups(df, p)
    r = ind.rsi(df["close"])
    assert st["setup"].any()
    assert (r[st["setup"]] <= 40).all()
    loose = swing.setups(df, swing.SwingParams(patterns=ALL_PATTERNS, trend_filter=False, vol_mult=0.0))
    assert st["setup"].sum() < loose["setup"].sum()


def test_entry_rsi_rising_above_60():
    from stonks import indicators as ind
    df = data.synthetic(2500, seed=7)
    p = swing.SwingParams(patterns=ALL_PATTERNS, trend_filter=False, vol_mult=0.0,
                          entry_rsi_min=60, entry_rsi_rising=True)
    t = swing.run(df, p).trades
    r = ind.rsi(df["close"])
    assert len(t) > 0
    assert (r.loc[t["entry_date"]] > 60).all()
    prev = r.shift(1).loc[t["entry_date"]].to_numpy()
    assert (r.loc[t["entry_date"]].to_numpy() > prev).all()
