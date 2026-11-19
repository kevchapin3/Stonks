import numpy as np
import pandas as pd
import pytest

from stonks import timing


def series(vals):
    return pd.Series(vals, index=pd.bdate_range("2020-01-01", periods=len(vals)), dtype=float)


def test_positions_follow_ma_and_apply_next_day():
    up = np.linspace(100, 200, 300)
    down = np.linspace(200, 100, 300)
    close = series(np.concatenate([up, down]))
    pos = timing.positions(close, ma=50)
    assert pos.iloc[49] == 1.0          # first bar with an MA, price above it
    assert pos.iloc[-1] == 0.0          # falling well below its MA at the end
    daily = timing.backtest(close, pos, cost_bps=0)
    # day 50's return is earned because day 49's close said "in"
    assert daily.iloc[50] == pytest.approx(close.iloc[50] / close.iloc[49] - 1)


def test_cash_earns_tbill():
    close = series(np.linspace(100, 50, 300))      # always below MA -> cash
    pos = timing.positions(close, ma=50)
    tb = series(np.full(300, 5.04))
    daily = timing.backtest(close, pos, tb, cost_bps=0)
    assert pos.iloc[60:].eq(0).all()
    assert daily.iloc[100] == pytest.approx(0.0504 / 252)


def test_band_reduces_switches():
    rng = np.random.default_rng(0)
    close = series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 3000))))
    n0 = timing.positions(close).diff().abs().sum()
    n1 = timing.positions(close, band=0.02).diff().abs().sum()
    nm = timing.positions(close, monthly=True).diff().abs().sum()
    assert n1 < n0 and nm < n0
