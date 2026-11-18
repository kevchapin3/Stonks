import numpy as np
import pandas as pd
import pytest

from stonks import data, earnings as er, swing


@pytest.fixture
def prices():
    return data.synthetic(3000, seed=5)


def test_reaction_bar_timing(prices):
    d = prices.index[100]
    bmo = er.align(prices, pd.DataFrame({"date": [d], "eps_actual": [1.0], "eps_estimate": [0.9],
                                         "timing": ["bmo"]}))
    amc = er.align(prices, pd.DataFrame({"date": [d], "eps_actual": [1.0], "eps_estimate": [0.9],
                                         "timing": ["amc"]}))
    assert bmo["bars_since"].iloc[100] == 0          # reacts the same morning
    assert np.isnan(amc["bars_since"].iloc[100])     # not known until next day
    assert amc["bars_since"].iloc[101] == 0
    assert amc["bars_to_next"].iloc[100] == 1


def test_surprise_and_growth():
    prices = data.synthetic(400, seed=1)
    dates = prices.index[[10, 73, 136, 199, 262]]
    e = pd.DataFrame({"date": dates, "eps_actual": [1.0, 1.1, 1.2, 1.3, 1.5],
                      "eps_estimate": [1.0, 1.0, 1.3, 1.2, 1.4]})
    f = er.align(prices, e)
    after_q2 = f.iloc[80]
    assert after_q2["beat"] == 1 and after_q2["surprise_pct"] == pytest.approx(0.1)
    assert np.isnan(after_q2["eps_growth"])          # no year-ago quarter yet
    assert f.iloc[150]["beat"] == 0                  # missed
    assert f.iloc[270]["eps_growth"] == 1            # 1.5 > 1.0 four quarters back


def test_align_no_lookahead(prices):
    e = er.synthetic(prices, seed=2)
    full = er.align(prices, e)
    cut = prices.iloc[:2000]
    part = er.align(cut, e[e["date"] <= cut.index[-1]])
    cols = ["bars_since", "surprise_pct", "beat", "eps_growth", "reaction_ret"]
    pd.testing.assert_frame_equal(full.iloc[:2000][cols], part[cols])


def test_trades_never_hold_through_earnings(prices):
    e = er.synthetic(prices, seed=3)
    p = swing.SwingParams(trend_filter=False, vol_mult=0.0, require_growth=False)
    res = swing.run(prices, p, earnings=e)
    t = res.trades
    assert len(t) > 0
    feat = er.align(prices, e)
    react_bars = prices.index[(feat["bars_since"] == 0).to_numpy()]
    for _, tr in t.iterrows():
        inside = react_bars[(react_bars > tr["entry_date"]) & (react_bars <= tr["exit_date"])]
        assert len(inside) == 0, f"trade {tr['entry_date']}..{tr['exit_date']} spans a report"
    # Every entry is within the post-earnings window after a beat.
    at_entry = feat.loc[t["entry_date"]]
    assert (at_entry["bars_since"].between(1, p.post_window)).all()
    assert (at_entry["beat"] == 1).all()


def test_earnings_rule_filters_trades(prices):
    p = swing.SwingParams(trend_filter=False, vol_mult=0.0)
    without = swing.run(prices, p)
    with_rule = swing.run(prices, p, earnings=er.synthetic(prices, seed=4))
    assert len(with_rule.trades) < len(without.trades)


def test_load_csv_roundtrip(tmp_path, prices):
    e = er.synthetic(prices, seed=1)
    path = tmp_path / "X.csv"
    e.to_csv(path, index=False)
    pd.testing.assert_frame_equal(er.load_csv(str(path)), e)
