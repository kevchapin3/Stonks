import numpy as np
import pandas as pd

from stonks import data, earnings as er, indicators as ind, momentum as mo


def test_qoq_streak_point_in_time():
    prices = data.synthetic(400, seed=1)
    d = prices.index[[10, 73, 136, 199, 262]]
    e = pd.DataFrame({"date": d, "eps_actual": [1.0, 1.1, 1.2, 1.1, 1.3], "eps_estimate": 1.0,
                      "timing": "bmo"})
    s = mo.qoq_streak(prices, e)
    assert s.iloc[9] == 0 and s.iloc[73] == 1 and s.iloc[150] == 2
    assert s.iloc[200] == 0 and s.iloc[262] == 1
    assert s.iloc[72] == 0  # not known before the report


def test_trades_obey_rules():
    prices = data.synthetic(3000, seed=4)
    e = er.synthetic(prices, seed=4, beat_prob=0.6)
    p = mo.MomentumParams(qoq_streak=1)
    t = mo.run(prices, e, p)
    assert len(t) > 5
    r = ind.rsi(prices["close"])
    assert (r.loc[t["entry_date"]] > 60).all()
    assert (mo.qoq_streak(prices, e).loc[t["entry_date"]] >= 1).all()
    closed = t[t.reason != "open"]
    assert (closed["bars_held"] <= p.hold).all()
    assert (t["entry_date"].iloc[1:].to_numpy() > t["exit_date"].iloc[:-1].to_numpy()).all()
    stops = closed[closed.reason == "stop"]
    assert (stops["exit_price"] <= stops["entry_price"] * 0.9 + 1e-9).all()
