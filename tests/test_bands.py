import numpy as np

from stonks import bands, data, indicators as ind


def _z(df):
    c = df["close"]
    return (c - ind.sma(c, 50)) / c.rolling(50).std(ddof=0)


def test_pullback_entries_are_2sd_below_and_above_200d():
    df = data.synthetic(4000, seed=2)
    t = bands.run(df, bands.BandParams(side="pullback"))
    assert len(t) > 3
    z = _z(df).loc[t["entry_date"]]
    assert (z < -2).all()
    assert (df["close"].loc[t["entry_date"]] > ind.sma(df["close"], 200).loc[t["entry_date"]]).all()


def test_breakout_entries_are_2sd_above():
    df = data.synthetic(4000, seed=2)
    t = bands.run(df, bands.BandParams(side="breakout"))
    assert len(t) > 3
    assert (_z(df).loc[t["entry_date"]] > 2).all()
