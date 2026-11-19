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


def test_rsi_cross_confirmation():
    df = data.synthetic(6000, seed=3)
    p = bands.BandParams(side="pullback", k=1.5, rsi_confirm="cross")
    t = bands.run(df, p)
    assert len(t) > 3
    c = df["close"]
    f, s = ind.rsi(c, 14), ind.rsi(c, 21)
    d = t["entry_date"]
    assert (f.loc[d] > s.loc[d]).all()
    assert (f.shift(1).loc[d] <= s.shift(1).loc[d]).all()
    # a band hit occurred within the prior 10 days
    hit = bands.signals(df, bands.BandParams(side="pullback", k=1.5))
    recent = hit.astype(int).rolling(11, min_periods=1).max().astype(bool)
    assert recent.loc[d].all()


def test_scan_reports_recent_signals_only():
    df = data.synthetic(4000, seed=2)
    hits = bands.signals(df, bands.BandParams())
    last = hits[hits].index[-1]
    cut = df.loc[:last]
    out = bands.scan({"X": cut}, days=1)
    assert len(out) == 1 and out["signal_date"].iloc[0] == last.date()
    assert bands.scan({"X": df.loc[: last + (df.index[1] - df.index[0]) * 30]}, days=1).empty
