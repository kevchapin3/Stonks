import pandas as pd

from stonks import bands, data, paper


def test_ledger_logs_closes_and_does_not_duplicate():
    df = data.synthetic(4000, seed=2)
    spy = data.synthetic(4000, seed=9)["close"]
    hits = bands.signals(df, bands.BandParams()).pipe(lambda s: s[s].index)
    first = hits[len(hits) // 2]
    # Run 1: data up to just after a signal
    cut = df.loc[: df.index[df.index.get_loc(first) + 2]]
    led = paper.update({"X": cut}, spy, pd.DataFrame(), since=df.index[df.index.get_loc(first) - 1], ks=(2.0,))
    assert len(led) == 1 and led.status.iloc[0] == "open"
    assert led.entry_price.iloc[0] == df.close.loc[first]
    # Run 2: 60 more bars -> the position closes after 40 bars, no duplicate for the same signal
    cut2 = df.loc[: df.index[df.index.get_loc(first) + 60]]
    led2 = paper.update({"X": cut2}, spy, led, since=cut.index[-1], ks=(2.0,))
    row = led2[led2.signal_date == first].iloc[0]
    assert row.status == "closed" and row.days_held == 40
    assert row.exit_price == df.close.iloc[df.index.get_loc(first) + 40]
    assert (led2.signal_date == first).sum() == 1
    md = paper.report(led2, cut2.index[-1])
    assert "Closed trades" in md and "2SD" in md
