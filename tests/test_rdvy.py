import numpy as np
import pandas as pd

from stonks import rdvy


def test_calendar_december_2026():
    r = rdvy.Rebalance(2026, 12)
    assert r.sub == "D"
    assert r.fundamentals_ref == pd.Timestamp("2026-09-30")
    assert r.price_ref == pd.Timestamp("2026-11-30")
    assert rdvy.third_friday(2026, 12) == pd.Timestamp("2026-12-18")
    assert r.effective == pd.Timestamp("2026-12-21")
    assert r.announcement == pd.Timestamp("2026-12-11")
    assert rdvy.Rebalance(2027, 3).fundamentals_ref == pd.Timestamp("2026-12-31")
    assert rdvy.next_rebalance(pd.Timestamp("2026-11-19")).month == 12


def test_infer_membership_recovers_synthetic_structure():
    rng = np.random.default_rng(1)
    asof = pd.Timestamp("2026-11-18")
    refs = rdvy.last_ref_dates(asof)
    idx = pd.bdate_range("2025-10-01", asof)
    n = 60
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.015, (len(idx), n)), 0)),
                          index=idx, columns=[f"T{i}" for i in range(n)])
    true = pd.DataFrame(rng.integers(0, 2, (n, 4)), index=prices.columns, columns=sorted(refs))
    true[true.sum(1) == 0] = [1, 0, 0, 0]
    slices = {"A": 0.47, "B": 0.51, "C": 0.53, "D": 0.45}
    w = sum(true[s] * slices[s] * prices.loc[asof] / prices[:refs[s]].iloc[-1] for s in refs)
    mem, u, err = rdvy.infer_membership(w, prices, asof, refs)
    assert err < 0.01
    assert (mem == true).values.mean() > 0.95
