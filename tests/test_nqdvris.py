import numpy as np
import pandas as pd

from stonks import nqdvris as nq


def rec(t, divs, eps, cash, debt, sector="Industrials", industry="Specialty Industrial Machinery", float_shares=1e9):
    """Synthetic fundamentals record: `divs` = annual dividend per share by year 2019..2026,
    `eps` = diluted EPS by fiscal year 2021..2025."""
    d = {}
    for y, amt in divs.items():
        for q in (2, 5, 8, 11):
            d[f"{y}-{q:02d}-15"] = amt / 4
    return {
        "ticker": t,
        "dividends": d,
        "inc_annual": {"DilutedEPS": {f"{y}-12-31": v for y, v in eps.items()}},
        "bs_quarterly": {"CashCashEquivalentsAndShortTermInvestments": {"2026-06-30": cash},
                         "TotalDebt": {"2026-06-30": debt}},
        "info": {"sector": sector, "industry": industry, "floatShares": float_shares,
                 "averageDailyVolume3Month": 1e6, "longName": t},
    }


def build(recs, price=100.0):
    idx = pd.bdate_range("2019-01-01", "2026-11-30")
    prices = pd.DataFrame({r["ticker"]: price for r in recs}, index=idx)
    fund = {r["ticker"]: r for r in recs}
    return fund, prices


GROW = {y: 1.0 + 0.2 * (y - 2019) for y in range(2019, 2027)}  # rising dividends
EPS_UP = {2021: 3.0, 2022: 3.5, 2023: 4.0, 2024: 4.5, 2025: 5.0}


def test_features_point_in_time():
    fund, prices = build([rec("AAA", GROW, EPS_UP, cash=60, debt=100)])
    f = nq.features(fund, prices, "2026-09-30", "2026-11-30", nq.Params())
    r = f.loc["AAA"]
    # TTM to 2026-09-30 = Nov 2025 + Feb/May/Aug 2026 quarters
    assert np.isclose(r.div_ttm, GROW[2025] / 4 + 3 * GROW[2026] / 4)
    assert r.eps == 5.0 and r.eps_3y == 3.5          # FY2025 vs FY2022
    assert np.isclose(r.cash_debt, 0.6)
    assert np.isclose(r.payout, r.div_ttm / 5.0)


def test_each_screen_rejects_its_failure():
    good = rec("GOOD", GROW, EPS_UP, 60, 100)
    flat_div = rec("FLAT", {y: 2.0 for y in range(2019, 2027)}, EPS_UP, 60, 100)
    eps_down = rec("EPSD", GROW, {2021: 6, 2022: 6, 2023: 5, 2024: 4, 2025: 4}, 60, 100)
    low_cash = rec("CASH", GROW, EPS_UP, 40, 100)
    high_pay = rec("PAY", {y: 4.0 + 0.2 * (y - 2019) for y in range(2019, 2027)}, EPS_UP, 60, 100)
    reit = rec("REIT", GROW, EPS_UP, 60, 100, sector="Real Estate", industry="REIT - Retail")
    fund, prices = build([good, flat_div, eps_down, low_cash, high_pay, reit])
    p = nq.Params(top_n=10, large_cap_n=10, min_adv=0)
    f = nq.screen(nq.features(fund, prices, "2026-09-30", "2026-11-30", p), p)
    assert f.loc["GOOD", "eligible"]
    assert not f.loc["FLAT", "t_div_growth"]
    assert not f.loc["EPSD", "t_eps"]
    assert not f.loc["CASH", "t_cash_debt"]
    assert not f.loc["PAY", "t_payout"]
    assert not f.loc["REIT", "eligible"]
    assert set(f.index[f.eligible]) == {"GOOD"}


def test_selection_ranks_and_industry_cap():
    recs = []
    for i in range(20):
        divs = {y: 1.0 + (0.05 + 0.01 * i) * (y - 2019) for y in range(2019, 2027)}
        sector = "Financial Services" if i < 12 else "Technology"
        recs.append(rec(f"S{i:02d}", divs, EPS_UP, 60, 100, sector=sector, industry="x"))
    fund, prices = build(recs)
    p = nq.Params(top_n=50, large_cap_n=50, min_adv=0, select_n=10, min_large=0, max_per_industry=5)
    f = nq.screen(nq.features(fund, prices, "2026-09-30", "2026-11-30", p), p)
    e = nq.select(f, p)
    sel = e[e.selected]
    assert len(sel) == 10
    assert (sel.icb.value_counts() <= 5).all()
    # bigger dividend growth -> better rank
    assert e.loc["S19", "r_inc"] < e.loc["S00", "r_inc"]


def test_dividend_achiever_years():
    idx = pd.to_datetime([f"{y}-06-15" for y in range(2010, 2027)])
    amounts = [1.0 + 0.1 * i for i in range(17)]
    amounts[5] = amounts[4]          # 2015 flat -> streak restarts in 2016
    divs = pd.Series(amounts, index=idx)
    assert nq.years_of_increases(divs, pd.Timestamp("2026-09-30")) == 10   # 2016..2025
    assert nq.years_of_increases(pd.Series(dtype=float), pd.Timestamp("2026-09-30")) == 0
    only_this_year = pd.Series([0.5], index=pd.to_datetime(["2026-03-01"]))
    assert nq.years_of_increases(only_this_year, pd.Timestamp("2026-09-30")) == 0


def test_special_dividends_and_gaps():
    idx = pd.to_datetime([f"{y}-{m:02d}-15" for y in range(2019, 2027) for m in (2, 5, 8, 11)])
    divs = pd.Series(1.0, index=idx)
    divs[pd.Timestamp("2023-11-15")] = 15.0           # special dividend
    reg = nq.regular_dividends(divs)
    assert pd.Timestamp("2023-11-15") not in reg.index and len(reg) == len(divs) - 1
    gappy = divs.drop([pd.Timestamp("2026-05-15")])   # one quarterly payment missing
    assert np.isclose(nq.ttm_div(gappy, pd.Timestamp("2026-09-30")), 4.0)
    assert np.isclose(nq.ttm_div(gappy, pd.Timestamp("2026-09-30"), fill_gaps=False), 3.0)


def test_zero_dividend_five_years_ago_passes():
    new_payer = {y: (0.0 if y < 2024 else 1.0 + 0.1 * (y - 2024)) for y in range(2019, 2027)}
    fund, prices = build([rec("NEW", new_payer, EPS_UP, 60, 100)])
    p = nq.Params(top_n=10, large_cap_n=10, min_adv=0)
    f = nq.screen(nq.features(fund, prices, "2026-09-30", "2026-11-30", p), p)
    assert f.loc["NEW", "t_div_growth"]


def test_recurring_annual_dividend_is_regular():
    idx = pd.to_datetime([f"{y}-{m:02d}-15" for y in range(2019, 2027) for m in (2, 5, 8, 11)])
    divs = pd.Series(0.1, index=idx)
    for y in range(2019, 2026):
        divs[pd.Timestamp(f"{y}-01-10")] = 1.0 + 0.2 * (y - 2019)   # annual variable dividend
    divs = divs.sort_index()
    assert len(nq.regular_dividends(divs)) == len(divs)


def test_missing_annual_eps_does_not_crash():
    r = rec("NOEPS", GROW, EPS_UP, 60, 100)
    r["inc_annual"] = {}
    fund, prices = build([r])
    f = nq.features(fund, prices, "2026-09-30", "2026-11-30", nq.Params())
    assert np.isnan(f.loc["NOEPS", "eps"])
