"""Replicate the Nasdaq US Rising Dividend Achievers Index (NQDVRIS) sub-portfolio selection.

The index (tracked by the RDVY ETF) holds four sub-portfolios of up to 50 equal-weighted
stocks; one is rebuilt each quarter. This module applies the published rules to
fundamentals downloaded from Yahoo Finance (`data/rdvy/fund/*.json`) as of a
reconstitution reference date. See rdvy/README.md for the rules.

Approximations (Nasdaq uses its own data vendors):
  * EPS: diluted EPS by fiscal year (Yahoo), falling back to quarterly reported EPS history
  * Cash-to-debt: cash, equivalents and short-term investments / total debt
  * Large-cap: top `large_cap_n` of the candidate universe by free-float market cap
  * ICB industry: mapped from Yahoo sector/industry, overridden by First Trust's labels
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Domiciles common for US-listed, US-style companies included in US benchmark indexes
US_STYLE_DOMICILES = {"Ireland", "Switzerland", "Bermuda", "United Kingdom", "Jersey", "Netherlands",
                      "Puerto Rico", "Panama", "Cayman Islands", "Curacao"}

# Yahoo sector -> ICB Industry (2019 structure used by First Trust's holdings page)
SECTOR_TO_ICB = {
    "Financial Services": "Financials", "Industrials": "Industrials", "Technology": "Technology",
    "Consumer Cyclical": "Consumer Discretionary", "Consumer Defensive": "Consumer Staples",
    "Healthcare": "Health Care", "Energy": "Energy", "Basic Materials": "Basic Materials",
    "Utilities": "Utilities", "Communication Services": "Telecommunications", "Real Estate": "Real Estate",
}
# Yahoo industries that ICB files elsewhere
INDUSTRY_TO_ICB = {
    "Credit Services": "Industrials",            # Visa, Mastercard, PayPal: ICB Industrial Support Services
    "Information Technology Services": "Industrials",  # Accenture, IT consulting
    "Staffing & Employment Services": "Industrials",
    "Discount Stores": "Consumer Discretionary",  # Costco, Walmart, Dollar stores are retailers in ICB
    "Grocery Stores": "Consumer Staples",
    "Internet Content & Information": "Technology",
    "Electronic Gaming & Multimedia": "Consumer Discretionary",
    "Entertainment": "Consumer Discretionary",
    "Advertising Agencies": "Consumer Discretionary",
    "Broadcasting": "Consumer Discretionary",
    "Publishing": "Consumer Discretionary",
    "Waste Management": "Industrials",
}


@dataclass
class Params:
    top_n: int = 750                 # top N by free-float market cap
    achievers_only: bool = False     # start from Dividend Achievers (10+ years of increases)
    min_achiever_years: int = 10
    min_adv: float = 5e6             # 3-month average daily value traded
    max_payout: float = 0.65
    increase_mode: str = "dollar"    # "dollar" or "pct": how the 5-year dividend increase is ranked
    countries: dict = field(default_factory=dict)  # ticker -> {"country", "financialCurrency"}
    min_cash_debt: float = 0.50
    select_n: int = 50
    min_large: int = 33
    large_cap_n: int = 300           # approximation of the NQUSB large-cap segment
    max_per_industry: int = 15
    report_lag_days: int = 45        # financial statements assumed public this long after period end
    exclude: set = field(default_factory=set)  # e.g. sub-portfolio weight-cap names


def load_fund(directory: str | Path) -> dict[str, dict]:
    return {f.stem: json.loads(f.read_text()) for f in Path(directory).glob("*.json")}


def load_earnings(*dirs: str | Path) -> dict[str, pd.DataFrame]:
    """Quarterly reported-EPS histories (date, eps_actual) from earnings CSV folders."""
    out = {}
    for d in dirs:
        for f in Path(d).glob("*.csv"):
            out.setdefault(f.stem, pd.read_csv(f))
    return out


def _s(d: dict) -> pd.Series:
    if not d:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    s = pd.Series(d, dtype=float)
    s.index = pd.to_datetime(s.index)
    return s.sort_index().dropna()


def regular_dividends(divs: pd.Series, mult: float = 2.5) -> pd.Series:
    """Drop special (one-off) dividends: payments more than `mult` x the median of nearby payments."""
    if len(divs) < 4:
        return divs
    big = {}
    for d, v in divs.items():
        near = divs[(divs.index >= d - pd.Timedelta(days=730)) & (divs.index <= d + pd.Timedelta(days=730))]
        med = near.drop(d, errors="ignore").median()
        big[d] = bool(pd.notna(med) and v > mult * med)
    keep = []
    for d in divs.index:
        if not big[d]:
            keep.append(True)
            continue
        # a large payment that recurs about a year apart is a regular annual dividend (e.g. PGR)
        recurs = any(big[o] for o in divs.index
                     if o != d and pd.Timedelta(days=270) <= abs(o - d) <= pd.Timedelta(days=460))
        keep.append(recurs)
    return divs[keep]


def payments_per_year(divs: pd.Series, asof: pd.Timestamp) -> int:
    """Typical number of payments per year over the five full years before `asof`."""
    w = divs[(divs.index.year < asof.year) & (divs.index.year >= asof.year - 5)]
    if w.empty:
        return 0
    counts = w.groupby(w.index.year).size()
    return int(counts.median())


def ttm_div(divs: pd.Series, asof: pd.Timestamp, fill_gaps: bool = True) -> float:
    """Trailing-12-month regular dividends. With `fill_gaps`, missing payments (data gaps)
    are filled with the most recent payment amount, based on the usual payment frequency."""
    w = divs[(divs.index > asof - pd.Timedelta(days=365)) & (divs.index <= asof)]
    total = float(w.sum())
    if fill_gaps and len(w):
        freq = payments_per_year(divs, asof)
        if freq and len(w) < freq:
            total += (freq - len(w)) * float(w.iloc[-1])
    return total


def years_of_increases(divs: pd.Series, asof: pd.Timestamp) -> int:
    """Consecutive calendar years (ending with the last full year before `asof`) in which the
    total dividend paid exceeded the prior year's: the Nasdaq Dividend Achievers test (10+)."""
    if divs.empty:
        return 0
    last_full = asof.year - 1
    yearly = divs[divs.index.year <= last_full].groupby(divs[divs.index.year <= last_full].index.year).sum()
    if yearly.empty:
        return 0
    yearly = yearly.reindex(range(int(yearly.index.min()), last_full + 1), fill_value=0.0)
    n = 0
    for y in range(last_full, int(yearly.index.min()), -1):
        if yearly[y] > yearly[y - 1] * 1.0001:
            n += 1
        else:
            break
    return n


def eps_pair(rec: dict, asof: pd.Timestamp, lag: int, earn_hist: pd.DataFrame | None = None):
    """(latest fiscal-year EPS known at `asof`, EPS three fiscal years earlier)."""
    a = _s(rec.get("inc_annual", {}).get("DilutedEPS", {}))
    known = a[a.index <= asof - pd.Timedelta(days=lag)]
    if len(known):
        cur_end = known.index[-1]
        cur = float(known.iloc[-1])
        prior = a[(a.index <= cur_end - pd.Timedelta(days=3 * 365 - 40)) &
                  (a.index >= cur_end - pd.Timedelta(days=3 * 365 + 40))]
        if len(prior):
            return cur, float(prior.iloc[-1])
    # Fallback: quarterly reported EPS history (earnings CSV), trailing four quarters
    if earn_hist is not None and len(earn_hist):
        e = earn_hist.dropna(subset=["eps_actual"])
        e = e[pd.to_datetime(e["date"]) <= asof].sort_values("date")
        if len(e) >= 16:
            cur = e["eps_actual"].iloc[-4:].sum()
            prior = e["eps_actual"].iloc[-16:-12].sum()
            return float(cur), float(prior)
        if len(known):
            return float(known.iloc[-1]), np.nan
    return (float(known.iloc[-1]) if len(known) else np.nan), np.nan


def cash_to_debt(rec: dict, asof: pd.Timestamp, lag: int) -> float:
    for freq in ("bs_quarterly", "bs_annual"):
        bs = rec.get(freq, {})
        cash = _s(bs.get("CashCashEquivalentsAndShortTermInvestments") or bs.get("CashAndCashEquivalents", {}))
        debt = _s(bs.get("TotalDebt", {}))
        cash = cash[cash.index <= asof - pd.Timedelta(days=lag)]
        if len(cash) == 0:
            continue
        d = debt.reindex(cash.index).iloc[-1] if len(debt) else np.nan
        c = float(cash.iloc[-1])
        if pd.isna(d) or d <= 0:
            return np.inf
        return c / float(d)
    return np.nan


def icb_industry(rec: dict, overrides: dict[str, str]) -> str:
    t = rec["ticker"]
    if t in overrides:
        return overrides[t]
    info = rec.get("info", {}) or {}
    return INDUSTRY_TO_ICB.get(info.get("industry"), SECTOR_TO_ICB.get(info.get("sector"), "Unknown"))


def features(fund: dict[str, dict], prices: pd.DataFrame, ref: str, price_ref: str,
             p: Params, overrides: dict[str, str] | None = None,
             earnings: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """One row per ticker with every input to the screen, as of the reference dates."""
    asof, pref = pd.Timestamp(ref), pd.Timestamp(price_ref)
    overrides = overrides or {}
    rows = []
    for t, rec in fund.items():
        info = rec.get("info", {}) or {}
        if t not in prices.columns:
            continue
        px = prices[t].dropna()
        if px.empty or px.index[0] > asof - pd.Timedelta(days=60):
            continue
        p_ref = float(px[:asof].iloc[-1]) if len(px[:asof]) else np.nan
        p_mc = float(px[:pref].iloc[-1]) if len(px[:pref]) else np.nan
        divs = regular_dividends(_s(rec.get("dividends", {})))
        d0 = ttm_div(divs, asof)
        d3 = ttm_div(divs, asof - pd.DateOffset(years=3))
        d5 = ttm_div(divs, asof - pd.DateOffset(years=5))
        e0, e3 = eps_pair(rec, asof, p.report_lag_days, (earnings or {}).get(t))
        shares = info.get("floatShares") or info.get("sharesOutstanding") or np.nan
        vol = info.get("averageDailyVolume3Month") or np.nan
        rows.append(dict(
            ticker=t, name=info.get("longName"), sector=info.get("sector"), industry=info.get("industry"),
            icb=icb_industry(rec, overrides), price=p_ref, div_ttm=d0, div_3y=d3, div_5y=d5,
            div_increase_5y=d0 - d5, yield_=d0 / p_ref if p_ref else np.nan,
            achiever_years=years_of_increases(divs, asof),
            eps=e0, eps_3y=e3, payout=d0 / e0 if e0 and e0 > 0 else np.nan,
            cash_debt=cash_to_debt(rec, asof, p.report_lag_days),
            float_mcap=p_mc * shares if shares else np.nan, adv=p_mc * vol if vol else np.nan,
        ))
    return pd.DataFrame(rows).set_index("ticker")


def screen(f: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Apply eligibility tests; returns `f` with a pass/fail column per test and `eligible`."""
    f = f.copy()
    f["is_reit"] = (f["sector"] == "Real Estate") | f["industry"].fillna("").str.startswith("REIT")
    # one security per company: drop share classes other than the most traded one
    names = f["name"].fillna(pd.Series(f.index, index=f.index))
    best = f.assign(_n=names).sort_values("adv", ascending=False).drop_duplicates("_n").index
    f["t_one_class"] = f.index.isin(best)
    pool = f[~f["is_reit"]].sort_values("float_mcap", ascending=False)
    top = set(pool.index[: p.top_n])
    f["size_rank"] = pool["float_mcap"].rank(ascending=False)
    f["large_cap"] = f["size_rank"] <= p.large_cap_n
    f["t_size"] = f.index.isin(top)
    f["t_liquid"] = f["adv"] >= p.min_adv
    f["t_div_growth"] = (f["div_ttm"] > f["div_3y"]) & (f["div_ttm"] > f["div_5y"])
    f["t_eps"] = (f["eps"] > 0) & (f["eps"] > f["eps_3y"])
    f["t_cash_debt"] = f["cash_debt"] > p.min_cash_debt
    f["t_payout"] = f["payout"] <= p.max_payout
    f["t_cap"] = ~f.index.isin(p.exclude)
    if p.countries:
        def us_company(t):
            c = p.countries.get(t) or {}
            if not c:
                return True
            if c.get("country") == "United States":
                return True
            # US-style companies domiciled abroad (e.g. Chubb, Accenture) report in USD;
            # ADRs of foreign companies generally report in their home currency
            return c.get("financialCurrency") == "USD" and c.get("country") in US_STYLE_DOMICILES
        f["t_us"] = [us_company(t) for t in f.index]
    else:
        f["t_us"] = True
    f["t_achiever"] = (f["achiever_years"] >= p.min_achiever_years) if p.achievers_only else True
    tests = ["t_size", "t_liquid", "t_div_growth", "t_eps", "t_cash_debt", "t_payout", "t_cap", "t_achiever",
             "t_us", "t_one_class"]
    f["eligible"] = (~f["is_reit"]) & f[tests].all(axis=1)
    return f


def select(f: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Rank eligible names and pick up to `select_n` with the large-cap and industry rules."""
    e = f[f["eligible"]].copy()
    if p.increase_mode == "pct":
        inc = (e["div_ttm"] / e["div_5y"].replace(0, np.nan) - 1).fillna(np.inf)  # new payers rank first
    else:
        inc = e["div_increase_5y"]
    e["r_inc"] = inc.rank(ascending=False, method="min")
    e["r_yield"] = e["yield_"].rank(ascending=False, method="min")
    e["r_payout"] = e["payout"].rank(ascending=True, method="min")
    e["score"] = e["r_inc"] + e["r_yield"] + e["r_payout"]
    e = e.sort_values(["score", "yield_"], ascending=[True, False])
    e["rank"] = np.arange(1, len(e) + 1)

    picked = list(e.index[: p.select_n])
    # Large-cap requirement: swap worst-ranked small/mid picks for best-ranked unpicked large caps
    def n_large(lst): return int(e.loc[lst, "large_cap"].sum())
    spare_large = [t for t in e.index[p.select_n:] if e.at[t, "large_cap"]]
    while n_large(picked) < p.min_large and spare_large:
        drop = [t for t in reversed(picked) if not e.at[t, "large_cap"]][0]
        picked.remove(drop)
        picked.append(spare_large.pop(0))
    # Industry cap: an over-limit industry loses its lowest-priority member (largest rank number)
    # to the next-ranked eligible security from another industry. Nasdaq's text says "highest-ranked";
    # validation against actual sub-portfolios shows this means the bottom of the order.
    for _ in range(200):
        counts = e.loc[picked, "icb"].value_counts()
        over = counts[counts > p.max_per_industry]
        if over.empty:
            break
        ind = over.index[0]
        top_member = max((t for t in picked if e.at[t, "icb"] == ind), key=lambda t: e.at[t, "rank"])
        repl = next((t for t in e.index if t not in picked and e.at[t, "icb"] != ind
                     and e.at[t, "icb"] not in counts[counts >= p.max_per_industry].index), None)
        if repl is None:
            break
        picked.remove(top_member)
        picked.append(repl)
    e["selected"] = e.index.isin(picked)
    return e
