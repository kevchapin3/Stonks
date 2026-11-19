"""RDVY / NQDVRIS: sub-portfolio inference, rebalance calendar and top-holding projection.

    python -m stonks.rdvy predict --holdings rdvy/holdings_2026-11-18.csv
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

SUB_BY_MONTH = {3: "A", 6: "B", 9: "C", 12: "D"}


def _last_bday(year: int, month: int) -> pd.Timestamp:
    return (pd.Timestamp(year, month, 1) + pd.offsets.BMonthEnd(0)).normalize()


def third_friday(year: int, month: int) -> pd.Timestamp:
    d = pd.Timestamp(year, month, 1)
    first_fri = d + pd.Timedelta(days=(4 - d.weekday()) % 7)
    return first_fri + pd.Timedelta(weeks=2)


@dataclass
class Rebalance:
    year: int
    month: int

    @property
    def sub(self) -> str:
        return SUB_BY_MONTH[self.month]

    @property
    def fundamentals_ref(self) -> pd.Timestamp:      # last trading day 3 months earlier
        m = self.month - 3 or 12
        y = self.year if self.month > 3 else self.year - 1
        return _last_bday(y, m)

    @property
    def price_ref(self) -> pd.Timestamp:             # last trading day of the prior month
        m = self.month - 1
        return _last_bday(self.year, m)

    @property
    def effective(self) -> pd.Timestamp:             # open of first trading day after 3rd Friday
        return third_friday(self.year, self.month) + pd.offsets.BDay(1)

    @property
    def announcement(self) -> pd.Timestamp:          # after the close, 6 trading days earlier
        return self.effective - pd.offsets.BDay(6)

    def __str__(self):
        return f"{self.year}-{self.month:02d} (sub-portfolio {self.sub})"


def next_rebalance(today: pd.Timestamp) -> Rebalance:
    for y in (today.year, today.year + 1):
        for m in (3, 6, 9, 12):
            r = Rebalance(y, m)
            if r.effective > today:
                return r
    raise RuntimeError


def last_ref_dates(asof: pd.Timestamp) -> dict[str, pd.Timestamp]:
    """Price reference date of each sub-portfolio's most recent rebuild on or before `asof`."""
    out = {}
    for y in (asof.year - 1, asof.year):
        for m, s in SUB_BY_MONTH.items():
            r = Rebalance(y, m)
            if r.effective <= asof:
                out[s] = r.price_ref
    return out


def infer_membership(weights: pd.Series, prices: pd.DataFrame, asof: pd.Timestamp,
                     refs: dict[str, pd.Timestamp] | None = None, iters: int = 50):
    """Decompose fund weights (percent) into sub-portfolio memberships.

    Model: weight_i = sum_s member_is * slice_s * price_i(asof) / price_i(ref_s).
    Returns (membership DataFrame of 0/1 with columns A-D, per-name slices in percent, fit error).
    """
    refs = refs or last_ref_dates(asof)
    subs = sorted(refs)
    px = prices.reindex(columns=weights.index)
    R = pd.DataFrame({s: px[:asof].iloc[-1] / px[:refs[s]].iloc[-1] for s in subs})
    R = R.fillna(1.0)
    w = (weights / weights.sum()).to_numpy()
    combos = [np.array(m) for m in itertools.product([0, 1], repeat=len(subs)) if any(m)]
    u = np.full(len(subs), 1 / 200)
    Rv = R.to_numpy()
    for _ in range(iters):
        M = np.array([combos[int(np.argmin([abs((m * u * Rv[i]).sum() - w[i]) for m in combos]))]
                      for i in range(len(w))])
        u_new, *_ = np.linalg.lstsq(M * Rv, w, rcond=None)
        if np.allclose(u_new, u, rtol=1e-7):
            break
        u = u_new
    fit = (M * u * Rv).sum(axis=1)
    mem = pd.DataFrame(M.astype(int), index=weights.index, columns=subs)
    return mem, pd.Series(u * 100 * weights.sum() / 100, index=subs), float(np.abs(fit - w).max() * 100)


def project(mem: pd.DataFrame, slices: pd.Series, prices: pd.DataFrame, asof: pd.Timestamp,
            reb: Rebalance, selected: set[str], spwc: float = 3.0) -> pd.DataFrame:
    """Projected weights right after `reb` takes effect.

    Non-rebuilt sub-portfolios keep their names and drift with price to the price reference
    date (or the latest price if that date is still ahead). The rebuilt one is split equally
    among `selected`. In March, all four sub-portfolios are first reset to 25% each.
    """
    refs = last_ref_dates(asof)
    end = min(reb.price_ref, prices.index[-1])
    px = prices.reindex(columns=mem.index)
    contrib = pd.DataFrame({s: mem[s] * slices[s] * px[:end].iloc[-1] / px[:refs[s]].iloc[-1]
                            for s in mem.columns}).fillna(0.0)
    sub_tot = contrib.sum()
    if reb.month == 3:
        contrib = contrib * (sub_tot.sum() / 4) / sub_tot
        sub_tot = contrib.sum()
    others = [s for s in contrib.columns if s != reb.sub]
    carried = contrib[others].sum(axis=1)
    capped = set(carried[carried >= spwc].index)
    sel = [t for t in selected if t not in capped]
    new_slice = sub_tot[reb.sub] / max(len(sel), 1)
    names = sorted(set(contrib.index) | set(sel))
    out = pd.DataFrame(index=names)
    out["carried"] = carried.reindex(names).fillna(0.0)
    out["n_other_subs"] = mem[others].sum(axis=1).reindex(names).fillna(0).astype(int)
    out["selected"] = out.index.isin(sel)
    out["weight"] = out["carried"] + np.where(out["selected"], new_slice, 0.0)
    out = out[out.weight > 0]
    total = out["weight"].sum()
    out["weight"] = out["weight"] / total * 100
    out["carried"] = out["carried"] / total * 100
    return out.sort_values("weight", ascending=False)


# --------------------------------------------------------------------------- #
# End-to-end: screen a rebuild, validate past rebuilds, project top holdings
# --------------------------------------------------------------------------- #

def run_screen(reb: Rebalance, fund, prices, earnings, params, overrides=None, exclude=()):
    from dataclasses import replace

    from . import nqdvris as nq
    p = replace(params, exclude=set(exclude))
    f = nq.features(fund, prices, str(reb.fundamentals_ref.date()), str(reb.price_ref.date()),
                    p, overrides, earnings)
    return nq.select(nq.screen(f, p), p), f


def validate(fund, prices, earnings, mem, params, overrides=None, asof=None) -> pd.DataFrame:
    """Re-run each of the last four rebuilds and compare picks with inferred members."""
    asof = asof or prices.index[-1]
    rows = []
    for s, ref in last_ref_dates(asof).items():
        reb = Rebalance(ref.year, ref.month + 1)   # rebuild month follows its price reference month
        assert reb.sub == s
        e, _ = run_screen(reb, fund, prices, earnings, params, overrides)
        picked = set(e.index[e.selected])
        actual = set(mem.index[mem[s] == 1])
        rows.append(dict(sub=s, rebuild=str(reb), actual=len(actual), picked=len(picked),
                         overlap=len(picked & actual),
                         missed=", ".join(sorted(actual - picked)), extra=", ".join(sorted(picked - actual))))
    return pd.DataFrame(rows).sort_values("rebuild")


# --------------------------------------------------------------------------- #
# Command line: forecast the top holdings for the next rebalance
# --------------------------------------------------------------------------- #

HOLDINGS_URL = "https://www.ftportfolios.com/retail/etf/etfholdings.aspx?Ticker=RDVY"


def fetch_holdings(url: str = HOLDINGS_URL) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Download RDVY's full holdings table from First Trust (ticker, name, sector, weight %)."""
    import html
    import re
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    s = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "ignore")
    t = html.unescape(re.sub(r"<[^>]+>", "|", re.sub(r"(?s)<script.*?</script>|<style.*?</style>", "", s)))
    t = re.sub(r"\|\s*\|+", "|", re.sub(r"[ \t\r\n]+", " ", t))
    rows = re.findall(r"\|([^|]+)\|([A-Z.\-]{1,6})\|([0-9A-Z]{9})\|([^|]+)\|([\d,]+)\|\$([\d,\.]+)\|([\d\.]+)%", t)
    df = pd.DataFrame(rows, columns=["name", "ticker", "cusip", "sector", "shares", "market_value", "weight"])
    df["weight"] = df["weight"].astype(float)
    m = re.search(r"as of (\d{1,2}/\d{1,2}/\d{4})", t)
    asof = pd.Timestamp(m.group(1)) if m else pd.Timestamp.today().normalize()
    return df, asof


def forecast(holdings: pd.DataFrame, asof: pd.Timestamp, prices: pd.DataFrame, top: int = 10) -> tuple[Rebalance, pd.DataFrame]:
    """Top holdings after the next rebalance, assuming names held in all three non-rebuilding
    sub-portfolios are re-selected (Nasdaq re-picked ~91% of such names in 2026)."""
    w = holdings.set_index("ticker")["weight"]
    mem, slices, err = infer_membership(w, prices, asof)
    reb = next_rebalance(asof)
    others = [s for s in mem.columns if s != reb.sub]
    persistent = set(mem.index[mem[others].sum(axis=1) == len(others)])
    filler = {f"_new{i}" for i in range(max(0, 50 - len(persistent)))}
    pr = project(mem, slices, prices, asof, reb, persistent | filler)
    pr = pr.drop(index=[t for t in pr.index if t.startswith("_new")])
    out = pr.head(top)[["carried", "weight"]].rename(columns={"carried": "if_not_repicked", "weight": "if_repicked"})
    out.attrs.update(fit_error=err, members=mem)
    return reb, out


def main(argv=None):
    import argparse

    import yfinance as yf

    ap = argparse.ArgumentParser(description="Forecast RDVY's top holdings for the next rebalance")
    ap.add_argument("--holdings", help="Holdings CSV (ticker, weight); default: download from First Trust")
    ap.add_argument("--asof", help="Holdings date (YYYY-MM-DD) when using --holdings")
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args(argv)

    if a.holdings:
        h, asof = pd.read_csv(a.holdings), pd.Timestamp(a.asof or pd.Timestamp.today().normalize())
    else:
        h, asof = fetch_holdings()
    start = (asof - pd.DateOffset(months=14)).strftime("%Y-%m-%d")
    raw = yf.download([t.replace(".", "-") for t in h.ticker], start=start, auto_adjust=False,
                      progress=False, group_by="ticker", threads=True)
    prices = pd.DataFrame({t: raw[t.replace(".", "-")]["Close"] for t in h.ticker
                           if t.replace(".", "-") in raw.columns.get_level_values(0)})
    reb, out = forecast(h, asof, prices, a.top)
    print(f"Holdings as of {asof.date()} ({len(h)} names). Sub-portfolio fit error {out.attrs['fit_error']:.3f} pp.")
    print(f"Next rebalance: {reb}. Fundamentals as of {reb.fundamentals_ref.date()}, weights set from "
          f"{reb.price_ref.date()} prices, announced after the close {reb.announcement.date()}, "
          f"effective at the open {reb.effective.date()}.")
    print("Projected weights (%) right after the rebalance:")
    print(out.round(2).to_string())


if __name__ == "__main__":
    main()
