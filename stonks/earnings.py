"""Earnings data and the earnings rule.

The idea: stocks follow profits. After a company reports better-than-expected
earnings, its price tends to keep drifting in the direction of the surprise
for weeks (post-earnings announcement drift). Quarterly reports are about
63 trading days apart, which leaves room for a 6-8 week hold *between* reports
without taking the gamble of holding through one.

So the rule is:
  * enter only within `post_window` bars after a report where EPS beat the
    estimate (and, optionally, grew year over year);
  * never hold through the next report: exit `exit_buffer` bars before it.

Earnings tables need at least a `date` column, plus `eps_actual` and
`eps_estimate` for the profit filters. An optional `timing` column ("bmo" =
before market open, "amc" = after market close) sets which bar first
reflects the news. Without it, reports are assumed to be after the close,
which is the conservative choice (no look-ahead).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

QUARTER_BARS = 63


def normalize(e: pd.DataFrame) -> pd.DataFrame:
    e = e.copy()
    e.columns = [str(c).strip().lower().replace(" ", "_") for c in e.columns]
    e = e.rename(columns={"reported_eps": "eps_actual", "eps": "eps_actual", "estimate": "eps_estimate"})
    dates = pd.to_datetime(e["date"])
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    e["date"] = dates.dt.normalize()
    for col in ("eps_actual", "eps_estimate"):
        if col not in e.columns:
            e[col] = np.nan
    if "timing" not in e.columns:
        e["timing"] = "amc"
    e["timing"] = e["timing"].fillna("amc").str.lower()
    return e[["date", "eps_actual", "eps_estimate", "timing"]].sort_values("date").drop_duplicates("date").reset_index(drop=True)


def load_csv(path: str) -> pd.DataFrame:
    return normalize(pd.read_csv(path))


def load_yahoo(ticker: str, limit: int = 100) -> pd.DataFrame:
    """Past and upcoming earnings dates with EPS actual/estimate from Yahoo (yfinance)."""
    import yfinance as yf

    raw = yf.Ticker(ticker).get_earnings_dates(limit=limit)
    if raw is None or raw.empty:
        return normalize(pd.DataFrame(columns=["date"]))
    ts = raw.index.tz_convert("America/New_York")
    return normalize(
        pd.DataFrame(
            {
                "date": ts.tz_localize(None),
                "eps_actual": raw["Reported EPS"].to_numpy(),
                "eps_estimate": raw["EPS Estimate"].to_numpy(),
                # Pre-market times are "bmo"; after-close or unknown (midnight) times
                # are treated as "amc", which never reacts a day early.
                "timing": np.where((ts.hour >= 4) & (ts.hour < 12), "bmo", "amc"),
            }
        )
    )


def align(prices: pd.DataFrame, earnings: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time earnings features for every price bar.

    Columns:
      bars_since   bars since the last report's reaction bar (0 on that bar)
      bars_to_next bars until the next reaction bar (estimated as last + 63
                   when the next date is unknown)
      surprise_pct (actual - estimate) / |estimate| for the last report
      beat         last report beat the estimate
      eps_growth   last report's EPS above the same quarter a year earlier
      reaction_ret close-to-close return on the last reaction bar
    """
    idx = prices.index
    n = len(idx)
    e = normalize(earnings)
    # Reaction bar: first bar on/after the date for "bmo", strictly after for "amc".
    side = np.where(e["timing"] == "bmo", "left", "right")
    react = np.array(
        [idx.searchsorted(d, side=s) for d, s in zip(e["date"], side)], dtype=int
    )
    e["react"] = react
    e = e.drop_duplicates("react", keep="last")
    e["surprise_pct"] = (e["eps_actual"] - e["eps_estimate"]) / e["eps_estimate"].abs()
    year_ago = e["eps_actual"].shift(4)
    e["eps_growth"] = (e["eps_actual"] > year_ago).astype(float).where(
        year_ago.notna() & e["eps_actual"].notna()
    )
    close = prices["close"].to_numpy(float)

    bars = np.arange(n)
    reacts = e["react"].to_numpy()                      # non-decreasing (sorted by date)
    rep = e[e["eps_actual"].notna()]                    # reports with an actual EPS number
    rep_reacts = rep["react"].to_numpy()

    # Last reported reaction bar at or before each bar.
    k = np.searchsorted(rep_reacts, bars, side="right") - 1
    has_past = k >= 0
    kk = np.where(has_past, k, 0)
    last_r = np.where(has_past, rep_reacts[kk] if len(rep_reacts) else 0, -1)

    def pick(col):
        vals = rep[col].to_numpy(float) if len(rep) else np.zeros(1)
        return np.where(has_past, vals[kk], np.nan)

    beat = np.where(rep["eps_estimate"].notna(), rep["eps_actual"] > rep["eps_estimate"], np.nan) \
        if len(rep) else np.zeros(1)
    prev_close = np.concatenate([[np.nan], close[:-1]])
    react_ret = close / prev_close - 1
    rr = np.where(has_past & (last_r > 0) & (last_r < n), react_ret[np.clip(last_r, 0, n - 1)], np.nan)

    # Next reaction bar strictly after each bar (scheduled or reported).
    j = np.searchsorted(reacts, bars, side="right")
    nxt = np.where(j < len(reacts), reacts[np.minimum(j, len(reacts) - 1)] if len(reacts) else n, n)
    to_next = np.where(nxt < n, nxt - bars, np.nan)
    est = np.maximum(QUARTER_BARS - (bars - last_r), 1).astype(float)
    to_next = np.where(np.isnan(to_next) & has_past, est, to_next)

    return pd.DataFrame(
        {
            "bars_since": np.where(has_past, bars - last_r, np.nan),
            "bars_to_next": to_next,
            "surprise_pct": pick("surprise_pct"),
            "beat": np.where(has_past, np.asarray(beat, float)[kk], np.nan),
            "eps_growth": pick("eps_growth"),
            "reaction_ret": rr,
        },
        index=idx,
    )


def synthetic(prices: pd.DataFrame, seed: int = 0, beat_prob: float = 0.6) -> pd.DataFrame:
    """Quarterly reports every ~63 bars with random EPS, for tests and demos."""
    rng = np.random.default_rng(seed)
    pos = np.arange(int(rng.integers(5, 40)), len(prices), QUARTER_BARS)
    eps = 1.0 + np.cumsum(rng.normal(0.02, 0.08, len(pos)))
    est = eps - np.where(rng.random(len(pos)) < beat_prob, 1, -1) * np.abs(rng.normal(0.05, 0.03, len(pos)))
    return normalize(pd.DataFrame({"date": prices.index[pos], "eps_actual": eps, "eps_estimate": est,
                                   "timing": rng.choice(["bmo", "amc"], len(pos))}))
