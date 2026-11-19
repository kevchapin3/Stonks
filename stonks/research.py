"""Compare strategy variants on a universe of CSVs, against a buy-anytime baseline.

    python -m stonks.research --csv-dir data/prices --earnings-dir data/earnings \
        --market data/market_MID.csv

Variants (all with the earnings rule and the default 10% stop):
  all_patterns        all 8 bullish reversal patterns
  hammers             hammer + inverted hammer only
  hammers+market      same, buying only while the market index is above its
                      200-day average (needs --market; the default system)
Baseline:
  any_day             buy every stock on every day, hold the same number of
                      days as the variant's average trade (what you'd get
                      with no timing skill at all)
  any_day_uptrend     same, only on days passing the trend filter
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, earnings as er, indicators as ind, swing

SPLIT = pd.Timestamp("2020-01-01")


def load(csv_dir: str, earnings_dir: str, min_bars: int = 300):
    prices, earns = {}, {}
    for f in sorted(Path(csv_dir).glob("*.csv")):
        df = data.load_csv(str(f))
        if len(df) < min_bars:
            continue
        ef = Path(earnings_dir) / f.name
        if ef.exists():
            e = er.load_csv(str(ef))
            if e["eps_actual"].notna().sum() >= 5:
                earns[f.stem] = e
        prices[f.stem] = df
    return prices, earns


def summarize(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"trades": 0}
    r = t["return"]
    losses = r[r <= 0].sum()
    return {
        "trades": len(t),
        "win_rate": (r > 0).mean(),
        "avg_ret": r.mean(),
        "median_ret": r.median(),
        "profit_factor": r[r > 0].sum() / -losses if losses < 0 else np.inf,
        "avg_R": t["r_multiple"].mean(),
        "avg_days": t["bars_held"].mean(),
    }


def baseline(prices: dict, horizon: int, uptrend: bool, cost_bps: float) -> dict:
    """Average forward return over `horizon` bars from every eligible stock-day."""
    rets_early, rets_late = [], []
    cost = cost_bps / 10_000
    for df in prices.values():
        c = df["close"]
        fwd = (c.shift(-horizon) * (1 - cost)) / (c * (1 + cost)) - 1
        if uptrend:
            s50, s200 = ind.sma(c, 50), ind.sma(c, 200)
            fwd = fwd[(c > s200) & (s50 > s200)]
        fwd = fwd.dropna()
        rets_early.append(fwd[fwd.index < SPLIT])
        rets_late.append(fwd[fwd.index >= SPLIT])
    out = {}
    for name, parts in (("2012-19", rets_early), ("2020-26", rets_late)):
        r = pd.concat(parts)
        out[name] = {"trades": len(r), "win_rate": (r > 0).mean(), "avg_ret": r.mean(),
                     "median_ret": r.median(), "avg_days": float(horizon)}
    r = pd.concat(rets_early + rets_late)
    out["all"] = {"trades": len(r), "win_rate": (r > 0).mean(), "avg_ret": r.mean(),
                  "median_ret": r.median(), "avg_days": float(horizon)}
    return out


def run_variant(prices, earns, p: swing.SwingParams, use_earnings: bool,
                market: pd.DataFrame | None = None) -> pd.DataFrame:
    trades = []
    for t, df in prices.items():
        if use_earnings and t not in earns:
            continue
        res = swing.run(df, p, ticker=t, earnings=earns.get(t) if use_earnings else None,
                        market=market)
        if not res.trades.empty:
            trades.append(res.trades)
    if not trades:
        return pd.DataFrame()
    out = pd.concat(trades, ignore_index=True)
    return out[out["reason"] != "open"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--earnings-dir", required=True)
    ap.add_argument("--out", help="Write all trades to this CSV")
    ap.add_argument("--market", help="Market index CSV (e.g. S&P 400) for the 200-day filter")
    args = ap.parse_args(argv)

    prices, earns = load(args.csv_dir, args.earnings_dir)
    print(f"{len(prices)} stocks with prices, {len(earns)} with usable earnings history\n")

    base = swing.SwingParams()
    market = data.load_csv(args.market) if args.market else None
    all_pats = tuple(k for k in ("hammer", "bullish_engulfing", "piercing_line", "morning_star",
                                 "bullish_harami", "tweezer_bottom", "inverted_hammer", "dragonfly_doji"))
    variants = {
        "all_patterns": (replace(base, patterns=all_pats), True, None),
        "hammers": (base, True, None),
    }
    if market is not None:
        variants["hammers+market"] = (base, True, market)
    main_name = list(variants)[-1]
    rows, all_trades = [], []
    for name, (p, use_e, mkt) in variants.items():
        t = run_variant(prices, earns, p, use_e, mkt)
        t["variant"] = name
        all_trades.append(t)
        for period, sub in (("all", t), ("2012-19", t[t["entry_date"] < SPLIT]),
                            ("2020-26", t[t["entry_date"] >= SPLIT])):
            rows.append({"variant": name, "period": period, **summarize(sub)})

    horizon = int(round(pd.concat(all_trades)["bars_held"].mean()))
    for name, up in (("any_day", False), ("any_day_uptrend", True)):
        for period, stats in baseline(prices, horizon, up, base.cost_bps).items():
            rows.append({"variant": name, "period": period, **stats})

    table = pd.DataFrame(rows).set_index(["variant", "period"])
    pd.set_option("display.width", 160)
    fmt = {"win_rate": "{:.1%}", "avg_ret": "{:+.2%}", "median_ret": "{:+.2%}",
           "profit_factor": "{:.2f}", "avg_R": "{:+.2f}", "avg_days": "{:.0f}"}
    shown = table.copy()
    for col, f in fmt.items():
        if col in shown:
            shown[col] = shown[col].map(lambda x, f=f: "" if pd.isna(x) else f.format(x))
    print(f"Baseline holds {horizon} trading days (the average trade length).\n")
    print(shown.to_string())

    trades = pd.concat(all_trades, ignore_index=True)
    main_t = trades[trades["variant"] == main_name]
    if len(main_t):
        print(f"\n{main_name} by pattern:")
        print(swing.pattern_report(main_t).to_string(float_format=lambda x: f"{x:.3f}"))
        print(f"\n{main_name} by exit reason:")
        print(main_t.groupby("reason")["return"].agg(["count", "mean"]).to_string())
        print(f"\n{main_name} by entry year:")
        y = main_t.groupby(main_t["entry_date"].dt.year)["return"]
        print(pd.DataFrame({"trades": y.size(), "win_rate": y.apply(lambda r: (r > 0).mean()),
                            "avg_ret": y.mean()}).to_string(float_format=lambda x: f"{x:.3f}"))
    if args.out:
        trades.to_csv(args.out, index=False)
        print(f"\nTrades written to {args.out}")


if __name__ == "__main__":
    main()
