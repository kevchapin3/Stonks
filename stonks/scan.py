"""Mid-cap candlestick scanner and multi-ticker backtest.

Usage:
    python -m stonks.scan --tickers DECK FIX EME --start 2015-01-01        # yfinance
    python -m stonks.scan --tickers-file midcaps.txt --backtest
    python -m stonks.scan --csv-dir data/                                 # one CSV per ticker

Mid-cap screen (defaults; change with flags):
    * market cap $2B-$10B (checked through yfinance when available)
    * price >= $10
    * 50-day average dollar volume >= $10M, so 6-8 week positions can be
      entered and exited without moving the price
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import data, indicators as ind, swing

MIDCAP_MIN, MIDCAP_MAX = 2e9, 10e9


@dataclass
class Universe:
    min_cap: float = MIDCAP_MIN
    max_cap: float = MIDCAP_MAX
    min_price: float = 10.0
    min_dollar_volume: float = 10e6


def market_cap(ticker: str) -> float | None:
    try:
        import yfinance as yf

        return float(yf.Ticker(ticker).fast_info["marketCap"])
    except Exception:
        return None


def liquidity_ok(df: pd.DataFrame, u: Universe) -> tuple[bool, float]:
    dv = float((df["close"] * df["volume"]).tail(50).mean())
    return (df["close"].iloc[-1] >= u.min_price and dv >= u.min_dollar_volume), dv


def current_signal(df: pd.DataFrame, p: swing.SwingParams) -> dict | None:
    """Status of the system on the latest bar: new entry, open trade, or pending setup."""
    res = swing.run(df, p)
    last = df.index[-1]
    atr = ind.atr(df["high"], df["low"], df["close"], p.atr_period).iloc[-1]
    if not res.trades.empty:
        t = res.trades.iloc[-1]
        if t["reason"] == "open":
            status = "ENTRY TODAY" if t["entry_date"] == last else "IN TRADE"
            return {
                "status": status, "pattern": t["pattern"], "pattern_date": t["pattern_date"].date(),
                "entry": round(t["entry_price"], 2), "stop": round(t["initial_stop"], 2),
                "bars_held": int(t["bars_held"]), "close": round(df["close"].iloc[-1], 2),
            }
    st = swing.setups(df, p).tail(p.confirm_bars)
    hits = st[st["setup"]]
    if len(hits):
        d = hits.index[-1]
        trigger = df.loc[d, "high"]
        return {
            "status": "AWAITING CONFIRMATION", "pattern": hits["pattern"].iloc[-1],
            "pattern_date": d.date(), "entry": round(trigger, 2),
            "stop": round(max(df.loc[d, "low"] - p.stop_atr * atr, trigger - p.max_risk_atr * atr), 2),
            "bars_held": 0, "close": round(df["close"].iloc[-1], 2),
        }
    return None


def load_universe(args) -> dict[str, pd.DataFrame]:
    out = {}
    if args.csv_dir:
        for f in sorted(Path(args.csv_dir).glob("*.csv")):
            out[f.stem.upper()] = data.load_csv(str(f))
    else:
        tickers = list(args.tickers or [])
        if args.tickers_file:
            tickers += [t.strip().upper() for t in Path(args.tickers_file).read_text().split() if t.strip()]
        for t in tickers:
            try:
                out[t] = data.load_yahoo(t, args.start, args.end)
            except Exception as e:  # network, delisted, etc.
                print(f"  ! {t}: {e}")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="stonks.scan", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tickers", nargs="+")
    src.add_argument("--tickers-file")
    src.add_argument("--csv-dir")
    ap.add_argument("--start", default="2012-01-01")
    ap.add_argument("--end")
    ap.add_argument("--backtest", action="store_true", help="Also backtest the system over the whole universe")
    ap.add_argument("--no-cap-check", action="store_true", help="Skip the market-cap check (e.g. offline)")
    ap.add_argument("--min-dollar-volume", type=float, default=10e6)
    ap.add_argument("--min-hold", type=int, default=30)
    ap.add_argument("--max-hold", type=int, default=40)
    args = ap.parse_args(argv)

    u = Universe(min_dollar_volume=args.min_dollar_volume)
    p = swing.SwingParams(min_hold=args.min_hold, max_hold=args.max_hold)
    universe = load_universe(args)
    check_cap = not (args.no_cap_check or args.csv_dir)

    signals, all_trades, skipped = [], [], []
    for t, df in universe.items():
        if len(df) < 250:
            skipped.append((t, "under 1 year of history"))
            continue
        ok, dv = liquidity_ok(df, u)
        if not ok:
            skipped.append((t, f"illiquid / low price (${dv / 1e6:.1f}M/day)"))
            continue
        if check_cap:
            cap = market_cap(t)
            if cap is not None and not (u.min_cap <= cap <= u.max_cap):
                skipped.append((t, f"market cap ${cap / 1e9:.1f}B outside mid-cap range"))
                continue
        sig = current_signal(df, p)
        if sig:
            signals.append({"ticker": t, **sig})
        if args.backtest:
            all_trades.append(swing.run(df, p, ticker=t).trades)

    pd.set_option("display.width", 160)
    print(f"\nScanned {len(universe)} tickers, {len(skipped)} filtered out.")
    for t, why in skipped:
        print(f"  - {t}: {why}")
    print("\n=== Current candlestick signals ===")
    print(pd.DataFrame(signals).to_string(index=False) if signals else "  none")

    if args.backtest and all_trades:
        trades = pd.concat([t for t in all_trades if not t.empty], ignore_index=True) if any(
            not t.empty for t in all_trades) else pd.DataFrame()
        if trades.empty:
            print("\nNo historical trades.")
            return
        closed = trades[trades["reason"] != "open"]
        print(f"\n=== Backtest: {len(closed)} closed trades across {trades['ticker'].nunique()} tickers ===")
        print(f"Win rate {(closed['return'] > 0).mean():.1%} | avg return {closed['return'].mean():.2%} "
              f"| median {closed['return'].median():.2%} | avg R {closed['r_multiple'].mean():.2f} "
              f"| avg hold {closed['bars_held'].mean():.0f} bars")
        print("\nBy exit reason:")
        print(closed.groupby("reason")["return"].agg(["count", "mean"]).to_string())
        print("\nBy pattern:")
        print(swing.pattern_report(closed).to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
