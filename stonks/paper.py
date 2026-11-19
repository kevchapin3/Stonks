"""Paper-trading tracker for the large-cap pullback strategy.

Each run:
  1. downloads recent prices for the ticker list (and SPY as a benchmark),
  2. logs every new pullback signal (close crosses below SMA50 - k*std50 while
     above SMA200) for each k in `--k` (default 2 and 3), one position per
     ticker per strategy at a time,
  3. closes positions after `hold` trading days at that day's close,
  4. writes `ledger.csv` and `report.md` in the output folder.

Entry price is the signal day's close, matching the backtest. Because the
scan runs weekly, `scan_price` also records the last close when the signal
was first seen, so the cost of acting late is visible (`return_from_scan`).

    python -m stonks.paper --tickers-file universe/largecaps.txt --out paper
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from . import bands, data

COLS = ["strategy", "ticker", "signal_date", "entry_price", "scan_date", "scan_price",
        "status", "exit_date", "exit_price", "last_close", "days_held", "return",
        "return_from_scan", "spy_return"]


def _label(k: float) -> str:
    return f"{k:g}SD"


def update(prices: dict[str, pd.DataFrame], spy: pd.Series, ledger: pd.DataFrame,
           since: pd.Timestamp | None, ks=(2.0, 3.0), hold: int = 40,
           today: pd.Timestamp | None = None) -> pd.DataFrame:
    """Add new signals after `since` and update every position. Returns the new ledger."""
    led = ledger.copy() if len(ledger) else pd.DataFrame(columns=COLS)
    for c in ("signal_date", "scan_date", "exit_date"):
        led[c] = pd.to_datetime(led[c])
    rows = []
    for k in ks:
        p = bands.BandParams(k=k, hold=hold)
        for t, df in prices.items():
            if len(df) < p.trend_ma + 5:
                continue
            sig = bands.signals(df, p)
            new = sig[sig].index
            if since is not None:
                new = new[new > since]
            for d in new:
                open_now = led[(led.strategy == _label(k)) & (led.ticker == t)]
                busy = open_now[(open_now.signal_date <= d) &
                                ((open_now.status == "open") | (open_now.exit_date >= d))]
                pending = [r for r in rows if r["strategy"] == _label(k) and r["ticker"] == t]
                if len(busy) or pending:
                    continue
                rows.append(dict(strategy=_label(k), ticker=t, signal_date=d,
                                 entry_price=float(df["close"].loc[d]),
                                 scan_date=today or df.index[-1],
                                 scan_price=float(df["close"].iloc[-1]), status="open"))
    if rows:
        led = pd.concat([led, pd.DataFrame(rows)], ignore_index=True)

    # Update / close every open position
    for i, r in led[led.status == "open"].iterrows():
        df = prices.get(r.ticker)
        if df is None or r.signal_date not in df.index:
            continue
        pos = df.index.get_loc(r.signal_date)
        end = min(pos + hold, len(df) - 1)
        close = float(df["close"].iloc[end])
        led.loc[i, "last_close"] = close
        led.loc[i, "days_held"] = end - pos
        led.loc[i, "return"] = close / r.entry_price - 1
        led.loc[i, "return_from_scan"] = close / r.scan_price - 1
        s = spy.reindex(df.index).ffill()
        led.loc[i, "spy_return"] = s.iloc[end] / s.iloc[pos] - 1
        if pos + hold <= len(df) - 1:
            led.loc[i, ["status", "exit_date", "exit_price"]] = ["closed", df.index[end], close]
    return led[COLS].sort_values(["status", "signal_date", "ticker"], ascending=[False, False, True])


def report(led: pd.DataFrame, asof) -> str:
    def pct(x):
        return "" if pd.isna(x) else f"{x:+.1%}"

    out = [f"# Paper trading: large-cap pullbacks", "",
           f"As of the {pd.Timestamp(asof).date()} close. Entries at the signal-day close; "
           "exits after 40 trading days (~8 weeks); no stop. "
           "`From scan` is the return if bought at the close when the weekly scan first saw the signal. "
           "`SPY` is the S&P 500 over the same days.", "",
           "## Summary", "", "| Strategy | Closed | Win rate | Avg return | Avg SPY | Avg vs SPY | Open | Avg open P&L |",
           "|---|---|---|---|---|---|---|---|"]
    for s, g in led.groupby("strategy"):
        c, o = g[g.status == "closed"], g[g.status == "open"]
        if len(c):
            win = "{:.0%}".format((c["return"] > 0).mean())
            avg, spy_avg = pct(c["return"].mean()), pct(c["spy_return"].mean())
            vs = pct((c["return"] - c["spy_return"]).mean())
        else:
            win = avg = spy_avg = vs = ""
        open_pl = pct(o["return"].mean()) if len(o) else ""
        out.append(f"| {s} | {len(c)} | {win} | {avg} | {spy_avg} | {vs} | {len(o)} | {open_pl} |")
    for title, st in (("Open positions", "open"), ("Closed trades", "closed")):
        g = led[led.status == st]
        out += ["", f"## {title}", ""]
        if g.empty:
            out.append("None yet.")
            continue
        out += ["| Strategy | Ticker | Signal | Entry | " + ("Last close | Days | P&L" if st == "open" else "Exit date | Exit | Return")
                + " | From scan | SPY |", "|---|---|---|---|---|---|---|---|---|"]
        for _, r in g.iterrows():
            mid = (f"{r.last_close:.2f} | {int(r.days_held) if pd.notna(r.days_held) else ''} | {pct(r['return'])}"
                   if st == "open" else
                   f"{pd.Timestamp(r.exit_date).date()} | {r.exit_price:.2f} | {pct(r['return'])}")
            out.append(f"| {r.strategy} | {r.ticker} | {pd.Timestamp(r.signal_date).date()} | {r.entry_price:.2f} | "
                       f"{mid} | {pct(r.return_from_scan)} | {pct(r.spy_return)} |")
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tickers-file")
    src.add_argument("--csv-dir")
    ap.add_argument("--spy-csv", help="SPY prices (default: download)")
    ap.add_argument("--out", default="paper")
    ap.add_argument("--k", type=float, nargs="+", default=[2.0, 3.0])
    ap.add_argument("--first-lookback", type=int, default=5,
                    help="On the first run, also log signals from the last N sessions")
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ledger_path, state_path = out / "ledger.csv", out / "state.json"
    ledger = pd.read_csv(ledger_path) if ledger_path.exists() else pd.DataFrame(columns=COLS)
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    start = (pd.Timestamp.today() - pd.Timedelta(days=500)).strftime("%Y-%m-%d")
    if a.csv_dir:
        prices = {f.stem: data.load_csv(str(f)) for f in sorted(Path(a.csv_dir).glob("*.csv"))}
    else:
        import yfinance as yf

        tickers = sorted(set(Path(a.tickers_file).read_text().split()))
        raw = yf.download(tickers, start=start, auto_adjust=True, progress=False, group_by="ticker", threads=True)
        prices = {}
        for t in tickers:
            try:
                df = data.normalize(raw[t].dropna(how="all")).dropna()
                if len(df):
                    prices[t] = df
            except Exception as e:  # missing ticker etc.
                print("skip", t, e)
    spy = (data.load_csv(a.spy_csv) if a.spy_csv else data.load_yahoo("SPY", start))["close"]

    last = max(df.index[-1] for df in prices.values())
    if "last_price_date" in state:
        since = pd.Timestamp(state["last_price_date"])
    else:
        ref = prices.get("SPY", next(iter(prices.values()))).index
        since = ref[-a.first_lookback - 1] if len(ref) > a.first_lookback else None
    led = update(prices, spy, ledger, since, tuple(a.k), today=last)
    led.to_csv(ledger_path, index=False, float_format="%.4f")
    (out / "report.md").write_text(report(led, last))
    state.update(last_price_date=str(last.date()), last_run=str(date.today()),
                 tickers=len(prices))
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    new = led[pd.to_datetime(led.signal_date) > since] if since is not None else led
    print(f"Prices through {last.date()} for {len(prices)} tickers. New signals: {len(new)}. "
          f"Open: {(led.status == 'open').sum()}. Closed: {(led.status == 'closed').sum()}.")
    print((out / "report.md").read_text())


if __name__ == "__main__":
    main()
