"""Standard-deviation band entries around the 50-day average, in a 200-day uptrend.

    pullback:  close crosses below SMA50 - k * std50   (buy the dip)
    breakout:  close crosses above SMA50 + k * std50   (buy the strength)

Both require close > SMA200 on the signal day; optionally the market index
above its own 200-day average. Exits via `momentum.simulate` (stop or hold).

RSI confirmation (`rsi_confirm`):
    "same_day": buy only if RSI(fast) > RSI(slow) on the band-cross day
    "cross":    after the band cross, wait up to `confirm_window` days for
                RSI(fast) to cross above RSI(slow) and buy that day (price must
                still be above SMA200)
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import indicators as ind
from .momentum import simulate
from .swing import market_ok


@dataclass
class BandParams:
    side: str = "pullback"   # or "breakout"
    k: float = 2.0
    ma: int = 50
    trend_ma: int = 200
    market_ma: int = 200
    stop_pct: float = 0.10
    hold: int = 40
    cost_bps: float = 10.0
    rsi_confirm: str | None = None   # None, "same_day" or "cross"
    rsi_fast: int = 14
    rsi_slow: int = 21
    confirm_window: int = 10


def signals(df: pd.DataFrame, p: BandParams, market: pd.DataFrame | None = None) -> pd.Series:
    c = df["close"]
    mid = ind.sma(c, p.ma)
    sd = c.rolling(p.ma, min_periods=p.ma).std(ddof=0)
    if p.side == "pullback":
        band = mid - p.k * sd
        hit = (c < band) & (c.shift(1) >= band.shift(1))
    elif p.side == "breakout":
        band = mid + p.k * sd
        hit = (c > band) & (c.shift(1) <= band.shift(1))
    else:
        raise ValueError(p.side)
    above200 = c > ind.sma(c, p.trend_ma)
    ok = hit & above200
    if p.rsi_confirm:
        fast, slow = ind.rsi(c, p.rsi_fast), ind.rsi(c, p.rsi_slow)
        if p.rsi_confirm == "same_day":
            ok &= fast > slow
        elif p.rsi_confirm == "cross":
            cross = (fast > slow) & (fast.shift(1) <= slow.shift(1))
            # a band hit in the last `confirm_window` days (including today)
            recent = ok.astype(int).rolling(p.confirm_window + 1, min_periods=1).max().astype(bool)
            ok = cross & recent & above200
        else:
            raise ValueError(p.rsi_confirm)
    if market is not None:
        ok &= pd.Series(market_ok(market, df.index, p.market_ma), index=df.index)
    return ok.fillna(False)


def run(df: pd.DataFrame, p: BandParams | None = None, market: pd.DataFrame | None = None,
        ticker: str = "") -> pd.DataFrame:
    p = p or BandParams()
    return simulate(df, signals(df, p, market).to_numpy(), p.stop_pct, p.hold, p.cost_bps, ticker)


def scan(prices: dict[str, pd.DataFrame], p: BandParams | None = None, days: int = 5) -> pd.DataFrame:
    """Stocks whose band signal fired in the last `days` sessions."""
    p = p or BandParams()
    rows = []
    for t, df in prices.items():
        if len(df) < max(p.trend_ma, p.ma) + 5:
            continue
        sig = signals(df, p)
        hits = sig[sig].index
        if len(hits) and hits[-1] >= df.index[-days]:
            c = df["close"]
            mid = ind.sma(c, p.ma).iloc[-1]
            sd = c.rolling(p.ma).std(ddof=0).iloc[-1]
            rows.append(dict(ticker=t, signal_date=hits[-1].date(),
                             signal_close=round(c.loc[hits[-1]], 2), last_close=round(c.iloc[-1], 2),
                             z_now=round((c.iloc[-1] - mid) / sd, 2),
                             pct_above_200d=f"{c.iloc[-1] / ind.sma(c, p.trend_ma).iloc[-1] - 1:+.1%}",
                             sell_by=(hits[-1] + pd.tseries.offsets.BDay(p.hold)).date()))
    return pd.DataFrame(rows).sort_values("signal_date", ascending=False) if rows else pd.DataFrame()


def main(argv=None):
    import argparse
    from pathlib import Path

    from . import data

    ap = argparse.ArgumentParser(description="Scan for 2-SD pullbacks above the 200-day average")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv-dir")
    src.add_argument("--tickers-file")
    ap.add_argument("--side", default="pullback", choices=["pullback", "breakout"])
    ap.add_argument("--days", type=int, default=5, help="Show signals from the last N sessions")
    a = ap.parse_args(argv)
    if a.csv_dir:
        prices = {f.stem: data.load_csv(str(f)) for f in sorted(Path(a.csv_dir).glob("*.csv"))}
    else:
        prices = {}
        for t in Path(a.tickers_file).read_text().split():
            try:
                prices[t] = data.load_yahoo(t, "2024-01-01")
            except Exception as e:
                print("skip", t, e)
    out = scan(prices, BandParams(side=a.side), a.days)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False) if len(out) else "No signals.")


if __name__ == "__main__":
    main()
