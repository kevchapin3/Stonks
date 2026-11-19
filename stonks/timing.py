"""200-day moving-average timing for an index fund vs buy-and-hold.

Rule: at each close, if the fund closes above its `ma`-day average, hold it
the next day; otherwise hold cash (earning the T-bill rate) the next day.
`band` adds a buffer: buy only above MA*(1+band), sell only below MA*(1-band),
which cuts down on whipsaws. `monthly=True` checks only at month-end.

    python -m stonks.timing data/etf_MDY.csv --tbill data/tbill_IRX.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import indicators as ind

DAYS = 252


def load(path: str) -> pd.Series:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    return df["close"].dropna()


def positions(close: pd.Series, ma: int = 200, band: float = 0.0, monthly: bool = False) -> pd.Series:
    """1 = in the fund, 0 = in cash, decided at each close (applied next day)."""
    sma = ind.sma(close, ma)
    upper, lower = sma * (1 + band), sma * (1 - band)
    pos, state = [], 0.0
    month_end = close.index.to_series().dt.month != close.index.to_series().shift(-1).dt.month
    for px, up, lo, me in zip(close.to_numpy(), upper.to_numpy(), lower.to_numpy(), month_end.to_numpy()):
        if not np.isnan(up) and (me or not monthly):
            if px > up:
                state = 1.0
            elif px < lo:
                state = 0.0
        pos.append(state)
    return pd.Series(pos, index=close.index, name="position")


def backtest(close: pd.Series, pos: pd.Series, tbill_pct: pd.Series | None = None,
             cost_bps: float = 5.0) -> pd.Series:
    """Daily strategy returns. `tbill_pct` is the annual T-bill yield in percent."""
    ret = close.pct_change().fillna(0.0)
    held = pos.shift(1).fillna(0.0)
    cash = (tbill_pct.reindex(close.index).ffill().fillna(0.0) / 100 / DAYS) if tbill_pct is not None \
        else pd.Series(0.0, index=close.index)
    switches = pos.diff().abs().fillna(0.0).shift(1).fillna(0.0)
    return held * ret + (1 - held) * cash - switches * cost_bps / 10_000


def stats(daily: pd.Series, held: pd.Series | None = None) -> dict:
    eq = (1 + daily).cumprod()
    years = len(daily) / DAYS
    yearly = (1 + daily).groupby(daily.index.year).prod() - 1
    out = {
        "CAGR": eq.iloc[-1] ** (1 / years) - 1,
        "Total": eq.iloc[-1] - 1,
        "Volatility": daily.std() * np.sqrt(DAYS),
        "Sharpe": daily.mean() / daily.std() * np.sqrt(DAYS) if daily.std() else 0.0,
        "Max drawdown": (eq / eq.cummax() - 1).min(),
        "Worst year": yearly.min(),
        "Best year": yearly.max(),
    }
    if held is not None:
        out["Time invested"] = held.mean()
        out["Switches/yr"] = held.diff().abs().sum() / years
    return out


def compare(close: pd.Series, tbill: pd.Series | None, start: str, end: str | None = None,
            **rule) -> pd.DataFrame:
    pos = positions(close, **rule)          # computed on full history so the MA is warm
    daily_bh = close.pct_change().fillna(0.0)
    daily_tm = backtest(close, pos, tbill)
    sl = slice(start, end)
    return pd.DataFrame({
        "Buy & hold": stats(daily_bh[sl]),
        "200-day timing": stats(daily_tm[sl], pos.shift(1)[sl]),
    })


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fund_csv")
    ap.add_argument("--tbill", help="CSV of the 13-week T-bill yield (^IRX), percent")
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--ma", type=int, default=200)
    ap.add_argument("--band", type=float, default=0.0)
    ap.add_argument("--monthly", action="store_true")
    a = ap.parse_args(argv)
    close = load(a.fund_csv)
    tb = load(a.tbill) if a.tbill else None
    t = compare(close, tb, a.start, ma=a.ma, band=a.band, monthly=a.monthly)
    pct = ["CAGR", "Total", "Volatility", "Max drawdown", "Worst year", "Best year", "Time invested"]
    print(t.apply(lambda col: ["" if pd.isna(v) else f"{v:.1%}" if k in pct else f"{v:.2f}" for k, v in col.items()]).to_string())


if __name__ == "__main__":
    main()
