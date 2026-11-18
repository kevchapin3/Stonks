"""Command-line entry point.

Examples:
    python -m stonks --synthetic --strategy ma_cross
    python -m stonks --csv prices.csv --strategy rsi
    python -m stonks --ticker AAPL --start 2015-01-01 --strategy macd   # needs yfinance
"""

from __future__ import annotations

import argparse

from . import backtest, data, indicators, signals


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="stonks", description="Technical analysis backtester")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="Path to OHLCV CSV")
    src.add_argument("--ticker", help="Download from Yahoo Finance (requires yfinance)")
    src.add_argument("--synthetic", action="store_true", help="Use random-walk demo data")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--strategy", choices=sorted(signals.STRATEGIES), default="ma_cross")
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--all", action="store_true", help="Compare every strategy")
    args = p.parse_args(argv)

    if args.csv:
        df = data.load_csv(args.csv)
    elif args.ticker:
        df = data.load_yahoo(args.ticker, args.start, args.end)
    else:
        df = data.synthetic()

    print(f"{len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}\n")
    last = indicators.add_all(df).iloc[-1]
    print("Latest indicators:")
    for col in ["close", "SMA_50", "SMA_200", "RSI_14", "MACD", "MACD_signal", "ATR_14", "ADX", "BB_pctb"]:
        print(f"  {col:<12} {last[col]:.2f}")
    print()

    names = sorted(signals.STRATEGIES) if args.all else [args.strategy]
    for name in names:
        pos = signals.STRATEGIES[name](df)
        res = backtest.run(df, pos, cost_bps=args.cost_bps)
        print(f"== {name} ==\n{res}\n")


if __name__ == "__main__":
    main()
