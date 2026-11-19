# Stonks

Technical analysis in Python, built around **long-only Japanese candlestick
swing trades in mid-cap stocks held for 6-8 weeks, timed around earnings**.
Nothing in this code shorts. Bearish patterns only close a stock you own.

## Layout

| Module | What it does |
|---|---|
| `stonks/candlesticks.py` | 22 candlestick patterns (hammer, engulfing, harami, piercing line, dark cloud, morning/evening star, three soldiers/crows, tweezers, doji family, marubozu), each checked against the trend it is reversing; weekly resampling |
| `stonks/swing.py` | The 6-8 week system: trend filter, pattern, volume, confirmation, ATR stops, time stop; trade-by-trade simulation and per-pattern report |
| `stonks/earnings.py` | Earnings history (Yahoo or CSV), point-in-time EPS surprise, year-over-year growth and days to the next report |
| `stonks/momentum.py` | Earnings-growth momentum: consecutive quarter-over-quarter EPS gains plus RSI crossing 60 |
| `stonks/bands.py` | Entries 2 standard deviations below (pullback) or above (breakout) the 50-day average, in a 200-day uptrend |
| `stonks/paper.py` | Weekly paper-trading tracker for 2-SD and 3-SD large-cap pullbacks; writes `paper/ledger.csv` and `paper/report.md` |
| `stonks/timing.py` | 200-day moving-average timing for an index fund (MDY, SPY) vs buy-and-hold, with T-bill yield on cash |
| `stonks/scan.py` | Mid-cap scanner: market cap, price and liquidity screen, today's signals, multi-ticker backtest |
| `stonks/indicators.py` | SMA/EMA/WMA, RSI, MACD, Stochastic, ROC, Williams %R, CCI, ATR, Bollinger, Keltner, Donchian, ADX, OBV, VWAP, MFI |
| `stonks/signals.py`, `stonks/backtest.py` | Classic indicator strategies and a vectorized backtester |
| `stonks/data.py` | CSV loader, Yahoo Finance loader, synthetic data |

## The swing system

**Entry** (long only):
1. Uptrend: close > 200-day SMA and 50-day SMA > 200-day SMA. Bullish candles are treated as pullback entries in an uptrend.
2. A **hammer** or **inverted hammer** after a short-term pullback (the other
   patterns are still available through `SwingParams(patterns=...)`).
3. **Market filter:** the S&P 400 mid-cap index (`^MID`) closes above its
   200-day average.
4. Pattern-bar volume ≥ 1.2× its 20-day average.
5. Confirmation: a close above the pattern bar's high within 3 bars. Entry is at that close.
6. **Earnings rule:** the latest quarterly report beat the EPS estimate and grew
   EPS year over year. The entry comes within 20 trading days of that report.

**Exit**, whichever comes first:
- Stop: 10% below the entry price (`stop_pct`). An ATR-based stop and a
  trailing stop are available but off by default.
- A bearish reversal pattern after 30 bars (about 6 weeks).
- Time stop at 40 bars (about 8 weeks).
- Trades may hold through the next earnings report. To sell before it instead,
  set `SwingParams(exit_before_earnings=True, min_bars_to_earnings=25)`.

Why earnings: prices follow profits, and stocks that beat estimates tend to keep
drifting up for weeks afterward (post-earnings drift). Buying within 4 weeks
of a beat puts the entry inside that drift window.

Costs default to 10 bps per side for mid-cap spreads and slippage. All of these
settings live in `SwingParams`.

**Mid-cap screen:** $2B–$10B market cap, price ≥ $10, 50-day average dollar
volume ≥ $10M.

## Usage

```bash
pip install -r requirements.txt

# Scan the 400-stock mid-cap sample and backtest the system (prices + earnings from Yahoo)
python -m stonks.scan --tickers-file universe/midcaps.txt --start 2012-01-01 --backtest

# Compare variants against a buy-any-day baseline on cached CSVs
python -m stonks.research --csv-dir data/prices --earnings-dir data/earnings --market data/market_MID.csv

# Or use your own CSVs, one per ticker:
#   prices:   Date,Open,High,Low,Close[,Adj Close],Volume
#   earnings: date,eps_actual,eps_estimate[,timing]   (timing = bmo / amc)
python -m stonks.scan --csv-dir data/ --earnings-dir earnings/ --backtest

# Classic indicator strategies on a single series
python -m stonks --ticker DECK --all
python -m stonks --synthetic --all

pytest
```

```python
from stonks import data, candlesticks as cs, swing

df = data.load_csv("DECK.csv")
cs.detect_all(df).tail()                 # which patterns fired on each bar
from stonks import earnings as er
eps = er.load_yahoo("DECK")              # or er.load_csv("DECK_earnings.csv")
res = swing.run(df, swing.SwingParams(), earnings=eps)
print(res); res.trades.tail()
swing.pattern_report(res.trades)         # which patterns actually make money
```

## Caveats

- Backtest the system on a broad mid-cap universe before trusting any single
  pattern. `pattern_report` shows which patterns earn their keep.
- Testing today's mid-cap list on past data adds survivorship bias. Companies
  that grew into large caps or went under are missing.
- Yahoo's earnings history only goes back a few years for many stocks. A paid
  source with a longer EPS history will give more trades to test.
