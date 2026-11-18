# Stonks

Technical analysis in Python, built around **Japanese candlestick swing trades
in mid-cap stocks held for 6-8 weeks**.

## Layout

| Module | What it does |
|---|---|
| `stonks/candlesticks.py` | 22 candlestick patterns (hammer, engulfing, harami, piercing line, dark cloud, morning/evening star, three soldiers/crows, tweezers, doji family, marubozu), each checked against the trend it is reversing; weekly resampling |
| `stonks/swing.py` | The 6-8 week system: trend filter, pattern, volume, confirmation, ATR stops, time stop; trade-by-trade simulation and per-pattern report |
| `stonks/scan.py` | Mid-cap scanner: market cap, price and liquidity screen, today's signals, multi-ticker backtest |
| `stonks/indicators.py` | SMA/EMA/WMA, RSI, MACD, Stochastic, ROC, Williams %R, CCI, ATR, Bollinger, Keltner, Donchian, ADX, OBV, VWAP, MFI |
| `stonks/signals.py`, `stonks/backtest.py` | Classic indicator strategies and a vectorized backtester |
| `stonks/data.py` | CSV loader, Yahoo Finance loader, synthetic data |

## The swing system

**Entry** (long only):
1. Uptrend: close > 200-day SMA and 50-day SMA > 200-day SMA. Bullish candles are treated as pullback entries in an uptrend.
2. A bullish reversal pattern after a short-term pullback.
3. Pattern-bar volume ≥ 1.2× its 20-day average.
4. Confirmation: a close above the pattern bar's high within 3 bars. Entry is at that close.

**Exit**, whichever comes first:
- Initial stop: pattern low − 0.5 ATR, but never more than 3 ATR below entry.
- Trailing chandelier stop: highest close − 3 ATR.
- A bearish reversal pattern after 30 bars (about 6 weeks).
- Time stop at 40 bars (about 8 weeks).

Costs default to 10 bps per side for mid-cap spreads and slippage. All of these
settings live in `SwingParams`.

**Mid-cap screen:** $2B–$10B market cap, price ≥ $10, 50-day average dollar
volume ≥ $10M.

## Usage

```bash
pip install -r requirements.txt

# Scan a list of tickers (one per line) and backtest the system on them
python -m stonks.scan --tickers-file midcaps.txt --start 2012-01-01 --backtest

# Or use your own CSVs (Date,Open,High,Low,Close[,Adj Close],Volume), one per ticker
python -m stonks.scan --csv-dir data/ --backtest

# Classic indicator strategies on a single series
python -m stonks --ticker DECK --all
python -m stonks --synthetic --all

pytest
```

```python
from stonks import data, candlesticks as cs, swing

df = data.load_csv("DECK.csv")
cs.detect_all(df).tail()                 # which patterns fired on each bar
res = swing.run(df, swing.SwingParams(max_hold=40))
print(res); res.trades.tail()
swing.pattern_report(res.trades)         # which patterns actually make money
```

## Caveats

- Backtest the system on a broad mid-cap universe before trusting any single
  pattern. `pattern_report` shows which patterns earn their keep.
- Testing today's mid-cap list on past data adds survivorship bias. Companies
  that grew into large caps or went under are missing.
- Earnings land inside most 6-8 week holds. Consider checking the earnings
  calendar before entering a trade.
