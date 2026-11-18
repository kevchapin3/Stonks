# Mid-cap universe

`midcaps.txt`: 400 US mid caps listed on NYSE or Nasdaq, taken from a [Bigdata.com](https://bigdata.com)
company screen run on 2026-11-18.

Screen: market cap $2B-$10B, price >= $10, daily volume >= 300k shares, actively
trading, no ETFs or funds.

This is a **sample**, not the full mid-cap universe. The screener returns
companies largest-first, 100 per page, and these four pages were pulled:

| Market-cap band | Tickers |
|---|---|
| $7.7B-$10B | 100 (complete band) |
| $6.2B-$7.7B | 100 (complete band) |
| $5.1B-$6.2B | 100 (top of the $4B-$6.2B band) |
| $2.8B-$3.5B | 100 (top of the $2B-$3.5B band) |

Gaps: $3.5B-$5.1B and $2B-$2.8B. Fill them before relying on results.

Membership is today's. Backtesting today's members over past years adds
survivorship bias (see the main README).
