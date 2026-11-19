# Mid-cap universe

`midcaps.txt`: 692 US mid caps listed on NYSE or Nasdaq, taken from a [Bigdata.com](https://bigdata.com)
company screens run on 2026-11-18 and 2026-11-19.

Screen: market cap $2B-$10B, price >= $10, daily volume >= 300k shares, actively
trading, no ETFs or funds.

The list covers every company the screen returned from $2B to $10B. It was
pulled in market-cap bands, 100 companies per page:

| Market-cap band | Pulled |
|---|---|
| $6.2B-$10B | 2026-11-18 |
| $5.1B-$6.2B | 2026-11-18 |
| $3.5B-$5.1B | 2026-11-19 |
| $2.8B-$3.5B | 2026-11-18 |
| $2.0B-$2.8B | 2026-11-19 |

Market caps move daily, so a few names near a band edge may be missing or
doubled up between pulls; duplicates were removed.

Membership is today's. Backtesting today's members over past years adds
survivorship bias (see the main README).
