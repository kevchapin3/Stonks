# RDVY: Nasdaq US Rising Dividend Achievers Index (NQDVRIS)

Source: Nasdaq index methodology (indexes.nasdaqomx.com/docs/methodology_NQDVRIS.pdf, 2025 edition)
and the First Trust RDVY fund page.

## Structure
Four sub-portfolios (A, B, C, D) of up to 50 equal-weighted stocks each. One is rebuilt per quarter:
A in March, B in June, C in September, D in December. A stock's fund weight is the sum of its
weights in every sub-portfolio it belongs to (about 0.5% per sub-portfolio, then drifting with price).
Each March the four sub-portfolios are reset to 25% each.

## Eligibility (applied to the rebuilding sub-portfolio)
- In the Nasdaq US Benchmark Index; common stock; one security per company
- Not a REIT or mortgage REIT (ICB)
- Top 750 by free-float market cap; 3-month average daily value traded >= $5M
- TTM dividend > TTM dividend 3 years ago and 5 years ago
- TTM EPS positive and > TTM EPS 3 years ago
- Cash-to-debt ratio > 50%
- TTM payout ratio <= 65%
- Combined weight >= 3% in the three other sub-portfolios -> ineligible (sub-portfolio weight cap)

## Selection
Rank by (1) 5-year dollar dividend increase, descending; (2) dividend yield, descending;
(3) payout ratio, ascending. Sum the ranks; lowest 50 selected (tie: higher yield wins).
At least 33 must be large-cap (NQUSB definition). Max 15 per ICB industry (highest-ranked
member of an over-limit industry is swapped for the next-ranked name from another industry).

## Calendar
- Fundamentals reference: last trading day of Dec / Mar / Jun / Sep (for the Mar / Jun / Sep / Dec rebuild)
- Market cap, liquidity and weighting prices: last trading day of Feb / May / Aug / Nov
- Announcement: after the close, 6 trading days before the effective date
- Effective: market open on the first trading day after the third Friday of Mar / Jun / Sep / Dec

## Files
- `holdings_2026-11-18.csv`: First Trust full holdings snapshot
- `membership_2026-11-18.csv`: inferred sub-portfolio membership (fit error < 0.005 pp per holding)

## Forecasting the top holdings before each rebalance

    python -m stonks.rdvy

Downloads RDVY's current holdings from First Trust, infers which sub-portfolios each stock is in
(from weights and price moves since each sub-portfolio's last rebuild), and projects weights right
after the next rebalance. Stocks in all three sub-portfolios that are not being rebuilt are assumed
to be re-selected: in 2026 Nasdaq re-picked 91-92% of names picked in the prior rebuilds. The table
shows each candidate's weight both if re-picked and if not.

Run it after the price reference date (last trading day of Feb / May / Aug / Nov) for the final
projection; Nasdaq announces the actual changes six trading days before the effective date.

### Screen replication (stonks/nqdvris.py)
Applies the published screens to Yahoo Finance fundamentals. Checked against the last four rebuilds
it reproduced 121-124 of 196 actual picks (about 63%) and 65-68% of names held in three or more
sub-portfolios. Misses cluster in mid-size financials and names where Yahoo's EPS (adjusted) or
dividend history differs from Nasdaq's data. Use it as a cross-check on the persistence forecast,
not as a replacement.
