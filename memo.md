# P4 Research Memo

## Objective

P4 studies whether a mixed universe of current S&P 500 equities and liquid ETFs can produce credible market-neutral statistical arbitrage signals once the usual selection-bias traps are handled explicitly. The project is designed around a simple claim: it is easy to find raw cointegrated spreads, but much harder to find spreads that survive validation, trading costs, capacity limits, and family-wise multiple-testing correction.

The repo therefore does not optimize for the single best backtest. It optimizes for a defensible research process:

- generate candidates from a structured mixed universe
- estimate spreads with standard cointegration tools
- reject unstable mean reversion with an OU half-life filter
- validate out of sample before entering the test set
- apply Bonferroni, White's Reality Check, and Hansen SPA on the full tested strategy family

## Data and Universe

The v1 live pipeline uses free/public daily data only. Current S&P 500 constituents are pulled from Wikipedia, ETF metadata is fixed in config, and price/volume history comes from `yfinance`. The verified smoke run used the reduced config in `configs/p4_smoke.yaml` and cached the resulting public data under `/Volumes/Crucial X9/data/market_data/p4_smoke/`.

The verified smoke universe contained:

- 12 equities
- 10 ETFs
- 22 assets total
- daily history from 2019-01-02 through 2025-12-30

This is intentionally a research MVP rather than an institutional production dataset. The major limitation is survivorship bias: the equity universe uses the current S&P 500 membership, not historical constituent membership. That limitation is material and should remain explicit in any external discussion of the results.

## Methodology

### Candidate generation

Candidate generation is controlled to avoid brute-force combinatorics:

- equity-equity pairs are generated within the same sub-industry first, with sector fallback
- ETF-ETF pairs are generated only within configured ETF family groups
- equity-ETF pairs are limited to sector-matched relationships
- 3-asset baskets are expanded from prefiltered pair families rather than from all possible triplets

For pairs, the formation-period spread is `log(P1) - beta * log(P2)`, where `beta` is estimated by OLS. For 3-asset baskets, the spread is the Johansen rank-1 eigenvector applied to log prices and normalized to unit gross absolute weight.

### Walk-forward structure

The default design uses:

- 504 trading days of formation
- 126 trading days of validation
- 126 trading days of test
- 126 trading day step size

The verified smoke run used a shorter but structurally identical walk-forward design:

- 378 formation days
- 126 validation days
- 126 test days
- 126 step size
- 9 windows in total

### OU filter and signal

Each candidate spread is fit to a discrete-time OU approximation using regression on lagged spread levels. The pipeline estimates:

- `kappa`
- `mu`
- `sigma`
- half-life

Candidates are retained only if the estimated half-life falls in the configured viability band. The smoke run used a half-life band of 4 to 35 trading days.

Signals are built from OU-centered z-scores:

- entry when `|z| > 1.8` in the smoke config
- exit when `|z| < 0.5`
- stop when `|z| > 4`

The position is a simple mean-reversion overlay on the spread weights. This is intentionally conservative and interpretable, not an optimized execution model.

### Cost and capacity assumptions

The backtest charges:

- 5 bps half-spread
- 3 bps slippage
- 50 bps/year short borrow on short notional

Capacity is limited through a 5% of 30-day ADV cap per asset. If multiple active strategies exceed aggregate capacity on a date, the portfolio is scaled down pro rata. This matters because otherwise a mixed-universe stat-arb backtest can report unrealistic scalability on thin overlaps across related assets.

### Multiple-testing deflation

The multiple-testing layer is the center of the project:

- Bonferroni on one-sided out-of-sample mean-return p-values
- White's Reality Check on the best test-set strategy versus a zero benchmark
- Hansen SPA on the same out-of-sample family

Bootstrap block length is derived from `2 x half_life`, clipped into a bounded practical range. This ties the dependence structure to the estimated mean-reversion timescale rather than using an arbitrary fixed block.

## Empirical Result: Real Smoke Run

The verified smoke run produced the following top-level output in `results/real_smoke/summary.json`:

- assets: 22
- windows: 9
- pair candidates: 4
- basket candidates: 0
- raw positive test strategies: 0
- Bonferroni survivors: 0
- SPA survivors: 0
- White RC p-value: `0.995`
- portfolio net Sharpe: `0.0`
- portfolio total return: `0.0`

The tested strategy family was small but still illustrative. The four discovered pair candidates were:

- `pair_w00_SMH_XLK`
- `pair_w01_GOOGL_META`
- `pair_w03_QQQ_SPY`
- `pair_w05_SMH_XLK`

All four had negative test-set Sharpe ratios after costs. The least bad result was `pair_w01_GOOGL_META` with test Sharpe `-0.235`. The worst was `pair_w03_QQQ_SPY` with test Sharpe `-8.359`. Since every validation net Sharpe was also non-positive, no strategy entered the portfolio layer, which is why `portfolio_returns.csv` is structurally present but empty.

The multiple-testing outputs reinforce the same point:

- Bonferroni threshold: `0.0125`
- Bonferroni survivors: none
- White RC best strategy: `pair_w01_GOOGL_META`
- White RC p-value: `0.995`
- SPA survivors: none

This is a useful result because it demonstrates the exact research story the repo was built to tell. Even after a relatively constrained search, the candidate family fails validation and then fails multiple-testing correction decisively.

## Interpretation

The negative smoke result should not be framed as a broken implementation. It is better interpreted as evidence that:

1. A disciplined stat-arb pipeline should be willing to return "nothing survives."
2. Costs and validation eliminate many spreads before multiple-testing even matters.
3. The final multiple-testing layer can still show that the best apparent candidate is statistically unconvincing.

There is also a structural lesson in the zero-basket outcome. The basket engine is implemented and exercised successfully in the fixture integration path, but the reduced live smoke universe did not generate any baskets that passed the full chain of screening conditions. That is plausible for a small, hand-curated live universe with strict relation-family rules.

## What Failed or Remains Weak

- The public-data live run is survivorship-biased because the equity universe is based on current S&P 500 membership.
- The verified smoke universe is intentionally small; it is a good acceptance test, not a full-scale research statement.
- The live smoke run produced no validation-positive candidates, so the portfolio layer is demonstrated operationally but not economically.
- Basket support is implemented, but the live smoke run did not surface any basket candidates.
- `pip` in the shared drive environment still emits pre-existing invalid-distribution warnings unrelated to P4 itself.

## Why the Repo Is Still Useful

Despite the weak live result, the repo is useful because it now contains the full research machinery needed for a more serious stat-arb pass:

- typed config and deterministic output paths
- public-data cache bootstrap
- pair and basket candidate generation
- OU screening
- cost-aware validation/test backtesting
- capacity-aware portfolio aggregation
- multiple-testing deflation
- fixture-backed offline integration coverage

That means future work can focus on better universe construction, better point-in-time membership handling, and better candidate families rather than on missing infrastructure.

## Next Steps

The most valuable upgrades are:

- replace current-constituent S&P 500 membership with historical point-in-time membership
- expand the live ETF and sector-mapping design only while preserving full tested-family accounting
- test richer bootstrap and dependence assumptions in the White RC and SPA layer
- run the full default real config after the smoke workflow is stable enough for longer research runs

## References

- Gatev, Goetzmann, and Rouwenhorst. "Pairs Trading: Performance of a Relative-Value Arbitrage Rule."
- White. "A Reality Check for Data Snooping."
- Hansen. "A Test for Superior Predictive Ability."
- Avellaneda and Lee. "Statistical Arbitrage in the U.S. Equities Market."
