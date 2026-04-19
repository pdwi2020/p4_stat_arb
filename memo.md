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

## Methodological Additions: Johansen and Kalman-OU
Week 5 added two pieces of methodology that matter once the
project moves beyond the smallest pair screen.
The first is a fuller Johansen cointegration module for
multi-asset panels.
The second is a Kalman-filter OU estimator that relaxes the
assumption that one fixed `(kappa, mu)` pair governs the entire
formation sample.

The original memo already noted that 3-asset baskets use a
Johansen rank-1 eigenvector.
Week 5 makes that machinery explicit and more general through
`src/p4/johansen.py`.
That matters because pairwise Engle-Granger remains the right
first screen for two assets, but it does not answer the main
question for `N > 2` assets.
For a basket, the issue is not just whether one chosen spread
looks stationary.
It is how many independent long-run equilibrium relations exist
inside the panel.
Johansen's likelihood-based rank test
(Johansen 1988, 1991; Lutkepohl 2005) addresses that directly.
If the cointegration rank is `r`, then there are `r`
independent linear combinations of the `N` log-price series that
pull the system back toward its long-run relationship.
Rank `0` means no cointegration.
Rank `1` gives one tradable stationary direction.
Higher rank means there is more than one stationary direction in
the same panel.

The implementation exposes both standard likelihood-ratio tests.
The trace statistic tests the null `rank <= r` with
`-T * sum_{i=r+1}^N log(1 - lambda_i)`.
The max-eigenvalue statistic tests `rank = r` against
`rank = r + 1` with `-T * log(1 - lambda_{r+1})`.
`johansen_test()` returns both statistic vectors, their critical
values, and rank estimates under each criterion.
`johansen_basket_weights()` then takes the leading
cointegrating eigenvector, normalizes it to unit gross absolute
weight, and emits a basket spread that the rest of the pipeline
can treat like any other spread.
`vecm_fit()` adds the corresponding VECM alpha/beta
decomposition when the adjustment-loadings view is more useful
than a single tradable weight vector.
This is the right extension if the repo is asked to handle
sector baskets, triplets, or larger same-theme panels rather
than only plain pairs.

The other Week 5 addition addresses a different weakness.
The baseline `OUEstimator` is an OLS fit of one discrete-time OU
model over the full formation window.
That is acceptable if the spread is genuinely stationary with
stable parameters.
It is not acceptable when the spread drifts across regimes.
Under drift, OLS compresses the sample into one average
`kappa` and one average `mu`.
That average can be biased toward the center of the window and
therefore too slow or too fast for the most recent regime.

`KalmanOU`, implemented in `src/p4/kalman_ou.py`, follows the
state-space treatment in Harvey (1989) and Durbin and Koopman
(2012).
Conceptually it models the latent OU parameter vector
`theta_t = (kappa_t, mu_t)` as a random walk and updates that
state as each new spread observation arrives.
The code carries the equivalent intercept/slope
representation internally and converts it back to the familiar
OU quantities, so the user-facing result is still an OU fit,
just not a time-invariant one.
The filter therefore recovers a time-varying path rather than
one averaged estimate.

The practical value is that this was built as a drop-in
replacement rather than as a separate branch of the pipeline.
`KalmanOU.fit(spread)` returns the same core keys as
`OUEstimator`: `kappa`, `mu`, `sigma`, and `half_life`.
It also returns `kappa_path` and `mu_path` for inspection, but
downstream code does not need to know that.
The existing selection, validation, and backtest layers can
therefore accept the Kalman fit without rewiring their
interfaces.
The unit tests in `tests/test_kalman_ou.py` cover both the
stable and drifting cases.
Most importantly, the regime-change test simulates a jump from
`kappa = 0.5` to `kappa = 2.0` and checks the filtered path
roughly 500 steps into the second regime.
By that point the path is closer to `2.0` than to `0.5`,
whereas a single OLS fit over the same 2,000-point sample would
have averaged the two regimes to something near `1.25`.
That is exactly the use case for carrying Kalman-OU as a
drop-in estimator in the repo rather than only as a research
side branch.

## S&P 500 Universe Scan: 4-Way OU Ablation
Week 6 uses the new estimators on a larger and less toy-like
cross section.
The goal was not to overwrite the smoke-universe result.
The goal was to exercise the Week 5 machinery on a broader
equity panel and make the four OU estimators directly
comparable on the same candidate set.
That work lives in `src/p4/sp500_universe.py` and
`src/p4/run_sp500_ablation.py`, with outputs under
`results/sp500_ablation/`.

The universe is a practical proxy, not a literal point-in-time
S&P 500 membership file.
The loader takes the top 500 names by `weight_pct` from the
Russell-3000 IWV-style holdings file and treats that as a
current large-cap universe.
`results/sp500_ablation/pair_universe.csv` therefore contains
500 tickers spread across sectors, but it should be described
plainly as a proxy.
The source file does not include real GICS sub-industry labels,
so the loader synthesizes `sub_industry` from within-sector
weight quartiles via `f"{sector}_q{q}"`, with `q` in
`{1, 2, 3, 4}`.
This is not real GICS sub-industry data.
It is a defensible "similar-size names within a sector" bucket
that gives the existing `PairSelector` a meaningful grouping
signal without pretending to know more taxonomy than the source
actually supplies.
The same paragraph needs the survivorship caveat made explicit:
this is today's top-500 large-cap proxy, not point-in-time
membership.

Within that proxy universe, the sector-aware grouping logic
produces 3,390 screenable candidate pairs.
`PairSelector` still does what the core repo does elsewhere:
rank by within-group return correlation, then run the pair
through Engle-Granger and the OU half-life band on the
formation spread.
The Week 6 output also records a Johansen trace statistic for
each surviving two-asset panel, so the result table carries a
multivariate diagnostic alongside the pairwise screen.
In practical terms, the cointegration-and-OU filter leaves
16 surviving pairs in `results/sp500_ablation/pair_candidates.csv`.
That is still a small family, but unlike the smoke run it is
large enough to perform a controlled method comparison on a
nontrivial set of spreads.

The ablation itself is simple by design.
For each of the 16 surviving pairs, Week 6 computes one common
in-sample spread and then fits four OU variants to that same
series:

- static OU via `OUEstimator`
- Kalman OU via `KalmanOU`
- neural OU via `p4.neural_ou`
- regime-switch OU via `p4.regime_switch`

Each fitted model then produces the same trading signal rule:
`sign(mu - spread)`.
That choice is intentionally plain.
It removes signal-design variation from the comparison, so the
ablation is about the parameter estimators rather than about
different execution overlays.
For every pair-method combination the script records gross
Sharpe, net Sharpe, turnover, `kappa`, and half-life in
`results/sp500_ablation/ou_ablation.csv`.
Net Sharpe is computed as gross Sharpe minus
`0.0005 * turnover * 252`, which is a reduced 5 bps turnover
penalty rather than the full smoke-run cost stack.
The resulting file has 64 rows, exactly `16 pairs * 4 methods`.

At the aggregate level the `summary.json` values are:

- static: median `kappa = 0.0248`, median net Sharpe `0.949`,
  mean net Sharpe `1.055`, `n_pairs = 16`
- kalman: median `kappa = 0.113`, median net Sharpe `0.749`,
  mean net Sharpe `0.941`, `n_pairs = 16`
- neural: median `kappa = 0.042`, median net Sharpe `0.357`,
  mean net Sharpe `0.419`, `n_pairs = 16`
- regime: median `kappa = 0.049`, median net Sharpe `1.068`,
  mean net Sharpe `0.975`, `n_pairs = 16`

Rendered in the memo's shorter format, the result table is:

| Method  | Median kappa | Median net Sharpe | Mean net Sharpe | n_pairs |
| ---     | ---:         | ---:              | ---:            | ---:    |
| static  | 0.025        | 0.95              | 1.06            | 16      |
| kalman  | 0.113        | 0.75              | 0.94            | 16      |
| neural  | 0.042        | 0.36              | 0.42            | 16      |
| regime  | 0.049        | **1.07**          | 0.98            | 16      |

The main read on that table is not that one estimator has been
proved superior. The sample is too small for that.
What the table does show is that the four estimators are now
comparable under a common interface and a common signal rule,
which is the main methodological step the repo lacked before Week
5 and Week 6.
The interpretation is still worth stating directly:

- Regime-switch OU has the best median net Sharpe at `1.068`,
  narrowly ahead of static OU at `0.949`.
- That edge is suggestive, not dispositive.
  With only `n = 16` pairs, the confidence intervals overlap
  heavily, and no Hansen-SPA-style multiple-testing correction has
  yet been applied to the method choice itself.
- Kalman-OU posts the highest median `kappa` at `0.113`.
  The filter is pulling noisy OLS fits toward a tighter and more
  reactive mean-reversion speed, but the drop in median net
  Sharpe from `0.949` to `0.749` suggests that on these roughly
  70 percent in-sample windows it may be over-correcting rather
  than helping.
- Neural OU underperforms badly, with median net Sharpe `0.357`.
  The likely explanation is not that neural OU is impossible in
  principle.
  The likely explanation is that a PyTorch model fit on roughly
  750 in-sample observations per pair is undertrained and
  over-parameterized for this use.
  That makes it a debugging target, not a verdict on the whole
  class of models.

The survivor list also lands in the sectors one would expect a
sector-aware mean-reversion screen to surface.
Representative pairs include `AEP/DUK` in utilities, `DD/PPG` in
materials, `TFC/USB` in banks, and `EGP/ESS` in REITs.
Because each surviving pair is evaluated by four estimators,
those representative names appear four times each in the
64-row ablation table.
At the pair level the list is broader than those four examples,
but the tilt is still toward classic mean-reversion neighborhoods:
utilities, materials, banks and other financials, and REITs.
That is a useful sanity check.
It suggests the sector-aware grouping logic is not producing
random cross-sector noise; it is surfacing the parts of the
equity universe where relative-value behavior is at least
plausible.

## Honest Caveats on the SP500 Run
The SP500 ablation should be read as a feasibility test, not as
a tournament bracket with a winner.
Sixteen pairs is a small `N`.
No serious claim that "method X beats method Y" should be made
from this table alone, and certainly not a claim that would
survive a Hansen SPA.
The output is useful because it shows that the four pipelines can
be fit, scored, and compared on the same pair universe with the same
signal rule.
That is a methodological milestone.
It is not yet a publishable inference about which estimator is
best.

The survivorship bias is also doubled here.
First, the universe is the current top-500 by weight rather than
point-in-time index membership.
Second, the run uses only the recent two-year price window that
was available for the panel exercise.
A real research claim would need point-in-time membership,
delisting handling, and something closer to a decade of history so
that the method comparison is not dominated by one recent market
regime.

The cost treatment is intentionally simplified.
Week 6 nets only `5 bps * turnover * 252` from gross Sharpe.
There is no explicit borrow model, no slippage model, and no
market-impact term.
That is acceptable for a quick ablation because the objective is
to compare estimators under one uniform penalty.
It is not the cost model that should be cited externally.
For any outward-facing claim, the right cost stack is still the
earlier smoke-run one: 5 bps half-spread, 3 bps slippage, and
50 bps/year borrow on short notional.

The durable finding worth keeping from Week 6 is therefore
methodological rather than promotional.
P4 now has one unified OU ablation interface across four
estimators, and that interface can be run on any pair universe the
repo can generate.
The SP500 proxy run is simply the first production exercise of
that interface.

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
