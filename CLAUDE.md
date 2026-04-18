# P4 — Mixed Equities + ETFs Statistical Arbitrage

## Status

Implemented and verified in the shared environment at `/Volumes/Crucial X9/alpha_engine/.venv`.

- `make install` uses the shared drive env only
- `make test` passes with 15 tests and 90% coverage
- `make download` works on the reduced real-data smoke config
- `make run` works on the reduced real-data smoke config

Latest verified live smoke output:

- run name: `real_smoke`
- assets: 22 total, 12 equities and 10 ETFs
- windows: 9
- pair candidates: 4
- basket candidates: 0
- Bonferroni survivors: 0
- SPA survivors: 0
- White RC p-value: `0.995`

## Root and Environment

- Root: `/Volumes/Crucial X9/projects/p4_stat_arb/`
- Shared env: `/Volumes/Crucial X9/alpha_engine/.venv`
- Default config: `/Volumes/Crucial X9/projects/p4_stat_arb/configs/p4_config.yaml`
- Verified smoke config: `/Volumes/Crucial X9/projects/p4_stat_arb/configs/p4_smoke.yaml`

Do not create or assume a local `.venv`.

## Command Surface

```bash
cd /Volumes/Crucial\ X9/projects/p4_stat_arb
make install
make test
CONFIG=/Volumes/Crucial\ X9/projects/p4_stat_arb/configs/p4_smoke.yaml make download
CONFIG=/Volumes/Crucial\ X9/projects/p4_stat_arb/configs/p4_smoke.yaml make run
```

## Project Intent

P4 studies cointegration-based stat-arb across a mixed universe:

- current S&P 500 constituents from a public table
- a fixed ETF sleeve defined in config
- pair and 3-asset basket candidates
- validation-aware ranking before test deployment
- explicit multiple-testing deflation as the center of the research story

The main claim should remain conservative: most raw candidates disappear after costs, validation, and multiple-testing correction.

## Module Map

```text
src/p4/
├── config.py            — typed config and path resolution
├── data_loader.py       — public-universe bootstrap, cache build, fixture loader
├── cointegration.py     — Engle-Granger and Johansen screens
├── pair_selector.py     — family generation, validation ranking, basket expansion
├── ou_estimator.py      — κ, μ, σ, half-life estimation from spread history
├── signal.py            — z-score entry, exit, stop logic
├── backtest.py          — spread backtest, costs, borrow, ADV capacity scaling
├── multiple_testing.py  — Bonferroni, White RC, Hansen SPA
└── pipeline.py          — full walk-forward pipeline and artifact writer
```

## Research Defaults

- Pair spread: `log(P1) - beta * log(P2)`
- Basket spread: Johansen rank-1 eigenvector on log prices
- Default OU viability band: 5 to 30 trading days half-life
- Default signal: entry `|z| > 2`, exit `|z| < 0.5`, stop `|z| > 4`
- Costs: 5 bps half-spread + 3 bps slippage per leg, 50 bps/year short borrow
- Capacity: 5% of 30-day ADV per asset
- Multiple testing: Bonferroni, White RC, Hansen SPA

## Output Contract

Every run writes `results/<run_name>/` with:

- `candidate_pairs.csv`
- `candidate_baskets.csv`
- `strategy_metrics.csv`
- `daily_strategy_returns.csv`
- `portfolio_returns.csv`
- `multiple_testing_report.json`
- `summary.json`
- `portfolio_cumulative_return.png`
- `portfolio_drawdown.png`
- `candidate_deflation.png`

AlphaEngine trace artifacts are mirrored to:

- `docs/alpha_engine_trace/blueprint.json`
- `docs/alpha_engine_trace/ambiguity_log.md`
- `docs/alpha_engine_trace/<run_name>/...`

## Known Limitations

- Current-constituent S&P 500 membership introduces survivorship bias.
- ETF grouping is explicit and fixed, not learned.
- The verified live smoke run did not find any basket candidates; basket support is still implemented and exercised in fixture integration tests.
- Public-data quality is acceptable for an interview-grade research MVP, not for production deployment claims.
