# P4 — Mixed Equities + ETFs Statistical Arbitrage

P4 is a real-data-first research repo for market-neutral statistical arbitrage across a mixed universe of current S&P 500 equities and a fixed ETF sleeve. The project screens pair and 3-asset basket candidates, filters them through cointegration and OU half-life tests, backtests validation/test windows with conservative trading frictions, and then deflates apparent winners with Bonferroni, White's Reality Check, and Hansen SPA.

The main research question is not "what is the highest Sharpe pair?" It is "how many seemingly attractive candidates remain after rigorous family-wise multiple-testing correction?"

## Verified Status

As of April 10, 2026, the repo is implemented and verified in the shared drive environment at `/Volumes/Crucial X9/alpha_engine/.venv`.

- `make install` succeeds in the shared env.
- `make test` passes with 15 tests and 90% coverage on `src/p4`.
- `make download` succeeds for the reduced real-data smoke config and cached 22 assets under `/Volumes/Crucial X9/data/market_data/p4_smoke/`.
- `make run` succeeds for the reduced real-data smoke config and writes outputs under `results/real_smoke/`.
- The offline fixture integration path exercises both pairs and baskets; the live smoke universe produced only pair candidates.

## Live Smoke Result

Verified with `CONFIG=/Volumes/Crucial X9/projects/p4_stat_arb/configs/p4_smoke.yaml`.

- Universe: 22 assets total, 12 equities and 10 ETFs
- History: 2019-01-02 through 2025-12-30
- Walk-forward windows: 9
- Pair candidates: 4
- Basket candidates: 0
- Raw positive test strategies: 0
- Bonferroni survivors: 0
- SPA survivors: 0
- White RC p-value of the best candidate: `0.995`
- Portfolio rows: 0 because no strategy cleared the validation filter with positive net Sharpe

This is a valid research outcome, not a hidden failure. On the smoke universe and cost assumptions, the apparent stat-arb signals did not survive validation or multiple-testing correction.

## Environment

P4 is standardized on the shared drive environment. Do not create a repo-local `.venv`.

```bash
cd /Volumes/Crucial\ X9/projects/p4_stat_arb
make install
make test
CONFIG=/Volumes/Crucial\ X9/projects/p4_stat_arb/configs/p4_smoke.yaml make download
CONFIG=/Volumes/Crucial\ X9/projects/p4_stat_arb/configs/p4_smoke.yaml make run
```

The default config remains `configs/p4_config.yaml` with `data_mode: real`. The smoke config is the verified manual acceptance path because it finishes quickly enough for iterative work.

## Research Design

- Universe bootstrap: current S&P 500 constituents from Wikipedia plus a fixed ETF list
- Candidate families:
  - equity-equity within sub-industry first, sector fallback
  - ETF-ETF within configured ETF families
  - equity-ETF within sector-matched families
- Basket scope: 3 assets only, expanded from prefiltered pair families rather than exhaustive triplets
- Walk-forward schedule: 504 formation days, 126 validation days, 126 test days, stepped every 126 days in the default config
- Spread definitions:
  - pairs: `log(P1) - beta * log(P2)` with `beta` from formation-period OLS
  - baskets: Johansen rank-1 eigenvector on log prices, normalized to unit gross weight
- OU viability filter: half-life between configured minimum and maximum
- Signal logic: enter at `|z| > 2`, exit at `|z| < 0.5`, stop at `|z| > 4` in the default config
- Cost model: 5 bps half-spread + 3 bps slippage per leg, 50 bps/year short borrow
- Capacity model: active exposure capped at 5% of 30-day ADV per asset, scaled down pro rata when breached
- Multiple testing:
  - Bonferroni on one-sided test-set mean-return p-values
  - White Reality Check on daily net return differentials versus zero
  - Hansen SPA on the same out-of-sample family

## Output Contract

Each run writes to `results/<run_name>/`:

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

The AlphaEngine trace for the latest run is written to `docs/alpha_engine_trace/` and mirrored under `docs/alpha_engine_trace/<run_name>/`.

## Key Limitations

- Real-data v1 uses current S&P 500 constituents, not historical constituent membership. This introduces survivorship bias.
- The ETF taxonomy is deliberately simple and fixed in config.
- The basket path is implemented, but the verified live smoke run did not surface any basket candidates in the reduced universe.
- Public-data quality is sufficient for a research MVP, not for institutional stat-arb claims.

## Repo Map

```text
src/p4/
├── config.py            # typed config loader
├── data_loader.py       # universe bootstrap, cache build, fixture loader
├── cointegration.py     # Engle-Granger and Johansen screens
├── pair_selector.py     # family generation and validation ranking
├── ou_estimator.py      # OU parameter estimation and half-life
├── signal.py            # z-score entry/exit/stop rules
├── backtest.py          # spread backtest, costs, and capacity scaling
├── multiple_testing.py  # Bonferroni, White RC, Hansen SPA
└── pipeline.py          # end-to-end walk-forward orchestrator
```
