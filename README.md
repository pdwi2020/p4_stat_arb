# P4 — Mixed Equities + ETFs Statistical Arbitrage with Multiple-Testing Discipline

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/) [![Tests](https://img.shields.io/badge/tests-60%20passing-brightgreen.svg)]() [![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> Market-neutral pair + 3-asset basket arbitrage on a real current S&P 500 + ETF universe, with sector-aware screening, OU half-life filters, and **Bonferroni / BH / BY / Storey / Hansen SPA / White Reality Check** deflation of apparent winners.

## TL;DR

The research question is not "what is the highest Sharpe pair?" It is **"how many seemingly attractive candidates remain after rigorous family-wise multiple-testing correction?"**

Headline ablation (median net Sharpe across 16 validated large-cap proxy pairs, 4-way OU comparison):

| Method | Median net Sharpe |
| --- | ---: |
| regime-switch OU | **1.07** |
| static OU | 0.95 |
| Kalman-OU | 0.75 |
| neural OU | 0.36 |

n=16 is too small for a strong method-choice claim under Hansen SPA, and the memo says so explicitly.

## Background

Statistical arbitrage on equities is well-trodden ground; the failure mode is data-mined Sharpe estimates that do not survive multiple-testing correction. This repo is built around the discipline that the validation step **kills more pairs than it accepts** and that this is the correct outcome.

The live pipeline works on a mixed universe of current S&P 500 equities plus liquid ETFs. The Week 7 ablation applies the same sector-aware selection logic to a practical large-cap proxy universe, then compares four OU estimators on the same 16 validated pairs.

Pipeline: candidate generation (sector-aware screening) -> cointegration tests (Engle-Granger, Johansen) -> OU half-life filters -> conservative-friction backtest -> multiple-testing deflation (Bonferroni + BH + BY + Storey + Hansen SPA + White Reality Check).

## What's in the repo

- [`src/p4/pair_selector.py`](src/p4/pair_selector.py) — sector-aware candidate pair generator
- [`src/p4/multiple_testing.py`](src/p4/multiple_testing.py) — Bonferroni + BH + BY + Storey + Hansen SPA + White Reality Check
- [`src/p4/johansen.py`](src/p4/johansen.py) — Johansen rank test (Week 5)
- [`src/p4/kalman_ou.py`](src/p4/kalman_ou.py) — Kalman-filter OU with time-varying theta, mu, sigma (Week 5)
- [`src/p4/neural_ou.py`](src/p4/neural_ou.py) — neural-OU baseline
- [`src/p4/regime_switch.py`](src/p4/regime_switch.py) — regime-switching OU
- [`src/p4/sp500_universe.py`](src/p4/sp500_universe.py) — S&P 500-proxy large-cap universe builder (Week 6)
- [`src/p4/run_sp500_ablation.py`](src/p4/run_sp500_ablation.py) — Week 7 4-way OU ablation runner
- [`Makefile`](Makefile) — reproducible entry points for `download`, `run`, `run-extended`, and `test`
- [`configs/`](configs/) — typed YAML
- [`tests/`](tests/) — 60 test functions across pipeline, estimators, and multiple-testing logic
- [`memo.md`](memo.md) — 450-line research memo (Johansen + Kalman-OU methodology, 4-way ablation, honest caveats)

## How to reproduce

```bash
git clone https://github.com/pdwi2020/p4_stat_arb.git
cd p4_stat_arb
python3.14 -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
make download         # build/cache the mixed S&P 500 + ETF universe and history
make backtest         # default pipeline (alias of make run)
make run-extended     # regime-aware Wave-2 pipeline
make test             # 60 test functions
```

## Methodology highlights

- **Cointegration**: Engle-Granger (pairs) + Johansen rank test (3-asset baskets)
- **OU half-life filtering**: reject pairs whose mean-reverting half-life is too long for the holding period
- **Static OU vs Kalman-OU vs neural-OU vs regime-switch OU**: identical pair universe, identical spread definition, identical signal rule within the Week 7 ablation
- **Multiple-testing**: Bonferroni (FWER), Benjamini-Hochberg (FDR), Benjamini-Yekutieli (dependent FDR), Storey q-value, Hansen SPA, White Reality Check
- **Pair capacity** estimation per surviving pair
- **Conservative trading frictions**: spread + slippage + borrow embedded in the backtest, not bolted on afterward

## Honest caveats

- **n=16 validated pairs is small**: the 4-way OU ablation reports a ranking, but it is not a strong method-choice claim under Hansen SPA. The memo says this in plain language.
- **Survivorship bias is real**: the live pipeline uses the current S&P 500 membership, while the Week 7 ablation uses a current top-500 large-cap proxy. Neither is a point-in-time historical constituent file.
- **Sub-industry quartile sector taxonomy is synthetic**: the proxy scan substitutes sector + within-sector weight quartiles for true GICS sub-industry labels. The memo notes this trade-off explicitly.
- **Corporate-action edge cases are not modeled explicitly**: beyond adjusted public price histories, events such as spinoffs, mergers, and ticker-history changes are out of scope for this research MVP.

## References

- Engle, R. F. & Granger, C. W. J. (1987). Co-integration and error correction. *Econometrica* 55(2).
- Johansen, S. (1991). Estimation and hypothesis testing of cointegration vectors in Gaussian VAR models. *Econometrica* 59(6).
- Avellaneda, M. & Lee, J.-H. (2010). Statistical arbitrage in the U.S. equities market. *Quantitative Finance* 10(7).
- White, H. (2000). A reality check for data snooping. *Econometrica* 68(5).
- Hansen, P. R. (2005). A test for superior predictive ability. *J. Business & Economic Statistics* 23(4).
- Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate. *J. Royal Statistical Society B* 57(1).
- Benjamini, Y. & Yekutieli, D. (2001). The control of the false discovery rate under dependency. *Annals of Statistics* 29(4).
- Storey, J. D. (2002). A direct approach to false discovery rates. *J. Royal Statistical Society B* 64(3).

## Project context

This repo is project **P4** in a 5-project quant research portfolio prepared for buy-side QR internship applications (Summer 2027). The other four:

- [`p1_factor_research`](https://github.com/pdwi2020/p1_factor_research) — Cross-sectional equity factor research on Russell-3000 with BARRA + Almgren-Chriss capacity
- [`p2_market_maker`](https://github.com/pdwi2020/p2_market_maker) — Avellaneda-Stoikov HFT market maker with HJB derivation and queue-reactive Glosten-Milgrom layer (flagship)
- [`p3_vol_surface`](https://github.com/pdwi2020/p3_vol_surface) — Volatility surface dynamics: SVI + SSVI + rBergomi calibration, HAR-RV vs GARCH(1,1) horserace
- [`p5_gpu_mc_exotics`](https://github.com/pdwi2020/p5_gpu_mc_exotics) — GPU-accelerated Monte Carlo for exotic options (Heston, Bates, HHW) on CUDA T4 (flagship)

## License

MIT. See [`LICENSE`](LICENSE).
