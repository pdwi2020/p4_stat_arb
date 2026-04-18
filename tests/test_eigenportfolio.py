from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.linalg import hadamard, orthogonal_procrustes

from p4.config import BacktestConfig
from p4.eigenportfolio import EigenportfolioBacktest, EigenportfolioStrategy, PCAFactorModel
from p4.multiple_testing import bonferroni_threshold, hansen_spa_test, one_sided_mean_pvalues, white_reality_check

from .conftest import write_config


def _factor_panel(*, seed: int = 7, n_obs: int = 900, n_assets: int = 8, n_factors: int = 3) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    full_basis = hadamard(n_assets).astype(float) / np.sqrt(n_assets)
    eigenvalues = np.array([4.5, 3.0, 2.0] + [0.35] * (n_assets - n_factors), dtype=float)
    covariance = full_basis @ np.diag(eigenvalues) @ full_basis.T
    returns = rng.multivariate_normal(mean=np.zeros(n_assets, dtype=float), cov=covariance, size=n_obs)
    columns = [f"Asset{idx}" for idx in range(n_assets)]
    index = pd.bdate_range("2020-01-02", periods=n_obs)
    return pd.DataFrame(returns, index=index, columns=columns), full_basis[:, :n_factors]


def _simulate_ou_process(
    *,
    rng: np.random.Generator,
    n_obs: int,
    kappa: float,
    mu: float,
    sigma: float,
) -> np.ndarray:
    phi = float(np.exp(-kappa))
    innovation_std = float(sigma * np.sqrt((1.0 - np.exp(-2.0 * kappa)) / (2.0 * max(kappa, 1e-8))))
    values = np.empty(n_obs, dtype=float)
    values[0] = mu
    for idx in range(1, n_obs):
        values[idx] = mu + phi * (values[idx - 1] - mu) + innovation_std * rng.normal()
    return values


def _mean_reverting_returns_panel(seed: int = 17) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n_obs = 1000
    n_assets = 5
    market_factor = rng.normal(scale=0.0025, size=n_obs)
    betas = np.linspace(0.8, 1.2, n_assets, dtype=float)

    idiosyncratic_states = []
    for offset in range(n_assets):
        state = _simulate_ou_process(
            rng=rng,
            n_obs=n_obs,
            kappa=0.20 + 0.02 * offset,
            mu=0.0,
            sigma=0.025,
        )
        idiosyncratic_states.append(state)
    idiosyncratic = np.column_stack(idiosyncratic_states)
    idio_returns = np.diff(idiosyncratic, axis=0, prepend=idiosyncratic[[0]])
    returns = market_factor[:, None] * betas[None, :] + 2.0 * idio_returns

    index = pd.bdate_range("2020-01-02", periods=n_obs)
    columns = [f"MR{idx}" for idx in range(n_assets)]
    return pd.DataFrame(returns, index=index, columns=columns), pd.DataFrame(idiosyncratic, index=index, columns=columns)


def _synthetic_backtest_results() -> tuple[EigenportfolioBacktest, object]:
    returns, _ = _mean_reverting_returns_panel()
    model = PCAFactorModel(n_components=1, standardize=True, lookback_days=252)
    strategy = EigenportfolioStrategy(
        pca_model=model,
        min_half_life=2.0,
        max_half_life=12.0,
        z_entry=0.8,
        z_exit=0.15,
        z_stop=4.0,
    )
    backtester = EigenportfolioBacktest(
        pca_model=model,
        strategy=strategy,
        cost_model=BacktestConfig(
            cost_halfspread_bps=0.0,
            cost_slippage_bps=0.0,
            borrow_cost_annual_bps=0.0,
            capacity_adv_fraction=1.0,
            portfolio_capital_usd=1_000_000.0,
        ),
        capacity_model=BacktestConfig(
            cost_halfspread_bps=0.0,
            cost_slippage_bps=0.0,
            borrow_cost_annual_bps=0.0,
            capacity_adv_fraction=1.0,
            portfolio_capital_usd=1_000_000.0,
        ),
    )
    adv = pd.DataFrame(1e12, index=returns.index, columns=returns.columns)
    results = backtester.walk_forward(
        returns=returns,
        formation_window=252,
        validation_window=126,
        test_window=126,
        step_window=126,
        adv_30d=adv,
    )
    return backtester, results


def test_pca_recovers_factors() -> None:
    returns, true_basis = _factor_panel()
    model = PCAFactorModel(n_components=3, standardize=True, lookback_days=len(returns))
    model.fit(returns)

    estimated = model.loadings.to_numpy(dtype=float)
    rotation, _ = orthogonal_procrustes(estimated, true_basis)
    aligned = estimated @ rotation
    error = np.linalg.norm(aligned - true_basis) / np.linalg.norm(true_basis)

    assert model.loadings.shape == (returns.shape[1], 3)
    assert float(model.explained_variance_ratio.sum()) > 0.75
    assert error < 0.2


def test_residuals_orthogonal_to_factors() -> None:
    returns, _ = _factor_panel(seed=11)
    model = PCAFactorModel(n_components=3, standardize=True, lookback_days=len(returns)).fit(returns)

    residuals = model.residuals(returns)
    factors = model.transform(returns)
    covariance = residuals.to_numpy(dtype=float).T @ factors.to_numpy(dtype=float) / len(residuals)

    assert residuals.shape == returns.shape
    assert np.max(np.abs(covariance)) < 1e-10


def test_ou_fit_on_synthetic_mean_reverting() -> None:
    rng = np.random.default_rng(23)
    kappa_true = 0.11
    mu_true = 0.35
    sigma_true = 0.18
    state = _simulate_ou_process(rng=rng, n_obs=600, kappa=kappa_true, mu=mu_true, sigma=sigma_true)
    residual_returns = np.concatenate([[state[0]], np.diff(state)])

    strategy = EigenportfolioStrategy(
        pca_model=PCAFactorModel(n_components=1),
        min_half_life=1.0,
        max_half_life=60.0,
        z_entry=1.5,
        z_exit=0.5,
        z_stop=3.0,
    )
    params = strategy.fit_residual_ou(pd.DataFrame({"AAA": residual_returns}))["AAA"]

    assert params["kappa"] == pytest.approx(kappa_true, rel=0.25)
    assert params["mu"] == pytest.approx(mu_true, rel=0.20, abs=0.05)
    assert params["sigma"] == pytest.approx(sigma_true, rel=0.30)


def test_full_backtest_synthetic() -> None:
    _, results = _synthetic_backtest_results()

    assert not results.candidate_eigenportfolios.empty
    assert not results.daily_strategy_returns.empty
    assert not results.portfolio_returns.empty
    assert results.candidate_eigenportfolios["target_ticker"].nunique() >= 3

    assert float(results.candidate_eigenportfolios["test_net_sharpe"].max()) > 1.0


def test_multiple_testing_integration() -> None:
    _, results = _synthetic_backtest_results()

    return_wide = results.daily_strategy_returns.pivot_table(index="date", columns="candidate_id", values="net_return", fill_value=0.0)
    pvalues = one_sided_mean_pvalues(return_wide)
    bonf_threshold = bonferroni_threshold(0.05, len(pvalues))
    bonf_survivors = pvalues.index[pvalues <= bonf_threshold].tolist()
    white_report = white_reality_check(return_wide, n_bootstrap=80, block_size=10, random_seed=5)
    spa_report = hansen_spa_test(return_wide, n_bootstrap=80, block_size=10, alpha=0.05, random_seed=5)
    expected_bonf = sorted(pvalues.index[pvalues <= bonf_threshold].tolist())
    candidate_ids = set(results.candidate_eigenportfolios["candidate_id"])

    assert (results.candidate_eigenportfolios["test_net_sharpe"] > 0).any()
    assert sorted(bonf_survivors) == expected_bonf
    assert set(spa_report["survivors"]).issubset(candidate_ids)
    assert 0.0 <= white_report["pvalue"] <= 1.0


def test_backward_compat(tmp_path: Path) -> None:
    pytest.importorskip("statsmodels")
    from p4.pipeline import run

    implicit_config_path = write_config(tmp_path)
    explicit_config_path = tmp_path / "p4_test_config_cointegration.yaml"

    payload = yaml.safe_load(implicit_config_path.read_text())
    payload["strategy_type"] = "cointegration"
    explicit_config_path.write_text(yaml.safe_dump(payload, sort_keys=False))

    implicit_summary = run(implicit_config_path)
    explicit_summary = run(explicit_config_path)

    assert implicit_summary == explicit_summary
