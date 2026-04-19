from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from p4.johansen import johansen_basket_weights, johansen_test, vecm_fit


def _ar1(rng: np.random.Generator, *, phi: float, sigma: float, n_obs: int) -> np.ndarray:
    values = np.zeros(n_obs, dtype=np.float64)
    shocks = rng.normal(scale=sigma, size=n_obs)
    for idx in range(1, n_obs):
        values[idx] = phi * values[idx - 1] + shocks[idx]
    return values


def _cointegrated_pair_panel(n_obs: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(seed=42)
    common = np.cumsum(rng.normal(scale=0.03, size=n_obs)).astype(np.float64)
    noise = _ar1(rng, phi=0.45, sigma=0.025, n_obs=n_obs)
    return pd.DataFrame({"X": common, "Y": common + noise}, dtype="float64")


def test_johansen_test_finds_rank_one_for_synthetic_pair() -> None:
    log_prices = _cointegrated_pair_panel()
    result = johansen_test(log_prices)
    basket = johansen_basket_weights(log_prices)
    assert result["rank_estimate_trace"] == 1
    assert basket["is_cointegrated"] is True


def test_johansen_test_finds_rank_zero_for_two_random_walks() -> None:
    rng = np.random.default_rng(seed=42)
    log_prices = pd.DataFrame(
        {
            "X": np.cumsum(rng.normal(scale=0.04, size=300)),
            "Y": np.cumsum(rng.normal(scale=0.04, size=300)),
        },
        dtype="float64",
    )
    result = johansen_test(log_prices)
    assert result["rank_estimate_trace"] == 0


def test_johansen_basket_weights_normalize_to_unit_l1() -> None:
    weights = johansen_basket_weights(_cointegrated_pair_panel())["weights"]
    assert np.isclose(np.abs(weights).sum(), 1.0, atol=1e-9)


def test_johansen_basket_spread_is_stationary() -> None:
    spread = johansen_basket_weights(_cointegrated_pair_panel())["spread"]
    _, pvalue, *_ = adfuller(pd.Series(spread, dtype="float64"))
    assert float(pvalue) < 0.05


def test_vecm_fit_returns_loadings_and_beta() -> None:
    result = vecm_fit(_cointegrated_pair_panel())
    assert result["alpha_loadings"].shape == (2, 1)
    assert result["beta_cointegration"].shape == (2, 1)
