from __future__ import annotations

import numpy as np
import pandas as pd

from p4.cointegration import engle_granger_test, johansen_test


def _ar1(rng: np.random.Generator, phi: float, sigma: float, n_obs: int) -> np.ndarray:
    values = np.zeros(n_obs, dtype=float)
    shocks = rng.normal(scale=sigma, size=n_obs)
    for idx in range(1, n_obs):
        values[idx] = phi * values[idx - 1] + shocks[idx]
    return values


def test_engle_granger_detects_cointegrated_pair() -> None:
    rng = np.random.default_rng(10)
    n_obs = 240
    common = np.cumsum(rng.normal(scale=0.02, size=n_obs))
    spread = _ar1(rng, phi=float(np.exp(-1.0 / 8.0)), sigma=0.03, n_obs=n_obs)
    log_prices = pd.DataFrame(
        {
            "A": common + 0.5 * spread,
            "B": common - 0.5 * spread,
        }
    )

    result = engle_granger_test(log_prices, alpha=0.10)
    assert result["pass"] is True
    assert 0.5 < float(result["beta"]) < 1.5
    assert len(result["spread"]) == n_obs
    assert np.isclose(np.abs(result["weights"]).sum(), 1.0)


def test_johansen_detects_cointegrated_basket() -> None:
    rng = np.random.default_rng(11)
    n_obs = 240
    common = np.cumsum(rng.normal(scale=0.02, size=n_obs))
    pair_spread = _ar1(rng, phi=float(np.exp(-1.0 / 9.0)), sigma=0.03, n_obs=n_obs)
    basket_spread = _ar1(rng, phi=float(np.exp(-1.0 / 12.0)), sigma=0.025, n_obs=n_obs)
    log_prices = pd.DataFrame(
        {
            "A": common + 0.5 * pair_spread + 0.35 * basket_spread,
            "B": common - 0.5 * pair_spread + 0.35 * basket_spread,
            "ETF": common - 0.70 * basket_spread,
        }
    )

    result = johansen_test(log_prices, alpha=0.10)
    assert result["pass"] is True
    assert len(result["spread"]) == n_obs
    assert len(result["weights"]) == 3
    assert np.isclose(np.abs(result["weights"]).sum(), 1.0)
