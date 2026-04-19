from __future__ import annotations

import numpy as np
import pandas as pd

from p4.kalman_ou import KalmanOU


def _simulate_ou(
    rng: np.random.Generator,
    *,
    n_obs: int,
    kappa: float,
    mu: float,
    sigma: float,
    start: float | None = None,
) -> np.ndarray:
    phi = float(np.exp(-kappa))
    innovation_std = float(sigma * np.sqrt((1.0 - phi**2) / max(2.0 * kappa, 1e-8)))
    values = np.empty(n_obs, dtype=np.float64)
    values[0] = mu if start is None else float(start)
    for idx in range(1, n_obs):
        values[idx] = mu + phi * (values[idx - 1] - mu) + rng.normal(scale=innovation_std)
    return values


def test_kalman_ou_recovers_static_parameters_on_long_stationary_series() -> None:
    rng = np.random.default_rng(seed=42)
    spread = pd.Series(_simulate_ou(rng, n_obs=2000, kappa=0.5, mu=0.0, sigma=0.1))
    result = KalmanOU().fit(spread)
    assert np.isclose(result["kappa"], 0.5, rtol=0.25)
    assert abs(float(result["mu"])) < 0.05


def test_kalman_ou_tracks_regime_change() -> None:
    rng = np.random.default_rng(seed=42)
    first = _simulate_ou(rng, n_obs=1000, kappa=0.5, mu=0.0, sigma=0.1)
    second = _simulate_ou(rng, n_obs=1001, kappa=2.0, mu=0.0, sigma=0.1, start=float(first[-1]))[1:]
    spread = pd.Series(np.concatenate([first, second]))
    result = KalmanOU().fit(spread)
    assert abs(float(result["kappa_path"][1500]) - 2.0) < abs(float(result["kappa_path"][1500]) - 0.5)


def test_kalman_ou_log_likelihood_is_finite() -> None:
    rng = np.random.default_rng(seed=42)
    spread = pd.Series(_simulate_ou(rng, n_obs=400, kappa=0.5, mu=0.0, sigma=0.1))
    result = KalmanOU().fit(spread)
    assert np.isfinite(result["log_likelihood"])


def test_kalman_ou_drop_in_compatible_with_ou_estimator_keys() -> None:
    rng = np.random.default_rng(seed=42)
    spread = pd.Series(_simulate_ou(rng, n_obs=400, kappa=0.5, mu=0.0, sigma=0.1))
    result = KalmanOU().fit(spread)
    assert {"kappa", "mu", "sigma", "half_life"}.issubset(result)
