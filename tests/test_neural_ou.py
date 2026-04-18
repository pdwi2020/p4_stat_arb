from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from p4.neural_ou import (
    discover_spreads_via_clustering,
    fit_neural_ou,
    train_autoencoder,
)


def _factor_returns(*, length: int = 160, n_assets: int = 6, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    factor_1 = rng.normal(scale=0.012, size=(length, 1))
    factor_2 = rng.normal(scale=0.006, size=(length, 1))
    loadings_1 = np.linspace(0.8, 1.2, n_assets, dtype=float)
    loadings_2 = np.array([1.0, 0.9, -0.8, -0.7, 0.4, 0.5], dtype=float)[:n_assets]
    noise = rng.normal(scale=0.002, size=(length, n_assets))
    return factor_1 * loadings_1 + factor_2 * loadings_2 + noise


def _simulate_ou(*, theta: float = 0.18, mu: float = 0.4, sigma: float = 0.18, length: int = 240, seed: int = 23) -> pd.Series:
    rng = np.random.default_rng(seed)
    phi = np.exp(-theta)
    innovation_std = sigma * np.sqrt((1.0 - np.exp(-2.0 * theta)) / (2.0 * theta))
    values = np.zeros(length, dtype=float)
    values[0] = mu
    for t in range(1, length):
        values[t] = mu + phi * (values[t - 1] - mu) + rng.normal(scale=innovation_std)
    return pd.Series(values, index=pd.bdate_range("2024-01-02", periods=length), dtype=float)


def test_autoencoder_reconstructs_input_with_reasonable_error() -> None:
    returns = _factor_returns()
    model, latent = train_autoencoder(returns, latent_dim=2, epochs=120, lr=1e-3)
    reconstructed = model.reconstruct_asset_matrix(returns.T)
    mse = float(np.mean((reconstructed - returns.T) ** 2))

    assert latent.shape == (returns.shape[1], 2)
    assert mse < 2e-5


def test_latent_representation_has_expected_dimension() -> None:
    returns = _factor_returns(length=120, n_assets=5)
    _, latent = train_autoencoder(returns, latent_dim=2, epochs=80)

    assert latent.shape == (5, 2)


def test_clustering_discovers_expected_pairs() -> None:
    latent_coords = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [5.0, 5.0],
            [5.1, 4.9],
            [-4.0, 3.0],
            [-3.9, 3.1],
        ],
        dtype=float,
    )
    pairs = discover_spreads_via_clustering(latent_coords, n_clusters=3)

    assert set(pairs) == {(0, 1), (2, 3), (4, 5)}


def test_neural_ou_recovers_theta_on_synthetic_data() -> None:
    spread = _simulate_ou(theta=0.16, mu=0.5, sigma=0.15)
    estimate = fit_neural_ou(spread, epochs=50)

    assert estimate["fitted_via"] == "neural_ou"
    assert abs(float(estimate["theta"]) - 0.16) / 0.16 <= 0.5


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_neural_ou_runs_on_mps_when_available() -> None:
    spread = _simulate_ou(length=180)
    estimate = fit_neural_ou(spread, epochs=20)

    assert estimate["device"] == "mps"


def test_autoencoder_is_deterministic_with_fixed_seed() -> None:
    returns = _factor_returns(seed=9)
    torch.manual_seed(123)
    np.random.seed(123)
    _, latent_a = train_autoencoder(returns, latent_dim=2, epochs=60)
    torch.manual_seed(123)
    np.random.seed(123)
    _, latent_b = train_autoencoder(returns, latent_dim=2, epochs=60)

    assert np.allclose(latent_a, latent_b, atol=1e-6)
