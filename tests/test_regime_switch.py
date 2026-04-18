from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from p4.regime_switch import GaussianHMM, fit_pair_regime, regime_filtered_trading_signal


def _simulate_markov_gaussian(
    *,
    length: int = 500,
    means: tuple[float, float] = (-1.2, 1.4),
    vols: tuple[float, float] = (0.25, 0.35),
    transition: np.ndarray | None = None,
    seed: int = 11,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    matrix = transition if transition is not None else np.array([[0.95, 0.05], [0.08, 0.92]], dtype=float)
    states = np.zeros(length, dtype=int)
    observations = np.zeros(length, dtype=float)
    for t in range(1, length):
        states[t] = int(rng.choice(2, p=matrix[states[t - 1]]))
    for t, state in enumerate(states):
        observations[t] = rng.normal(loc=means[state], scale=vols[state])
    return observations, states


def _simulate_switching_ou(
    *,
    length: int = 600,
    seed: int = 19,
) -> tuple[pd.Series, np.ndarray]:
    rng = np.random.default_rng(seed)
    transition = np.array([[0.95, 0.05], [0.06, 0.94]], dtype=float)
    phi = np.array([np.exp(-1.0 / 8.0), np.exp(-1.0 / 24.0)], dtype=float)
    means = np.array([-0.4, 0.6], dtype=float)
    sigmas = np.array([0.10, 0.18], dtype=float)

    states = np.zeros(length, dtype=int)
    values = np.zeros(length, dtype=float)
    values[0] = means[0]
    for t in range(1, length):
        states[t] = int(rng.choice(2, p=transition[states[t - 1]]))
        state = states[t]
        values[t] = means[state] + phi[state] * (values[t - 1] - means[state]) + rng.normal(scale=sigmas[state])
    index = pd.bdate_range("2023-01-02", periods=length)
    return pd.Series(values, index=index, name="spread"), states


def test_fit_two_regime_hmm_recovers_means() -> None:
    observations, _ = _simulate_markov_gaussian()
    model = GaussianHMM(n_regimes=2, max_iter=80, tol=1e-5)
    model.fit(observations)

    estimated = np.sort([params["mean"] for params in model.emission_params])
    assert np.allclose(estimated, np.array([-1.2, 1.4]), atol=0.2)


def test_viterbi_assigns_correct_regimes_on_synthetic_series() -> None:
    observations, states = _simulate_markov_gaussian()
    model = GaussianHMM(n_regimes=2, max_iter=80, tol=1e-5)
    model.fit(observations)
    predicted = model.predict_states(observations)

    accuracy = max(float(np.mean(predicted == states)), float(np.mean(1 - predicted == states)))
    assert accuracy >= 0.80


def test_transition_matrix_rows_sum_to_one() -> None:
    observations, _ = _simulate_markov_gaussian()
    model = GaussianHMM(n_regimes=2)
    model.fit(observations)

    assert np.allclose(model.transition_matrix.sum(axis=1), 1.0, atol=1e-6)


def test_smoothed_probabilities_sum_to_one() -> None:
    observations, _ = _simulate_markov_gaussian()
    model = GaussianHMM(n_regimes=2)
    model.fit(observations)
    smoothed = model.smoothed_probs(observations)

    assert np.allclose(smoothed.sum(axis=1), 1.0, atol=1e-6)


def test_log_likelihood_is_monotone_during_em() -> None:
    observations, _ = _simulate_markov_gaussian()
    model = GaussianHMM(n_regimes=2, max_iter=60, tol=1e-6)
    model.fit(observations)

    diffs = np.diff(model.log_likelihood_history)
    assert np.all(diffs >= -1e-6)


def test_regime_filtered_signal_trades_only_in_active_regime() -> None:
    spread = pd.Series([-3.0, -2.5, -0.2, 2.5, 3.1, 0.1], index=pd.RangeIndex(6), dtype=float)
    regime = pd.Series([1, 1, 0, 0, 1, 0], index=spread.index, dtype=int)
    signal = regime_filtered_trading_signal(spread, regime, active_regime=1, z_threshold=1.0)

    assert (signal.loc[regime != 1] == 0.0).all()
    assert set(signal.loc[regime == 1].unique()).issubset({-1.0, 0.0, 1.0})


def test_input_validation_for_bad_regime_count_and_empty_input() -> None:
    with pytest.raises(ValueError):
        GaussianHMM(n_regimes=1)
    with pytest.raises(ValueError):
        GaussianHMM(n_regimes=2).fit([])
    with pytest.raises(ValueError):
        fit_pair_regime(pd.Series(dtype=float), n_regimes=2)


def test_pair_half_life_estimator_returns_reasonable_values() -> None:
    spread, _ = _simulate_switching_ou()
    fit = fit_pair_regime(spread, n_regimes=2)
    half_lives = np.asarray(fit["half_life_per_regime"], dtype=float)

    assert np.all(np.isfinite(half_lives))
    assert np.all((half_lives >= 5.0) & (half_lives <= 100.0))
    assert int(fit["high_reversion_regime_idx"]) == int(np.argmin(half_lives))
