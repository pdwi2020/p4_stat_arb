"""Regime-switching utilities for pair selection.

The model is intentionally small and transparent: a Gaussian-emission HMM fitted
with Baum-Welch EM plus a state-conditional AR(1) half-life estimator. The HMM
captures level/volatility regimes; mean-reversion speed is estimated in a
second step from state-weighted dynamics rather than assumed by the emissions.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def _validate_observations(observations: NDArray | pd.Series | list[float]) -> NDArray[np.float64]:
    values = np.asarray(observations, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("observations must be non-empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError("observations contain non-finite values.")
    return values


def _gaussian_pdf(values: NDArray[np.float64], means: NDArray[np.float64], variances: NDArray[np.float64]) -> NDArray[np.float64]:
    diff = values[:, None] - means[None, :]
    safe_variances = np.maximum(variances, 1e-8)
    scale = np.sqrt(2.0 * math.pi * safe_variances)
    density = np.exp(-0.5 * diff * diff / safe_variances[None, :]) / scale[None, :]
    return np.clip(density, 1e-300, None)


def _overall_ar_slope(values: NDArray[np.float64]) -> float:
    if values.size < 3:
        return 0.5
    x = values[:-1]
    y = values[1:]
    design = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(np.clip(abs(coef[1]), 1e-6, 0.999999))


def _weighted_half_life(
    values: NDArray[np.float64],
    state_probs: NDArray[np.float64],
    *,
    reference_slope: float,
) -> float:
    if values.size < 3:
        return float("inf")
    weights = np.sqrt(np.clip(state_probs[:-1], 0.0, None) * np.clip(state_probs[1:], 0.0, None))
    if float(weights.sum()) <= 1e-6:
        return float("inf")

    x = values[:-1]
    y = values[1:]
    design = np.column_stack([np.ones_like(x), x])
    sqrt_w = np.sqrt(weights)
    coef, *_ = np.linalg.lstsq(design * sqrt_w[:, None], y * sqrt_w, rcond=None)
    slope = float(abs(coef[1]))
    slope = 0.65 * float(reference_slope) + 0.35 * slope
    slope = float(np.clip(slope, 1e-6, 0.999999))
    return float(math.log(2.0) / max(-math.log(slope), 1e-8))


class GaussianHMM:
    """Gaussian emission HMM with Baum-Welch estimation and Viterbi decoding."""

    def __init__(
        self,
        n_regimes: int = 2,
        emission: str = "gaussian",
        *,
        max_iter: int = 100,
        tol: float = 1e-4,
        min_variance: float = 1e-6,
        random_seed: int = 0,
    ) -> None:
        if n_regimes < 2:
            raise ValueError("n_regimes must be at least 2.")
        if emission != "gaussian":
            raise ValueError("Only gaussian emissions are supported.")
        self.n_regimes = int(n_regimes)
        self.emission = emission
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.min_variance = float(min_variance)
        self.random_seed = int(random_seed)

        self.initial_probs = np.full(self.n_regimes, 1.0 / self.n_regimes, dtype=float)
        self.transition_matrix = np.full((self.n_regimes, self.n_regimes), 1.0 / self.n_regimes, dtype=float)
        self.emission_params: list[dict[str, float]] = []
        self.log_likelihood = float("-inf")
        self.log_likelihood_history: list[float] = []
        self._fitted = False

    def _initialize(self, observations: NDArray[np.float64]) -> None:
        centered = observations - float(np.mean(observations))
        base_variance = float(np.var(centered, ddof=0))
        quantiles = np.linspace(0.1, 0.9, self.n_regimes)
        means = np.quantile(observations, quantiles)
        if np.allclose(means, means[0]):
            rng = np.random.default_rng(self.random_seed)
            means = means + rng.normal(scale=max(np.std(observations), 1e-3), size=self.n_regimes) * 0.05
        self.initial_probs = np.full(self.n_regimes, 1.0 / self.n_regimes, dtype=float)
        self.transition_matrix = np.full((self.n_regimes, self.n_regimes), 0.1 / max(self.n_regimes - 1, 1), dtype=float)
        np.fill_diagonal(self.transition_matrix, 0.9)
        self.emission_params = [
            {"mean": float(mean), "variance": max(base_variance, self.min_variance)}
            for mean in np.sort(means)
        ]
        self._sort_regimes()

    def _sort_regimes(self) -> None:
        means = np.asarray([params["mean"] for params in self.emission_params], dtype=float)
        order = np.argsort(means)
        self.initial_probs = self.initial_probs[order]
        self.transition_matrix = self.transition_matrix[np.ix_(order, order)]
        self.emission_params = [self.emission_params[int(idx)] for idx in order]

    def _emission_matrix(self, observations: NDArray[np.float64]) -> NDArray[np.float64]:
        means = np.asarray([params["mean"] for params in self.emission_params], dtype=float)
        variances = np.asarray([params["variance"] for params in self.emission_params], dtype=float)
        return _gaussian_pdf(observations, means, variances)

    def _forward_backward(
        self,
        observations: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float]:
        emit = self._emission_matrix(observations)
        length = observations.size
        alpha = np.zeros((length, self.n_regimes), dtype=float)
        beta = np.zeros_like(alpha)
        scales = np.zeros(length, dtype=float)

        alpha[0] = self.initial_probs * emit[0]
        scales[0] = max(float(alpha[0].sum()), 1e-300)
        alpha[0] /= scales[0]

        for t in range(1, length):
            alpha[t] = (alpha[t - 1] @ self.transition_matrix) * emit[t]
            scales[t] = max(float(alpha[t].sum()), 1e-300)
            alpha[t] /= scales[t]

        beta[-1] = 1.0
        for t in range(length - 2, -1, -1):
            beta[t] = (self.transition_matrix * (emit[t + 1] * beta[t + 1])[None, :]).sum(axis=1)
            beta[t] /= max(scales[t + 1], 1e-300)

        gamma = alpha * beta
        gamma /= np.clip(gamma.sum(axis=1, keepdims=True), 1e-300, None)

        xi = np.zeros((max(length - 1, 0), self.n_regimes, self.n_regimes), dtype=float)
        for t in range(length - 1):
            numer = alpha[t][:, None] * self.transition_matrix * (emit[t + 1] * beta[t + 1])[None, :]
            denom = max(float(numer.sum()), 1e-300)
            xi[t] = numer / denom

        log_likelihood = float(np.log(scales).sum())
        return gamma, xi, emit, log_likelihood

    def fit(self, observations: NDArray | pd.Series | list[float]) -> GaussianHMM:
        values = _validate_observations(observations)
        if values.size < self.n_regimes * 5:
            raise ValueError("Not enough observations to fit the HMM.")

        self._initialize(values)
        self.log_likelihood_history = []

        for _ in range(self.max_iter):
            gamma, xi, _, log_likelihood = self._forward_backward(values)
            self.log_likelihood_history.append(log_likelihood)

            self.initial_probs = gamma[0]
            if xi.size:
                trans_numer = xi.sum(axis=0)
                trans_denom = np.clip(gamma[:-1].sum(axis=0)[:, None], 1e-300, None)
                self.transition_matrix = trans_numer / trans_denom
                self.transition_matrix /= np.clip(self.transition_matrix.sum(axis=1, keepdims=True), 1e-300, None)

            regime_weights = np.clip(gamma.sum(axis=0), 1e-300, None)
            means = (gamma * values[:, None]).sum(axis=0) / regime_weights
            variances = (gamma * (values[:, None] - means[None, :]) ** 2).sum(axis=0) / regime_weights
            self.emission_params = [
                {"mean": float(mean), "variance": float(max(variance, self.min_variance))}
                for mean, variance in zip(means, variances, strict=True)
            ]
            self._sort_regimes()

            if len(self.log_likelihood_history) >= 2:
                improvement = self.log_likelihood_history[-1] - self.log_likelihood_history[-2]
                if improvement >= 0.0 and improvement < self.tol:
                    break

        self.log_likelihood = float(self.log_likelihood_history[-1])
        self._fitted = True
        return self

    def smoothed_probs(self, observations: NDArray | pd.Series | list[float]) -> NDArray[np.float64]:
        if not self._fitted:
            raise ValueError("Model must be fitted before calling smoothed_probs.")
        values = _validate_observations(observations)
        gamma, _, _, _ = self._forward_backward(values)
        return gamma

    def predict_states(self, observations: NDArray | pd.Series | list[float]) -> NDArray[np.int_]:
        if not self._fitted:
            raise ValueError("Model must be fitted before calling predict_states.")
        values = _validate_observations(observations)
        emit = self._emission_matrix(values)
        log_emit = np.log(np.clip(emit, 1e-300, None))
        log_initial = np.log(np.clip(self.initial_probs, 1e-300, None))
        log_transition = np.log(np.clip(self.transition_matrix, 1e-300, None))

        delta = np.zeros((values.size, self.n_regimes), dtype=float)
        psi = np.zeros((values.size, self.n_regimes), dtype=int)
        delta[0] = log_initial + log_emit[0]

        for t in range(1, values.size):
            scores = delta[t - 1][:, None] + log_transition
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = scores[psi[t], np.arange(self.n_regimes)] + log_emit[t]

        states = np.zeros(values.size, dtype=int)
        states[-1] = int(np.argmax(delta[-1]))
        for t in range(values.size - 2, -1, -1):
            states[t] = int(psi[t + 1, states[t + 1]])
        return states


def fit_pair_regime(spread_series: pd.Series, n_regimes: int = 2) -> dict[str, object]:
    """Fit a Gaussian HMM to a spread and derive regime-specific half-lives."""

    spread = pd.Series(spread_series, dtype=float).dropna()
    if spread.empty:
        raise ValueError("spread_series must be non-empty.")

    model = GaussianHMM(n_regimes=n_regimes)
    model.fit(spread.to_numpy(dtype=float))
    smoothed = model.smoothed_probs(spread.to_numpy(dtype=float))
    viterbi = model.predict_states(spread.to_numpy(dtype=float))
    reference_slope = _overall_ar_slope(spread.to_numpy(dtype=float))

    regime_means = np.asarray([params["mean"] for params in model.emission_params], dtype=float)
    regime_vols = np.sqrt(np.asarray([params["variance"] for params in model.emission_params], dtype=float))
    half_lives = np.asarray(
        [
            _weighted_half_life(
                spread.to_numpy(dtype=float),
                smoothed[:, idx],
                reference_slope=reference_slope,
            )
            for idx in range(n_regimes)
        ],
        dtype=float,
    )
    finite_half_lives = np.where(np.isfinite(half_lives), half_lives, np.inf)
    high_reversion_regime_idx = int(np.argmin(finite_half_lives))

    return {
        "regime_means": regime_means,
        "regime_vols": regime_vols,
        "transition_matrix": model.transition_matrix.copy(),
        "viterbi": pd.Series(viterbi, index=spread.index, name="regime", dtype=int),
        "smoothed_probs": pd.DataFrame(
            smoothed,
            index=spread.index,
            columns=[f"regime_{idx}" for idx in range(n_regimes)],
        ),
        "half_life_per_regime": half_lives,
        "high_reversion_regime_idx": high_reversion_regime_idx,
        "log_likelihood": model.log_likelihood,
        "log_likelihood_history": list(model.log_likelihood_history),
        "model": model,
    }


def regime_filtered_trading_signal(
    spread: pd.Series,
    regime: pd.Series,
    active_regime: int,
    z_threshold: float = 2.0,
) -> pd.Series:
    """Emit ±1/0 signals only when the spread is in the selected regime."""

    if z_threshold <= 0.0:
        raise ValueError("z_threshold must be positive.")

    spread_series = pd.Series(spread, dtype=float).dropna()
    if spread_series.empty:
        raise ValueError("spread must be non-empty.")

    regime_series = pd.Series(regime).reindex(spread_series.index)
    if regime_series.isna().any():
        raise ValueError("regime must align with spread and contain no missing values.")

    active_mask = regime_series.astype(int) == int(active_regime)
    if not bool(active_mask.any()):
        return pd.Series(0.0, index=spread_series.index, name="regime_signal", dtype=float)

    active_values = spread_series.loc[active_mask]
    mu = float(active_values.mean())
    sigma = float(active_values.std(ddof=0))
    if sigma <= 1e-8:
        sigma = float(spread_series.std(ddof=0))
    sigma = max(sigma, 1e-8)
    zscore = (spread_series - mu) / sigma

    signal = np.zeros(len(spread_series), dtype=float)
    signal[(active_mask.to_numpy()) & (zscore.to_numpy() <= -z_threshold)] = 1.0
    signal[(active_mask.to_numpy()) & (zscore.to_numpy() >= z_threshold)] = -1.0
    return pd.Series(signal, index=spread_series.index, name="regime_signal", dtype=float)
