"""Kalman-filter estimation for time-varying OU parameters.

References
----------
Harvey (1989), "Forecasting, structural time series models and the Kalman
filter"; Durbin and Koopman (2012), section 4.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from p4.ou_estimator import OUEstimator


def static_ou_initial_estimate(spread: pd.Series) -> tuple[float, float]:
    """Estimate static OU kappa and mu by OLS."""

    params = OUEstimator().fit(spread)
    return float(params["kappa"]), float(params["mu"])


class KalmanOU:
    """Track time-varying OU parameters with a linear-Gaussian Kalman filter."""

    def fit(self, spread: pd.Series, q_var: float = 1e-5, r_var: float | None = None) -> dict[str, float | np.ndarray]:
        clean = pd.Series(spread, dtype="float64").dropna()
        if len(clean) < 30:
            raise ValueError("Not enough observations to fit OU parameters.")
        if q_var <= 0.0:
            raise ValueError("q_var must be positive.")

        values = clean.to_numpy(dtype=np.float64)
        x_prev = values[:-1]
        x_next = values[1:]
        design = np.column_stack([np.ones_like(x_prev), x_prev]).astype(np.float64)
        intercept0, phi0 = np.linalg.lstsq(design, x_next, rcond=None)[0].astype(np.float64)
        phi0 = float(np.clip(phi0, 1e-6, 0.999999))
        intercept0 = float(intercept0)
        residual0 = x_next - (intercept0 + phi0 * x_prev)
        obs_var = float(np.var(residual0, ddof=1)) if len(residual0) > 1 else 1e-6
        obs_var = float(r_var if r_var is not None else obs_var)
        obs_var = max(obs_var, 1e-8)

        kappa0, mu0 = static_ou_initial_estimate(clean)
        phi_init = float(math.exp(-kappa0))
        state = np.array([mu0 * (1.0 - phi_init), phi_init], dtype=np.float64)
        cov = np.eye(2, dtype=np.float64) * max(obs_var, 1e-4)
        process_cov = np.diag([0.1 * float(q_var), 4.0 * float(q_var)]).astype(np.float64)
        identity = np.eye(2, dtype=np.float64)

        kappa_path: list[float] = []
        mu_path: list[float] = []
        intercept_path: list[float] = []
        phi_path: list[float] = []
        innovations: list[float] = []
        log_likelihood = 0.0

        for prev_value, next_value in zip(x_prev, x_next, strict=False):
            pred_state = state.copy()
            pred_cov = cov + process_cov
            obs_matrix = np.array([[1.0, float(prev_value)]], dtype=np.float64)
            predicted = float((obs_matrix @ pred_state).item())
            innovation = float(next_value - predicted)
            innovation_var = float((obs_matrix @ pred_cov @ obs_matrix.T).item() + obs_var)
            gain = (pred_cov @ obs_matrix.T) / innovation_var
            state = pred_state + gain[:, 0] * innovation
            state[1] = float(np.clip(state[1], 1e-6, 0.999999))
            cov = (identity - gain @ obs_matrix) @ pred_cov

            phi = float(np.clip(state[1], 1e-6, 0.999999))
            kappa = float(-math.log(phi))
            mu = float(state[0] / max(1.0 - phi, 1e-8))
            intercept_path.append(float(state[0]))
            phi_path.append(phi)
            kappa_path.append(kappa)
            mu_path.append(mu)
            innovations.append(innovation)
            log_likelihood += -0.5 * (math.log(2.0 * math.pi * innovation_var) + (innovation**2) / innovation_var)

        residual_std = float(np.std(np.asarray(innovations, dtype=np.float64), ddof=1)) if len(innovations) > 1 else 0.0
        filtered_kappa = float(kappa_path[-1])
        filtered_mu = float(mu_path[-1])
        summary_phi = float(np.clip(np.mean(np.asarray(phi_path, dtype=np.float64)), 1e-6, 0.999999))
        summary_intercept = float(np.mean(np.asarray(intercept_path, dtype=np.float64)))
        summary_kappa = float(-math.log(summary_phi))
        summary_mu = float(summary_intercept / max(1.0 - summary_phi, 1e-8))
        stationary_sigma = float(residual_std / math.sqrt(max(1.0 - summary_phi**2, 1e-8)))
        sigma = float(stationary_sigma * math.sqrt(max(2.0 * summary_kappa, 1e-8)))
        half_life = float(math.log(2.0) / max(summary_kappa, 1e-8))

        return {
            "kappa": summary_kappa,
            "mu": summary_mu,
            "sigma": sigma,
            "stationary_sigma": stationary_sigma,
            "half_life": half_life,
            "kappa_path": np.asarray(kappa_path, dtype=np.float64),
            "mu_path": np.asarray(mu_path, dtype=np.float64),
            "log_likelihood": float(log_likelihood),
            "residual_std": residual_std,
            "slope": summary_phi,
            "intercept": summary_intercept,
            "kappa_filtered_final": filtered_kappa,
            "mu_filtered_final": filtered_mu,
        }
