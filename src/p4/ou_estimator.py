"""Ornstein-Uhlenbeck parameter estimation for spreads."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


class OUEstimator:
    """Estimate OU parameters from a univariate spread series."""

    def fit(self, spread: pd.Series) -> dict[str, float]:
        clean = pd.Series(spread, dtype=float).dropna()
        if len(clean) < 30:
            raise ValueError("Not enough observations to fit OU parameters.")

        x = clean.iloc[:-1].to_numpy(dtype=float)
        y = clean.iloc[1:].to_numpy(dtype=float)
        design = np.column_stack([np.ones_like(x), x])
        intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
        slope = float(np.clip(slope, 1e-6, 0.999999))
        kappa = float(-math.log(slope))
        mu = float(intercept / (1.0 - slope))
        residual = y - (intercept + slope * x)
        residual_std = float(np.std(residual, ddof=1)) if len(residual) > 1 else 0.0
        stationary_sigma = float(residual_std / math.sqrt(max(1.0 - slope**2, 1e-8)))
        sigma = float(stationary_sigma * math.sqrt(max(2.0 * kappa, 1e-8)))
        half_life = float(math.log(2.0) / max(kappa, 1e-8))
        return {
            "kappa": kappa,
            "mu": mu,
            "sigma": sigma,
            "stationary_sigma": stationary_sigma,
            "half_life": half_life,
            "residual_std": residual_std,
            "slope": slope,
            "intercept": float(intercept),
        }

    def half_life(self, spread: pd.Series) -> float:
        return float(self.fit(spread)["half_life"])
