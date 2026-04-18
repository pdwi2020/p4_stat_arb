"""Cointegration helpers for pair and basket screening."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def _normalize_weights(values: np.ndarray) -> np.ndarray:
    weights = np.asarray(values, dtype=float)
    weights = weights - np.mean(weights)
    if np.allclose(weights, 0.0):
        weights = np.asarray(values, dtype=float)
    norm = np.sum(np.abs(weights))
    if norm <= 0:
        raise ValueError("Cannot normalize zero weight vector.")
    return weights / norm


def engle_granger_test(log_prices: pd.DataFrame, alpha: float = 0.05) -> dict[str, float | bool | np.ndarray | pd.Series]:
    """Run a simple Engle-Granger screen on a two-asset log-price panel."""

    if log_prices.shape[1] != 2:
        raise ValueError("Engle-Granger requires exactly two assets.")
    clean = log_prices.dropna().copy()
    if len(clean) < 60:
        raise ValueError("Not enough observations for Engle-Granger.")

    y = clean.iloc[:, 0].to_numpy(dtype=float)
    x = clean.iloc[:, 1].to_numpy(dtype=float)
    beta = float(np.dot(x, y) / np.dot(x, x))
    spread = clean.iloc[:, 0] - beta * clean.iloc[:, 1]
    coint_stat, coint_pvalue, _ = coint(clean.iloc[:, 0], clean.iloc[:, 1], trend="c", autolag="aic")
    adf_stat, adf_pvalue, _, _, critical_values, _ = adfuller(spread.to_numpy(dtype=float), autolag="AIC")
    weights = _normalize_weights(np.array([1.0, -beta], dtype=float))
    return {
        "pass": bool(adf_pvalue < alpha),
        "beta": beta,
        "weights": weights,
        "spread": spread,
        "coint_stat": float(coint_stat),
        "coint_pvalue": float(coint_pvalue),
        "adf_stat": float(adf_stat),
        "adf_pvalue": float(adf_pvalue),
        "adf_critical_5pct": float(critical_values["5%"]),
    }


def johansen_test(log_prices: pd.DataFrame, alpha: float = 0.05) -> dict[str, float | bool | np.ndarray | pd.Series]:
    """Run Johansen rank-1 screening on a three-asset basket."""

    if log_prices.shape[1] != 3:
        raise ValueError("Johansen test is restricted to 3-asset baskets in P4 v1.")
    clean = log_prices.dropna().copy()
    if len(clean) < 90:
        raise ValueError("Not enough observations for Johansen test.")

    result = coint_johansen(clean.to_numpy(dtype=float), det_order=0, k_ar_diff=1)
    alpha_index = 1 if alpha <= 0.05 else 0
    trace_stat = float(result.lr1[0])
    critical_value = float(result.cvt[0, alpha_index])
    weights = _normalize_weights(result.evec[:, 0])
    spread = pd.Series(clean.to_numpy(dtype=float) @ weights, index=clean.index, name="spread")
    return {
        "pass": bool(trace_stat > critical_value),
        "weights": weights,
        "spread": spread,
        "trace_stat": trace_stat,
        "critical_value": critical_value,
        "eigenvalue": float(result.eig[0]),
    }
