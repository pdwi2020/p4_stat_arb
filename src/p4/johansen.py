"""Johansen basket cointegration and VECM helpers.

References
----------
Johansen (1988, 1991), "Estimation and hypothesis testing of cointegration
vectors in Gaussian vector autoregressive models"; Lütkepohl (2005), section 6;
Hamilton (1994), chapter 19.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen


_ALPHA_TO_COL = {0.10: 0, 0.05: 1, 0.01: 2}
_DET_TO_VECM = {-1: "n", 0: "ci", 1: "li"}


def _validate_log_prices(log_prices: pd.DataFrame, *, min_obs: int = 60) -> pd.DataFrame:
    if not isinstance(log_prices, pd.DataFrame):
        raise ValueError("log_prices must be a pandas DataFrame.")
    if log_prices.shape[1] < 2:
        raise ValueError("Johansen methods require at least two assets.")
    if log_prices.empty:
        raise ValueError("log_prices must be non-empty.")
    if log_prices.isna().all(axis=1).any():
        raise ValueError("log_prices contains all-NaN rows.")
    if log_prices.isna().all(axis=0).any():
        raise ValueError("log_prices contains all-NaN columns.")

    clean = log_prices.astype("float64").dropna(how="any")
    if len(clean) < min_obs:
        raise ValueError(f"Need at least {min_obs} complete observations for Johansen methods.")
    return clean


def _critical_value_column(alpha: float) -> int:
    for level, col in _ALPHA_TO_COL.items():
        if np.isclose(alpha, level):
            return col
    raise ValueError("alpha must be one of {0.10, 0.05, 0.01}.")


def _rank_from_stats(stats: np.ndarray, critical_values: np.ndarray) -> int:
    for rank, (stat, crit) in enumerate(zip(stats, critical_values, strict=False)):
        if float(stat) <= float(crit):
            return int(rank)
    return int(len(stats))


def _normalize_weights(vector: np.ndarray) -> np.ndarray:
    weights = np.asarray(vector, dtype=np.float64)
    norm = float(np.sum(np.abs(weights)))
    if norm <= 0.0:
        raise ValueError("Cannot normalize a zero cointegrating vector.")
    return weights / norm


def johansen_test(
    log_prices: pd.DataFrame,
    det_order: int = 0,
    k_ar_diff: int = 1,
    alpha: float = 0.05,
) -> dict[str, int | np.ndarray]:
    """Run the Johansen rank test on an N-asset log-price panel."""

    if det_order not in _DET_TO_VECM:
        raise ValueError("det_order must be -1, 0, or 1.")
    if k_ar_diff < 0:
        raise ValueError("k_ar_diff must be non-negative.")

    clean = _validate_log_prices(log_prices)
    alpha_col = _critical_value_column(alpha)
    result = coint_johansen(clean.to_numpy(dtype=np.float64), det_order=det_order, k_ar_diff=k_ar_diff)

    trace_stats = np.asarray(getattr(result, "trace_stat", result.lr1), dtype=np.float64)
    trace_critical_values = np.asarray(getattr(result, "trace_stat_crit_vals", result.cvt), dtype=np.float64)
    max_eig_stats = np.asarray(getattr(result, "max_eig_stat", result.lr2), dtype=np.float64)
    max_eig_critical_values = np.asarray(getattr(result, "max_eig_stat_crit_vals", result.cvm), dtype=np.float64)
    eigenvectors = np.asarray(result.evec, dtype=np.float64)
    eigenvalues = np.asarray(result.eig, dtype=np.float64)

    return {
        "trace_stats": trace_stats,
        "trace_critical_values": trace_critical_values,
        "max_eig_stats": max_eig_stats,
        "max_eig_critical_values": max_eig_critical_values,
        "rank_estimate_trace": _rank_from_stats(trace_stats, trace_critical_values[:, alpha_col]),
        "rank_estimate_max_eig": _rank_from_stats(max_eig_stats, max_eig_critical_values[:, alpha_col]),
        "eigenvectors": eigenvectors,
        "eigenvalues": eigenvalues,
    }


def johansen_basket_weights(log_prices: pd.DataFrame, **kwargs: int | float) -> dict[str, int | bool | float | pd.Series]:
    """Extract the leading Johansen cointegrating vector as basket weights."""

    clean = _validate_log_prices(log_prices)
    test_result = johansen_test(clean, **kwargs)
    weights = _normalize_weights(np.asarray(test_result["eigenvectors"], dtype=np.float64)[:, 0])
    spread = pd.Series(clean.to_numpy(dtype=np.float64) @ weights, index=clean.index, name="spread", dtype="float64")
    rank = int(test_result["rank_estimate_trace"])
    return {
        "weights": pd.Series(weights, index=clean.columns, name="weights", dtype="float64"),
        "spread": spread,
        "rank": rank,
        "is_cointegrated": bool(rank > 0),
        "trace_stat": float(np.asarray(test_result["trace_stats"], dtype=np.float64)[0]),
    }


def vecm_fit(
    log_prices: pd.DataFrame,
    k_ar_diff: int = 1,
    det_order: int = 0,
    coint_rank: int | None = None,
) -> dict[str, int | np.ndarray | str]:
    """Fit a VECM and return the alpha/beta decomposition."""

    if det_order not in _DET_TO_VECM:
        raise ValueError("det_order must be -1, 0, or 1.")
    if k_ar_diff < 0:
        raise ValueError("k_ar_diff must be non-negative.")

    clean = _validate_log_prices(log_prices)
    rank = int(coint_rank) if coint_rank is not None else int(johansen_test(clean, det_order=det_order, k_ar_diff=k_ar_diff)["rank_estimate_trace"])
    if rank < 1:
        raise ValueError("VECM requires a positive cointegration rank.")

    model = VECM(
        clean.to_numpy(dtype=np.float64),
        k_ar_diff=k_ar_diff,
        coint_rank=rank,
        deterministic=_DET_TO_VECM[det_order],
    )
    fit = model.fit()
    return {
        "alpha_loadings": np.asarray(fit.alpha, dtype=np.float64),
        "beta_cointegration": np.asarray(fit.beta, dtype=np.float64),
        "fit_summary": str(fit.summary()),
        "rank": rank,
    }
