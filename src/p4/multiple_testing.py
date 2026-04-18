"""Multiple-testing correction utilities for P4.

The module keeps the original family-wise controls used in P4 and extends them
with lighter-touch false-discovery-rate procedures. BH/BHY are useful when the
goal is ranking or screening a broad candidate set, while Romano-Wolf retains
FWER control under dependence at the cost of bootstrap work and lower power.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats


def bonferroni_threshold(alpha: float, n_tests: int) -> float:
    return float(alpha / max(n_tests, 1))


def one_sided_mean_pvalues(return_df: pd.DataFrame) -> pd.Series:
    pvalues: dict[str, float] = {}
    for column in return_df.columns:
        sample = pd.Series(return_df[column], dtype=float).dropna()
        if len(sample) < 3 or float(sample.std()) == 0.0:
            pvalues[column] = 1.0
            continue
        t_stat, two_sided = stats.ttest_1samp(sample, popmean=0.0)
        if np.isnan(t_stat) or np.isnan(two_sided):
            pvalues[column] = 1.0
        elif t_stat > 0:
            pvalues[column] = float(two_sided / 2.0)
        else:
            pvalues[column] = float(1.0 - two_sided / 2.0)
    return pd.Series(pvalues).sort_index()


def _block_bootstrap_indices(length: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    if length <= 0:
        return np.array([], dtype=int)
    block_size = max(1, min(block_size, length))
    indices = []
    while len(indices) < length:
        start = int(rng.integers(0, max(length - block_size + 1, 1)))
        indices.extend(range(start, min(start + block_size, length)))
    return np.asarray(indices[:length], dtype=int)


def white_reality_check(
    return_df: pd.DataFrame,
    *,
    n_bootstrap: int,
    block_size: int,
    random_seed: int,
) -> dict[str, float | str | list[float] | int]:
    returns = return_df.fillna(0.0).to_numpy(dtype=float)
    if returns.size == 0:
        return {"pvalue": 1.0, "best_strategy": None, "observed_max_mean": 0.0, "block_size": block_size}

    rng = np.random.default_rng(random_seed)
    means = returns.mean(axis=0)
    observed_max_mean = float(np.max(means))
    centered = returns - means
    boot_stats = []
    for _ in range(n_bootstrap):
        idx = _block_bootstrap_indices(len(returns), block_size, rng)
        boot_stats.append(float(np.max(centered[idx].mean(axis=0))))

    best_strategy = str(return_df.columns[int(np.argmax(means))])
    pvalue = float(np.mean(np.asarray(boot_stats) >= observed_max_mean))
    return {
        "pvalue": pvalue,
        "best_strategy": best_strategy,
        "observed_max_mean": observed_max_mean,
        "block_size": int(block_size),
    }


def hansen_spa_test(
    return_df: pd.DataFrame,
    *,
    n_bootstrap: int,
    block_size: int,
    alpha: float,
    random_seed: int,
) -> dict[str, object]:
    returns = return_df.fillna(0.0).to_numpy(dtype=float)
    if returns.size == 0:
        return {"pvalue": 1.0, "threshold": 0.0, "survivors": [], "block_size": block_size}

    rng = np.random.default_rng(random_seed)
    means = returns.mean(axis=0)
    std = returns.std(axis=0, ddof=1)
    std = np.where(std <= 1e-12, np.nan, std)
    t_obs = np.where(np.isnan(std), 0.0, np.sqrt(len(returns)) * np.maximum(means, 0.0) / std)
    centered = returns - np.maximum(means, 0.0)

    boot_max = []
    for _ in range(n_bootstrap):
        idx = _block_bootstrap_indices(len(returns), block_size, rng)
        sample = centered[idx]
        boot_mean = sample.mean(axis=0)
        boot_t = np.where(np.isnan(std), 0.0, np.sqrt(len(sample)) * np.maximum(boot_mean, 0.0) / std)
        boot_max.append(float(np.nanmax(boot_t)))

    threshold = float(np.quantile(boot_max, 1.0 - alpha))
    survivors = [str(column) for column, stat in zip(return_df.columns, t_obs, strict=True) if stat > threshold]
    pvalue = float(np.mean(np.asarray(boot_max) >= np.nanmax(t_obs)))
    candidate_pvalues = {
        str(column): float(np.mean(np.asarray(boot_max) >= float(stat)))
        for column, stat in zip(return_df.columns, t_obs, strict=True)
    }
    return {
        "pvalue": pvalue,
        "threshold": threshold,
        "survivors": survivors,
        "candidate_pvalues": candidate_pvalues,
        "block_size": int(block_size),
    }


def _as_1d_float_array(values: NDArray | list[float] | tuple[float, ...]) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("Expected a one-dimensional array.")
    if array.size == 0:
        return array.astype(float)
    if not np.all(np.isfinite(array)):
        raise ValueError("Input contains non-finite values.")
    return array


def _validate_pvalues(pvalues: NDArray | list[float] | tuple[float, ...]) -> NDArray[np.float64]:
    array = _as_1d_float_array(pvalues)
    if array.size and (np.any(array < 0.0) or np.any(array > 1.0)):
        raise ValueError("P-values must lie in [0, 1].")
    return array


def _step_up_adjusted_pvalues(ordered_pvalues: NDArray[np.float64], scale: float) -> NDArray[np.float64]:
    if ordered_pvalues.size == 0:
        return ordered_pvalues.astype(float)
    ranks = np.arange(1, ordered_pvalues.size + 1, dtype=float)
    adjusted = np.minimum.accumulate((scale * ordered_pvalues.size * ordered_pvalues / ranks)[::-1])[::-1]
    return np.clip(adjusted, 0.0, 1.0)


def _unsort(values: NDArray[np.float64] | NDArray[np.bool_], order: NDArray[np.int_]) -> NDArray:
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    return values[inverse]


def benjamini_hochberg(
    pvalues: NDArray | list[float] | tuple[float, ...],
    alpha: float = 0.05,
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Benjamini-Hochberg (1995) step-up FDR procedure."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1).")
    raw = _validate_pvalues(pvalues)
    if raw.size == 0:
        return np.zeros(0, dtype=bool), raw

    order = np.argsort(raw, kind="mergesort")
    ordered = raw[order]
    thresholds = alpha * np.arange(1, ordered.size + 1, dtype=float) / ordered.size
    passing = np.flatnonzero(ordered <= thresholds)
    reject_sorted = np.zeros(ordered.size, dtype=bool)
    if passing.size:
        reject_sorted[: passing[-1] + 1] = True
    adjusted_sorted = _step_up_adjusted_pvalues(ordered, scale=1.0)
    return _unsort(reject_sorted, order), _unsort(adjusted_sorted, order)


def benjamini_yekutieli(
    pvalues: NDArray | list[float] | tuple[float, ...],
    alpha: float = 0.05,
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Benjamini-Yekutieli (2001) dependency-robust FDR control."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1).")
    raw = _validate_pvalues(pvalues)
    if raw.size == 0:
        return np.zeros(0, dtype=bool), raw

    harmonic_number = float(np.sum(1.0 / np.arange(1, raw.size + 1, dtype=float)))
    order = np.argsort(raw, kind="mergesort")
    ordered = raw[order]
    thresholds = alpha * np.arange(1, ordered.size + 1, dtype=float) / (ordered.size * harmonic_number)
    passing = np.flatnonzero(ordered <= thresholds)
    reject_sorted = np.zeros(ordered.size, dtype=bool)
    if passing.size:
        reject_sorted[: passing[-1] + 1] = True
    adjusted_sorted = _step_up_adjusted_pvalues(ordered, scale=harmonic_number)
    return _unsort(reject_sorted, order), _unsort(adjusted_sorted, order)


def storey_qvalue(
    pvalues: NDArray | list[float] | tuple[float, ...],
    lambda_: float = 0.5,
) -> NDArray[np.float64]:
    """Storey (2002) q-value estimator.

    This is not an exact rejection rule on its own. It estimates the minimum FDR
    level at which each hypothesis would be called significant and is therefore a
    useful ranking proxy alongside BH/BHY.
    """

    if not 0.0 <= lambda_ < 1.0:
        raise ValueError("lambda_ must lie in [0, 1).")
    raw = _validate_pvalues(pvalues)
    if raw.size == 0:
        return raw

    denom = max(1.0 - lambda_, np.finfo(float).eps)
    pi0 = float(min(1.0, np.mean(raw > lambda_) / denom))
    order = np.argsort(raw, kind="mergesort")
    ordered = raw[order]
    q_sorted = _step_up_adjusted_pvalues(ordered, scale=pi0)
    return _unsort(q_sorted, order)


def romano_wolf_stepwise(
    test_statistics: NDArray | list[float] | tuple[float, ...],
    null_distribution: NDArray | list[list[float]],
    alpha: float = 0.05,
    method: Literal["studentized", "raw"] = "studentized",
) -> NDArray[np.bool_]:
    """Romano-Wolf (2005) step-down multiple-testing under dependence.

    The implementation assumes `null_distribution` contains bootstrap draws of
    the joint test-statistic vector under the null. The procedure is step-down:
    hypotheses are ordered from strongest to weakest and testing stops at the
    first non-rejection.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1).")
    observed = _as_1d_float_array(test_statistics)
    if observed.size == 0:
        return np.zeros(0, dtype=bool)

    bootstrap = np.asarray(null_distribution, dtype=float)
    if bootstrap.ndim != 2:
        raise ValueError("null_distribution must be a 2-D bootstrap array.")
    if bootstrap.shape[1] != observed.size:
        raise ValueError("null_distribution must have one column per hypothesis.")
    if bootstrap.shape[0] == 0 or not np.all(np.isfinite(bootstrap)):
        raise ValueError("null_distribution must contain finite bootstrap draws.")
    if method not in {"studentized", "raw"}:
        raise ValueError("method must be 'studentized' or 'raw'.")

    if method == "studentized":
        scale = bootstrap.std(axis=0, ddof=1)
        scale = np.where(scale <= 1e-8, 1.0, scale)
    else:
        scale = np.ones_like(observed)

    observed_scaled = np.abs(observed / scale)
    bootstrap_scaled = np.abs(bootstrap / scale)
    order = np.argsort(observed_scaled)[::-1]
    active = order.tolist()
    rejected = np.zeros(observed.size, dtype=bool)

    for hypothesis in order:
        active_idx = np.asarray(active, dtype=int)
        bootstrap_max = np.max(bootstrap_scaled[:, active_idx], axis=1)
        cutoff = float(np.quantile(bootstrap_max, 1.0 - alpha, method="higher"))
        if observed_scaled[hypothesis] > cutoff:
            rejected[hypothesis] = True
            active.remove(int(hypothesis))
            continue
        break
    return rejected
