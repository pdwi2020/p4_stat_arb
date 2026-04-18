"""Multiple-testing correction utilities for P4."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
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
