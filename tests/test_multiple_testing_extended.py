from __future__ import annotations

import numpy as np

from p4.multiple_testing import (
    benjamini_hochberg,
    benjamini_yekutieli,
    romano_wolf_stepwise,
    storey_qvalue,
)


def test_bh_sanity_has_no_rejections_for_high_uniform_pvalues() -> None:
    pvalues = np.linspace(0.10, 0.95, 20)
    reject, adjusted = benjamini_hochberg(pvalues, alpha=0.05)

    assert not reject.any()
    assert np.all(adjusted >= pvalues)


def test_bh_rejects_exactly_k_strong_signals() -> None:
    pvalues = np.array([0.4, 1e-4, 0.8, 1e-6, 0.6, 0.9, 0.5, 1e-5, 0.7, 0.3], dtype=float)
    reject, _ = benjamini_hochberg(pvalues, alpha=0.05)

    assert int(reject.sum()) == 3


def test_bhy_is_more_conservative_than_bh() -> None:
    pvalues = np.array([0.001, 0.004, 0.009, 0.02, 0.2, 0.5, 0.8], dtype=float)
    bh_reject, bh_adj = benjamini_hochberg(pvalues, alpha=0.05)
    bhy_reject, bhy_adj = benjamini_yekutieli(pvalues, alpha=0.05)

    assert int(bhy_reject.sum()) <= int(bh_reject.sum())
    assert np.all(bhy_adj >= bh_adj - 1e-12)


def test_storey_qvalue_is_monotone_in_rank() -> None:
    pvalues = np.array([0.12, 0.01, 0.08, 0.20, 0.03, 0.50, 0.04], dtype=float)
    qvalues = storey_qvalue(pvalues)

    order = np.argsort(pvalues)
    assert np.all(np.diff(qvalues[order]) >= -1e-12)


def test_romano_wolf_familywise_error_rate_is_bounded_on_synthetic_null() -> None:
    rng = np.random.default_rng(7)
    covariance = 0.4 * np.ones((5, 5), dtype=float) + 0.6 * np.eye(5, dtype=float)
    any_rejection = []
    for _ in range(40):
        observed = rng.multivariate_normal(np.zeros(5), covariance)
        null_distribution = rng.multivariate_normal(np.zeros(5), covariance, size=300)
        any_rejection.append(bool(romano_wolf_stepwise(observed, null_distribution, alpha=0.05).any()))

    assert float(np.mean(any_rejection)) <= 0.10


def test_adjusted_pvalues_are_monotone_in_raw_pvalue_order() -> None:
    pvalues = np.array([0.09, 0.04, 0.20, 0.01, 0.03, 0.5], dtype=float)
    _, bh_adj = benjamini_hochberg(pvalues)
    _, bhy_adj = benjamini_yekutieli(pvalues)

    order = np.argsort(pvalues)
    assert np.all(np.diff(bh_adj[order]) >= -1e-12)
    assert np.all(np.diff(bhy_adj[order]) >= -1e-12)


def test_empty_arrays_are_handled() -> None:
    empty = np.array([], dtype=float)
    bh_reject, bh_adj = benjamini_hochberg(empty)
    bhy_reject, bhy_adj = benjamini_yekutieli(empty)
    qvalues = storey_qvalue(empty)
    rejected = romano_wolf_stepwise(empty, np.empty((5, 0), dtype=float))

    assert bh_reject.size == 0 and bh_adj.size == 0
    assert bhy_reject.size == 0 and bhy_adj.size == 0
    assert qvalues.size == 0
    assert rejected.size == 0


def test_all_zero_pvalues_are_handled() -> None:
    pvalues = np.zeros(6, dtype=float)
    bh_reject, bh_adj = benjamini_hochberg(pvalues)
    bhy_reject, bhy_adj = benjamini_yekutieli(pvalues)
    qvalues = storey_qvalue(pvalues)

    assert bh_reject.all()
    assert bhy_reject.all()
    assert np.allclose(bh_adj, 0.0)
    assert np.allclose(bhy_adj, 0.0)
    assert np.allclose(qvalues, 0.0)
