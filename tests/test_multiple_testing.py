from __future__ import annotations

import numpy as np
import pandas as pd

from p4.multiple_testing import bonferroni_threshold, hansen_spa_test, one_sided_mean_pvalues, white_reality_check


def test_bonferroni_threshold_is_alpha_over_tests() -> None:
    assert bonferroni_threshold(0.05, 10) == 0.005


def test_multiple_testing_reports_identify_strong_winner() -> None:
    rng = np.random.default_rng(31)
    returns = pd.DataFrame(
        {
            "winner": 0.004 + rng.normal(scale=0.0015, size=120),
            "flat": rng.normal(scale=0.0015, size=120),
            "loser": -0.002 + rng.normal(scale=0.0015, size=120),
        }
    )

    pvalues = one_sided_mean_pvalues(returns)
    rc = white_reality_check(returns, n_bootstrap=120, block_size=5, random_seed=7)
    spa = hansen_spa_test(returns, n_bootstrap=120, block_size=5, alpha=0.10, random_seed=7)

    assert pvalues["winner"] < 0.01
    assert rc["best_strategy"] == "winner"
    assert 0.0 <= rc["pvalue"] <= 1.0
    assert "winner" in spa["survivors"]
    assert 0.0 <= spa["pvalue"] <= 1.0
