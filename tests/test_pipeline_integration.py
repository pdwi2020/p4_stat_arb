from __future__ import annotations

import json

import pandas as pd

from p4.config import load_config
from p4.pipeline import run

from .conftest import write_config


def test_pipeline_fixture_run_writes_expected_artifacts(tmp_path) -> None:
    config_path = write_config(tmp_path, overrides={"run_name": "integration_fixture"})
    summary = run(config_path)
    config = load_config(config_path)
    run_dir = config.results_path()

    assert summary["n_pair_candidates"] >= 1
    assert summary["n_basket_candidates"] >= 1

    expected_files = [
        "candidate_pairs.csv",
        "candidate_baskets.csv",
        "strategy_metrics.csv",
        "daily_strategy_returns.csv",
        "portfolio_returns.csv",
        "multiple_testing_report.json",
        "summary.json",
        "portfolio_cumulative_return.png",
        "portfolio_drawdown.png",
        "candidate_deflation.png",
    ]
    for file_name in expected_files:
        assert (run_dir / file_name).exists(), file_name

    strategy_metrics = pd.read_csv(run_dir / "strategy_metrics.csv")
    assert {"candidate_id", "test_net_sharpe", "one_sided_pvalue", "bonferroni_survivor"}.issubset(strategy_metrics.columns)

    report = json.loads((run_dir / "multiple_testing_report.json").read_text())
    assert report["n_tested_strategies"] == len(strategy_metrics)
    assert config.alpha_trace_path("blueprint.json").exists()
    assert config.alpha_trace_path("ambiguity_log.md").exists()
