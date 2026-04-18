from __future__ import annotations

import json

import numpy as np
import pandas as pd

from p4.config import load_config
from p4.extended_pipeline import run_regime_aware_stat_arb

from .conftest import write_config


def test_extended_pipeline_smoke_runs_on_synthetic_data(tmp_path, monkeypatch) -> None:
    config = load_config(write_config(tmp_path, overrides={"run_name": "extended_smoke"}))
    dates = pd.bdate_range("2023-01-02", periods=84)
    rng = np.random.default_rng(7)
    base = np.log(100.0) + np.linspace(0.0, 0.03, len(dates))
    spread = 0.18 * np.sin(np.linspace(0.0, 10.0, len(dates))) + rng.normal(scale=0.02, size=len(dates))
    prices = pd.DataFrame(
        {"AAA": np.exp(base + 0.5 * spread), "BBB": np.exp(base - 0.5 * spread)},
        index=dates,
    )
    adv = pd.DataFrame({"AAA": 5_000_000.0, "BBB": 6_000_000.0}, index=dates)
    pair_df = pd.DataFrame(
        [{"candidate_id": "pair_w00_AAA_BBB", "tickers_json": json.dumps(["AAA", "BBB"]), "weights_json": json.dumps([1.0, -1.0])}]
    )
    window = {
        "window_id": 0,
        "formation_dates": dates[:36],
        "validation_dates": dates[36:60],
        "test_dates": dates[60:],
    }

    monkeypatch.setattr("p4.extended_pipeline.load_market_inputs", lambda _cfg: (["AAA", "BBB"], pd.DataFrame(), prices, adv))
    monkeypatch.setattr("p4.extended_pipeline._walkforward_windows", lambda _index, _cfg: [window])
    monkeypatch.setattr("p4.extended_pipeline.PairSelector.select", lambda self, **_kwargs: (pair_df, pd.DataFrame()))

    summary = run_regime_aware_stat_arb(config, tmp_path / "extended_run")

    assert summary["n_pair_candidates"] == 1
    assert len(summary["candidates"]) == 1
    assert (tmp_path / "extended_run" / "regime_aware_stat_arb.json").exists()
