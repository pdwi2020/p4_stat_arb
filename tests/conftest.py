from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"


def _deep_update(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def write_config(tmp_path: Path, *, overrides: dict | None = None) -> Path:
    payload = {
        "run_name": "test_fixture_run",
        "data_mode": "fixture",
        "paths": {
            "cache_dir": str(tmp_path / "cache"),
            "results_dir": str(tmp_path / "results"),
            "fixture_dir": str(FIXTURE_DIR),
            "alpha_trace_dir": str(tmp_path / "alpha_trace"),
        },
        "market": {
            "data_start": "2020-01-01",
            "data_end": "2025-12-31",
            "etf_universe": ["SPY", "QQQ", "XLK", "XLF"],
            "price_batch_size": 10,
            "snapshot_period": "3mo",
        },
        "selection": {
            "max_equities": 6,
            "min_price": 5.0,
            "min_adtv_usd": 1_000_000.0,
            "min_pair_correlation": 0.05,
            "max_pairs_per_family": 4,
            "max_baskets_per_family": 2,
        },
        "cointegration": {
            "engle_granger_alpha": 0.10,
            "johansen_alpha": 0.10,
        },
        "ou": {
            "half_life_min_days": 5.0,
            "half_life_max_days": 30.0,
        },
        "signal": {
            "entry_z": 1.5,
            "exit_z": 0.25,
            "stop_z": 4.0,
        },
        "backtest": {
            "cost_halfspread_bps": 5.0,
            "cost_slippage_bps": 3.0,
            "borrow_cost_annual_bps": 50.0,
            "capacity_adv_fraction": 0.05,
            "portfolio_capital_usd": 1_000_000.0,
        },
        "walkforward": {
            "formation_days": 180,
            "validation_days": 60,
            "test_days": 60,
            "step_days": 60,
        },
        "multiple_testing": {
            "n_bootstrap": 60,
            "block_length_mode": "ou_half_life",
            "alpha": 0.05,
            "random_seed": 42,
        },
    }
    if overrides:
        payload = _deep_update(payload, deepcopy(overrides))
    config_path = tmp_path / "p4_test_config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return config_path
