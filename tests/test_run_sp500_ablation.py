from __future__ import annotations

import json

import numpy as np
import pandas as pd

from p4.run_sp500_ablation import compute_pair_spread, evaluate_ou_strategy, run_ablation


def _ou_spread(length: int, seed: int, theta: float = 0.11, sigma: float = 0.04) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.zeros(length, dtype=float)
    for idx in range(1, length):
        values[idx] = (1.0 - theta) * values[idx - 1] + rng.normal(scale=sigma)
    return values


def _write_panel_fixture(tmp_path):
    dates = pd.bdate_range("2024-01-02", periods=250)
    data: dict[str, np.ndarray] = {}
    rows = []
    for sector_idx in range(3):
        sector = f"Sector{sector_idx}"
        base = np.log(90.0 + 5.0 * sector_idx) + np.linspace(0.0, 0.08, len(dates))
        for pair_idx in range(2):
            spread = _ou_spread(len(dates), seed=sector_idx * 10 + pair_idx)
            left = f"S{sector_idx}A{pair_idx}"
            right = f"S{sector_idx}B{pair_idx}"
            data[left] = np.exp(base + 0.5 * spread)
            data[right] = np.exp(base - 0.5 * spread)
        rows.extend(
            {
                "ticker": ticker,
                "name": ticker,
                "sector": sector,
                "weight_pct": 10.0 - len(rows) * 0.1 - idx * 0.01,
                "shares": 1_000_000,
            }
            for idx, ticker in enumerate([f"S{sector_idx}A0", f"S{sector_idx}B0", f"S{sector_idx}A1", f"S{sector_idx}B1"])
        )
    panel = pd.DataFrame(data, index=dates)
    panel_path = tmp_path / "panel_adj_close.parquet"
    universe_path = tmp_path / "universe.csv"
    panel.to_parquet(panel_path)
    pd.DataFrame(rows).to_csv(universe_path, index=False)
    return panel_path, universe_path


def test_run_sp500_ablation_smoke(tmp_path) -> None:
    panel_path, universe_path = _write_panel_fixture(tmp_path)
    out_dir = tmp_path / "results"

    code = run_ablation(panel_path, universe_path, out_dir, top_n=12, max_pairs=4)

    assert code == 0
    for name in ["pair_universe.csv", "pair_candidates.csv", "ou_ablation.csv", "summary.json"]:
        assert (out_dir / name).exists()
    summary = json.loads((out_dir / "summary.json").read_text())
    assert set(summary) == {"static", "kalman", "neural", "regime"}
    ablation = pd.read_csv(out_dir / "ou_ablation.csv")
    assert set(ablation["method"]) == {"static", "kalman", "neural", "regime"}


def test_compute_pair_spread_strategy_sharpe_synthetic() -> None:
    dates = pd.bdate_range("2024-01-02", periods=220)
    spread = _ou_spread(len(dates), seed=123)
    prices = pd.DataFrame({
        "AAA": np.exp(np.log(100.0) + 0.5 * spread),
        "BBB": np.exp(np.log(100.0) - 0.5 * spread),
    }, index=dates)

    pair_spread = compute_pair_spread(prices, [0.5, -0.5])
    sharpe_gross, _, _ = evaluate_ou_strategy(pair_spread, mu=0.0)

    assert sharpe_gross > 0.0


def test_run_handles_missing_neural_or_regime_module_gracefully(tmp_path, monkeypatch) -> None:
    panel_path, universe_path = _write_panel_fixture(tmp_path)
    out_dir = tmp_path / "results"
    monkeypatch.setattr("p4.run_sp500_ablation._fit_neural_ou", lambda spread: (_ for _ in ()).throw(RuntimeError("torch missing")))

    code = run_ablation(panel_path, universe_path, out_dir, top_n=12, max_pairs=4)

    assert code == 0
    ablation = pd.read_csv(out_dir / "ou_ablation.csv")
    assert "neural" not in set(ablation["method"])
    assert {"static", "kalman", "regime"}.issubset(set(ablation["method"]))
