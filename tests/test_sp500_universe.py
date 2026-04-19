from __future__ import annotations

import numpy as np
import pandas as pd

from p4.sp500_universe import load_sp500_panel, summarize_sp500_universe


def _write_universe_fixture(tmp_path, n_tickers: int = 600, n_sectors: int = 4):
    dates = pd.bdate_range("2024-01-02", periods=160)
    tickers = [f"T{i:03d}" for i in range(n_tickers)]
    panel = pd.DataFrame({ticker: 50.0 + np.arange(len(dates)) * 0.05 + idx * 0.01 for idx, ticker in enumerate(tickers)}, index=dates)
    sectors = [f"Sector{idx % n_sectors}" for idx in range(n_tickers)]
    universe = pd.DataFrame({
        "ticker": tickers,
        "name": tickers,
        "sector": sectors,
        "weight_pct": np.linspace(10.0, 0.1, n_tickers),
        "shares": np.linspace(1_000_000, 500_000, n_tickers),
    })
    panel_path = tmp_path / "panel_adj_close.parquet"
    universe_path = tmp_path / "universe.csv"
    panel.to_parquet(panel_path)
    universe.to_csv(universe_path, index=False)
    return panel, universe, panel_path, universe_path


def test_load_sp500_panel_shapes(tmp_path) -> None:
    _, _, panel_path, universe_path = _write_universe_fixture(tmp_path)
    price_panel, metadata = load_sp500_panel(panel_path, universe_path, top_n=500)

    assert price_panel.shape[1] == 500
    assert list(metadata.columns) == ["ticker", "asset_type", "sector", "sub_industry", "family", "theme_group", "mapped_sector"]
    assert metadata["ticker"].nunique() == 500
    assert set(metadata["asset_type"]) == {"equity"}


def test_load_sp500_panel_history_filter(tmp_path) -> None:
    panel, universe, panel_path, universe_path = _write_universe_fixture(tmp_path, n_tickers=20)
    bad_ticker = universe.iloc[0]["ticker"]
    panel.loc[:, bad_ticker] = np.nan
    panel.iloc[:80, 0] = np.linspace(1.0, 2.0, 80)
    panel.to_parquet(panel_path)

    price_panel, metadata = load_sp500_panel(panel_path, universe_path, top_n=10, min_history_days=120)

    assert bad_ticker not in price_panel.columns
    assert bad_ticker not in metadata["ticker"].tolist()
    assert price_panel.shape[1] == 10


def test_subindustry_quartile_assignment(tmp_path) -> None:
    _, _, panel_path, universe_path = _write_universe_fixture(tmp_path, n_tickers=400, n_sectors=4)
    _, metadata = load_sp500_panel(panel_path, universe_path, top_n=400)

    for sector, sector_df in metadata.groupby("sector"):
        counts = sector_df["sub_industry"].value_counts().sort_index()
        assert counts.index.tolist() == [f"{sector}_q1", f"{sector}_q2", f"{sector}_q3", f"{sector}_q4"]
        assert counts.max() - counts.min() <= 2


def test_summarize_returns_expected_shape(tmp_path) -> None:
    _, _, panel_path, universe_path = _write_universe_fixture(tmp_path, n_tickers=120, n_sectors=4)
    _, metadata = load_sp500_panel(panel_path, universe_path, top_n=100)
    summary = summarize_sp500_universe(metadata)

    expected = {"n_tickers", "n_sectors", "n_sub_industries", "sector_counts", "tickers_per_sub_industry_min", "tickers_per_sub_industry_max"}
    assert set(summary) == expected
    assert sum(summary["sector_counts"].values()) == summary["n_tickers"]
