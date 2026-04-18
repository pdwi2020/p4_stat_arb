from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from p4.config import load_config
from p4.data_loader import build_cache, load_market_inputs

from .conftest import write_config


def test_load_config_resolves_paths_and_symbols(tmp_path) -> None:
    config_path = write_config(
        tmp_path,
        overrides={"market": {"etf_universe": ["spy", "qqq"]}},
    )
    config = load_config(config_path)
    assert config.paths.fixture_dir.is_absolute()
    assert config.market.etf_universe == ["SPY", "QQQ"]
    assert config.results_path().is_absolute()


def test_load_market_inputs_fixture_returns_expected_frames(tmp_path) -> None:
    config = load_config(write_config(tmp_path))
    universe, metadata, prices, adv_30d = load_market_inputs(config)
    assert {"equity", "etf"}.issubset(set(metadata["asset_type"]))
    assert len(universe) == len(metadata)
    assert not prices.empty
    assert prices.shape == adv_30d.shape


def test_load_market_inputs_real_missing_cache_fails_cleanly(tmp_path) -> None:
    config = load_config(
        write_config(
            tmp_path,
            overrides={
                "data_mode": "real",
                "paths": {"cache_dir": str(tmp_path / "missing_cache")},
            },
        )
    )
    with pytest.raises(FileNotFoundError, match="make download"):
        load_market_inputs(config)


def test_load_market_inputs_real_cache_roundtrip(tmp_path) -> None:
    config = load_config(
        write_config(
            tmp_path,
            overrides={
                "data_mode": "real",
                "paths": {"cache_dir": str(tmp_path / "cache_roundtrip")},
            },
        )
    )
    cache_dir = config.paths.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.bdate_range("2023-01-02", periods=5)
    metadata = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "name": ["Alpha", "Beta"],
            "asset_type": ["equity", "etf"],
            "sector": ["Tech", "ETF"],
            "sub_industry": ["Software", "Broad Market"],
            "family": ["software", "broad_market"],
            "theme_group": ["tech", "broad_market"],
            "mapped_sector": ["Tech", "ETF"],
            "price": [100.0, 200.0],
            "adtv_usd": [50_000_000.0, 80_000_000.0],
        }
    )
    universe = metadata[
        ["ticker", "asset_type", "sector", "sub_industry", "family", "theme_group", "mapped_sector", "price", "adtv_usd"]
    ].copy()
    prices = pd.DataFrame({"AAA": [100, 101, 102, 103, 104], "BBB": [200, 199, 201, 202, 203]}, index=dates)
    volume = pd.DataFrame({"AAA": [1_000_000.0] * 5, "BBB": [2_000_000.0] * 5}, index=dates)
    adv_30d = (prices * volume).rolling(window=2, min_periods=1).mean()

    universe.to_parquet(cache_dir / "universe.parquet")
    metadata.to_parquet(cache_dir / "metadata.parquet")
    prices.to_parquet(cache_dir / "prices.parquet")
    volume.to_parquet(cache_dir / "volume.parquet")
    adv_30d.to_parquet(cache_dir / "adv_30d.parquet")

    loaded_universe, loaded_metadata, loaded_prices, loaded_adv = load_market_inputs(config)
    pd.testing.assert_frame_equal(loaded_universe.reset_index(drop=True), universe.reset_index(drop=True), check_dtype=False)
    pd.testing.assert_frame_equal(loaded_metadata.reset_index(drop=True), metadata.reset_index(drop=True), check_dtype=False)
    pd.testing.assert_frame_equal(loaded_prices, prices, check_dtype=False, check_freq=False)
    pd.testing.assert_frame_equal(loaded_adv, adv_30d, check_dtype=False, check_freq=False)


def test_build_cache_with_mocked_downloads(tmp_path, monkeypatch) -> None:
    config_path = write_config(
        tmp_path,
        overrides={
            "data_mode": "real",
            "paths": {"cache_dir": str(tmp_path / "cache_build")},
            "selection": {"max_equities": 5, "min_adtv_usd": 1_000.0},
            "market": {"etf_universe": ["XLK", "XLF"], "price_batch_size": 10},
        },
    )

    fake_constituents = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "name": ["Alpha", "Beta", "Gamma"],
            "asset_type": ["equity", "equity", "equity"],
            "sector": ["Information Technology", "Financials", "Utilities"],
            "sub_industry": ["Software", "Banks", "Electric Utilities"],
            "family": ["software", "banks", "electric_utilities"],
            "theme_group": ["information_technology", "financials", "utilities"],
            "mapped_sector": ["Information Technology", "Financials", "Utilities"],
        }
    )

    def fake_snapshot(tickers: list[str], _period: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ticker": tickers,
                "price": np.linspace(100.0, 110.0, len(tickers)),
                "adtv_usd": np.linspace(20_000_000.0, 40_000_000.0, len(tickers)),
            }
        )

    class FakeYFinance:
        @staticmethod
        def download(tickers, **_kwargs):
            tickers_list = [tickers] if isinstance(tickers, str) else list(tickers)
            dates = pd.bdate_range("2022-01-03", periods=80)
            adj_close = pd.DataFrame(
                {ticker: np.linspace(90.0 + idx, 120.0 + idx, len(dates)) for idx, ticker in enumerate(tickers_list)},
                index=dates,
            )
            volume = pd.DataFrame(
                {ticker: np.full(len(dates), 1_000_000.0 + 100_000.0 * idx) for idx, ticker in enumerate(tickers_list)},
                index=dates,
            )
            return pd.concat({"Adj Close": adj_close, "Volume": volume}, axis=1)

    monkeypatch.setattr("p4.data_loader.fetch_sp500_constituents", lambda: fake_constituents)
    monkeypatch.setattr("p4.data_loader._latest_snapshot_for_batch", fake_snapshot)
    monkeypatch.setitem(sys.modules, "yfinance", FakeYFinance)

    manifest = build_cache(config_path)
    assert manifest["n_assets"] == 5
    assert manifest["n_equities"] == 3
    assert manifest["n_etfs"] == 2
    assert tuple(manifest["price_shape"])[1] == 5
