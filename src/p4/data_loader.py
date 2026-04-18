"""Market data ingestion, caching, and fixture loading for P4."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from p4.config import P4Config, load_config


ETF_METADATA = {
    "SPY": {"family": "broad_market", "theme_group": "broad_market", "mapped_sector": None, "sub_industry": "US Large Cap Blend"},
    "QQQ": {"family": "broad_market", "theme_group": "broad_market", "mapped_sector": "Information Technology", "sub_industry": "US Large Cap Growth"},
    "IWM": {"family": "broad_market", "theme_group": "broad_market", "mapped_sector": None, "sub_industry": "US Small Cap Blend"},
    "DIA": {"family": "broad_market", "theme_group": "broad_market", "mapped_sector": None, "sub_industry": "US Large Cap Value"},
    "XLK": {"family": "sector_information_technology", "theme_group": "sector", "mapped_sector": "Information Technology", "sub_industry": "Technology Select Sector"},
    "XLF": {"family": "sector_financials", "theme_group": "sector", "mapped_sector": "Financials", "sub_industry": "Financial Select Sector"},
    "XLE": {"family": "sector_energy", "theme_group": "sector", "mapped_sector": "Energy", "sub_industry": "Energy Select Sector"},
    "XLU": {"family": "sector_utilities", "theme_group": "sector", "mapped_sector": "Utilities", "sub_industry": "Utilities Select Sector"},
    "XLP": {"family": "sector_consumer", "theme_group": "sector", "mapped_sector": "Consumer Staples", "sub_industry": "Consumer Staples Select Sector"},
    "XLY": {"family": "sector_consumer", "theme_group": "sector", "mapped_sector": "Consumer Discretionary", "sub_industry": "Consumer Discretionary Select Sector"},
    "XLI": {"family": "sector_industrials", "theme_group": "sector", "mapped_sector": "Industrials", "sub_industry": "Industrial Select Sector"},
    "XLV": {"family": "sector_health_care", "theme_group": "sector", "mapped_sector": "Health Care", "sub_industry": "Health Care Select Sector"},
    "XLB": {"family": "sector_materials", "theme_group": "sector", "mapped_sector": "Materials", "sub_industry": "Materials Select Sector"},
    "XLC": {"family": "sector_communication_services", "theme_group": "sector", "mapped_sector": "Communication Services", "sub_industry": "Communication Services Select Sector"},
    "VNQ": {"family": "sector_real_estate", "theme_group": "sector", "mapped_sector": "Real Estate", "sub_industry": "US Real Estate"},
    "XBI": {"family": "sector_health_care", "theme_group": "sector", "mapped_sector": "Health Care", "sub_industry": "Biotechnology"},
    "SMH": {"family": "sector_information_technology", "theme_group": "sector", "mapped_sector": "Information Technology", "sub_industry": "Semiconductors"},
    "KRE": {"family": "sector_financials", "theme_group": "sector", "mapped_sector": "Financials", "sub_industry": "Regional Banks"},
    "HYG": {"family": "fixed_income", "theme_group": "fixed_income", "mapped_sector": None, "sub_industry": "US High Yield Credit"},
    "TLT": {"family": "fixed_income", "theme_group": "fixed_income", "mapped_sector": None, "sub_industry": "US Long Treasury"},
}

REQUIRED_CACHE_FILES = ("universe.parquet", "metadata.parquet", "prices.parquet", "volume.parquet", "adv_30d.parquet")


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def _chunked(items: Iterable[str], size: int) -> Iterable[list[str]]:
    batch: list[str] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _normalise_ticker(symbol: str) -> str:
    return str(symbol).strip().upper().replace(".", "-")


def _coerce_history(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        panel = raw.stack(level=1, future_stack=True).rename_axis(index=["date", "ticker"])
        panel.columns = [str(col).strip().lower().replace(" ", "_") for col in panel.columns]
        return panel.sort_index()
    frame = raw.copy()
    frame.columns = [str(col).strip().lower().replace(" ", "_") for col in frame.columns]
    frame["ticker"] = tickers[0]
    return frame.reset_index().rename(columns={"Date": "date"}).set_index(["date", "ticker"]).sort_index()


def _latest_snapshot_for_batch(tickers: list[str], period: str) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        tickers=tickers,
        period=period,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    panel = _coerce_history(raw, tickers)
    if panel.empty:
        return pd.DataFrame(columns=["ticker", "price", "adtv_usd"])

    close_col = "adj_close" if "adj_close" in panel.columns else "close"
    if close_col not in panel.columns or "volume" not in panel.columns:
        return pd.DataFrame(columns=["ticker", "price", "adtv_usd"])

    panel["adtv_usd"] = (
        (panel[close_col] * panel["volume"])
        .groupby(level="ticker")
        .rolling(window=20, min_periods=5)
        .mean()
        .droplevel(0)
    )
    latest = panel.reset_index().sort_values("date").groupby("ticker").tail(1)
    return latest[["ticker", close_col, "adtv_usd"]].rename(columns={close_col: "price"})


def fetch_sp500_constituents() -> pd.DataFrame:
    response = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers={"User-Agent": "Mozilla/5.0 (compatible; Codex P4 downloader/1.0)"},
        timeout=30,
    )
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    frame = tables[0].rename(
        columns={
            "Symbol": "ticker",
            "Security": "name",
            "GICS Sector": "sector",
            "GICS Sub-Industry": "sub_industry",
        }
    )
    frame["ticker"] = frame["ticker"].map(_normalise_ticker)
    frame["asset_type"] = "equity"
    frame["family"] = frame["sub_industry"].astype(str).str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True)
    frame["theme_group"] = frame["sector"].astype(str).str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True)
    frame["mapped_sector"] = frame["sector"]
    return frame[["ticker", "name", "asset_type", "sector", "sub_industry", "family", "theme_group", "mapped_sector"]]


def build_etf_metadata(config: P4Config) -> pd.DataFrame:
    rows = []
    for ticker in config.market.etf_universe:
        profile = ETF_METADATA.get(ticker, {})
        rows.append(
            {
                "ticker": ticker,
                "name": ticker,
                "asset_type": "etf",
                "sector": profile.get("mapped_sector") or "ETF",
                "sub_industry": profile.get("sub_industry", "ETF"),
                "family": profile.get("family", "etf"),
                "theme_group": profile.get("theme_group", "etf"),
                "mapped_sector": profile.get("mapped_sector"),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_universe(config: P4Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    universe_path = config.paths.cache_dir / "universe.parquet"
    metadata_path = config.paths.cache_dir / "metadata.parquet"
    if universe_path.exists() and metadata_path.exists():
        return pd.read_parquet(universe_path), pd.read_parquet(metadata_path)

    equity_metadata = fetch_sp500_constituents()
    snapshot_frames = []
    for batch in _chunked(equity_metadata["ticker"].tolist(), config.market.price_batch_size):
        snapshot = _latest_snapshot_for_batch(batch, config.market.snapshot_period)
        if not snapshot.empty:
            snapshot_frames.append(snapshot)
    if not snapshot_frames:
        raise RuntimeError("Failed to fetch current S&P 500 snapshots from yfinance.")

    equity_snapshot = pd.concat(snapshot_frames, ignore_index=True).drop_duplicates(subset=["ticker"], keep="last")
    eligible_equities = equity_metadata.merge(equity_snapshot, on="ticker", how="inner")
    eligible_equities = eligible_equities.loc[
        (eligible_equities["price"] >= config.selection.min_price)
        & (eligible_equities["adtv_usd"] >= config.selection.min_adtv_usd)
    ].copy()
    selected_equities = (
        eligible_equities.sort_values("adtv_usd", ascending=False)
        .head(config.selection.max_equities)
        .reset_index(drop=True)
    )

    etf_metadata = build_etf_metadata(config)
    etf_snapshot = _latest_snapshot_for_batch(config.market.etf_universe, config.market.snapshot_period)
    selected_etfs = etf_metadata.merge(etf_snapshot, on="ticker", how="left")
    selected_etfs = selected_etfs.loc[selected_etfs["price"].fillna(config.selection.min_price) >= config.selection.min_price].copy()
    selected_etfs["adtv_usd"] = selected_etfs["adtv_usd"].fillna(config.selection.min_adtv_usd)

    metadata = pd.concat([selected_equities, selected_etfs], ignore_index=True)
    metadata = metadata.drop_duplicates(subset=["ticker"], keep="last").reset_index(drop=True)
    universe = metadata[["ticker", "asset_type", "sector", "sub_industry", "family", "theme_group", "mapped_sector", "price", "adtv_usd"]].copy()

    _write_parquet(universe, universe_path)
    _write_parquet(metadata, metadata_path)
    return universe, metadata


def download_market_history(metadata: pd.DataFrame, config: P4Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices_path = config.paths.cache_dir / "prices.parquet"
    volume_path = config.paths.cache_dir / "volume.parquet"
    adv_path = config.paths.cache_dir / "adv_30d.parquet"
    if prices_path.exists() and volume_path.exists() and adv_path.exists():
        return pd.read_parquet(prices_path), pd.read_parquet(volume_path), pd.read_parquet(adv_path)

    import yfinance as yf

    price_frames: list[pd.DataFrame] = []
    volume_frames: list[pd.DataFrame] = []
    tickers = metadata["ticker"].tolist()
    for batch in _chunked(tickers, config.market.price_batch_size):
        raw = yf.download(
            tickers=batch,
            start=config.market.data_start,
            end=config.market.data_end,
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=True,
        )
        panel = _coerce_history(raw, batch)
        if panel.empty:
            continue
        close_col = "adj_close" if "adj_close" in panel.columns else "close"
        if close_col not in panel.columns or "volume" not in panel.columns:
            continue
        price_frame = panel[[close_col]].rename(columns={close_col: "price"}).reset_index().pivot(index="date", columns="ticker", values="price")
        volume_frame = panel[["volume"]].reset_index().pivot(index="date", columns="ticker", values="volume")
        price_frames.append(price_frame)
        volume_frames.append(volume_frame)

    if not price_frames:
        raise RuntimeError("Failed to download historical price data from yfinance.")

    prices = pd.concat(price_frames, axis=1).sort_index()
    prices = prices.loc[:, ~prices.columns.duplicated()]
    volume = pd.concat(volume_frames, axis=1).sort_index()
    volume = volume.loc[:, ~volume.columns.duplicated()]
    prices = prices.reindex(columns=tickers)
    volume = volume.reindex(columns=tickers)
    prices = prices.ffill().dropna(axis=1, how="all")
    volume = volume.fillna(0.0)
    adv_30d = (prices * volume).rolling(window=30, min_periods=10).mean()

    _write_parquet(prices, prices_path)
    _write_parquet(volume, volume_path)
    _write_parquet(adv_30d, adv_path)
    return prices, volume, adv_30d


def build_cache(config_path: str | Path | None = None) -> dict[str, object]:
    config = load_config(config_path)
    universe, metadata = bootstrap_universe(config)
    prices, volume, adv_30d = download_market_history(metadata, config)
    manifest = {
        "cache_dir": str(config.paths.cache_dir),
        "n_assets": int(len(metadata)),
        "n_equities": int((metadata["asset_type"] == "equity").sum()),
        "n_etfs": int((metadata["asset_type"] == "etf").sum()),
        "price_shape": tuple(prices.shape),
        "volume_shape": tuple(volume.shape),
        "adv_shape": tuple(adv_30d.shape),
        "date_start": str(prices.index.min().date()) if not prices.empty else None,
        "date_end": str(prices.index.max().date()) if not prices.empty else None,
    }
    print(json.dumps(manifest, indent=2))
    return manifest


def _load_fixture_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.to_datetime(frame.index)
    return frame.sort_index()


def load_market_inputs(config: P4Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if config.data_mode == "fixture":
        metadata = pd.read_csv(config.paths.fixture_dir / "metadata.csv")
        universe = metadata[["ticker", "asset_type", "sector", "sub_industry", "family", "theme_group", "mapped_sector", "price", "adtv_usd"]].copy()
        prices = _load_fixture_frame(config.paths.fixture_dir / "prices.csv")
        volume = _load_fixture_frame(config.paths.fixture_dir / "volume.csv")
        adv_30d = _load_fixture_frame(config.paths.fixture_dir / "adv_30d.csv")
        return universe, metadata, prices, adv_30d

    missing = [name for name in REQUIRED_CACHE_FILES if not (config.paths.cache_dir / name).exists()]
    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing required real-data caches in {config.paths.cache_dir}: {missing_list}. "
            "Run `make download` or use a fixture config for tests."
        )

    universe = pd.read_parquet(config.paths.cache_dir / "universe.parquet")
    metadata = pd.read_parquet(config.paths.cache_dir / "metadata.parquet")
    prices = pd.read_parquet(config.paths.cache_dir / "prices.parquet").sort_index()
    volume = pd.read_parquet(config.paths.cache_dir / "volume.parquet").sort_index()
    adv_30d = pd.read_parquet(config.paths.cache_dir / "adv_30d.parquet").sort_index()
    return universe, metadata, prices, adv_30d


def main(config_path: str | Path | None = None) -> dict[str, object]:
    return build_cache(config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
