"""S&P 500 proxy universe loader for the Russell-3000 panel.

This module uses a pragmatic proxy for S&P 500 membership: the top `top_n`
names by `weight_pct` from the Russell-3000/IWV-style holdings file. The
source `universe.csv` does not include real GICS sub-industries, so this
loader synthesizes a `sub_industry` bucket from within-sector weight quartiles.
Those quartile buckets are encoded as ``"{sector}_q{1..4}"`` and are intended
only as a same-sector, similar-size grouping signal for P4's `PairSelector`.
They are not true GICS classifications.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _resolve_panel_path(panel_dir: Path | str) -> Path:
    panel_path = Path(panel_dir)
    return panel_path / "panel_adj_close.parquet" if panel_path.is_dir() else panel_path


def _assign_sector_quartiles(frame: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for sector, sector_df in frame.groupby("sector", sort=False, dropna=False):
        ordered = sector_df.sort_values(["weight_pct", "ticker"], ascending=[False, True]).copy()
        bucket = np.floor(np.arange(len(ordered)) * 4 / max(len(ordered), 1)).astype(int) + 1
        ordered["sub_industry"] = [f"{sector}_q{int(min(value, 4))}" for value in bucket]
        pieces.append(ordered)
    return pd.concat(pieces, ignore_index=True) if pieces else frame.assign(sub_industry=pd.Series(dtype=str))


def load_sp500_panel(
    panel_dir: Path | str,
    universe_csv: Path | str,
    top_n: int = 500,
    min_history_days: int = 120,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel_path = _resolve_panel_path(panel_dir)
    if top_n < 1:
        raise ValueError("top_n must be positive.")
    if min_history_days < 1:
        raise ValueError("min_history_days must be positive.")

    price_panel = pd.read_parquet(panel_path).sort_index()
    price_panel.index = pd.to_datetime(price_panel.index)
    universe = pd.read_csv(universe_csv)
    history = price_panel.notna().sum(axis=0)
    eligible = universe.loc[universe["ticker"].isin(history.index[history >= min_history_days])].copy()
    selected = eligible.sort_values(["weight_pct", "ticker"], ascending=[False, True]).head(top_n).copy()
    selected = _assign_sector_quartiles(selected)

    tickers = selected["ticker"].tolist()
    panel = price_panel.reindex(columns=tickers).copy()
    metadata = pd.DataFrame(
        {
            "ticker": selected["ticker"].to_numpy(),
            "asset_type": "equity",
            "sector": selected["sector"].to_numpy(),
            "sub_industry": selected["sub_industry"].to_numpy(),
            "family": "equity_sp500_proxy",
            "theme_group": "sp500_top500",
            "mapped_sector": selected["sector"].to_numpy(),
        }
    )
    return panel, metadata


def summarize_sp500_universe(metadata: pd.DataFrame) -> dict:
    subindustry_counts = metadata.groupby("sub_industry", dropna=False).size() if not metadata.empty else pd.Series(dtype=int)
    sector_counts = metadata.groupby("sector", dropna=False).size().sort_index()
    return {
        "n_tickers": int(len(metadata)),
        "n_sectors": int(metadata["sector"].nunique(dropna=False)) if not metadata.empty else 0,
        "n_sub_industries": int(metadata["sub_industry"].nunique(dropna=False)) if not metadata.empty else 0,
        "sector_counts": {str(key): int(value) for key, value in sector_counts.items()},
        "tickers_per_sub_industry_min": int(subindustry_counts.min()) if not subindustry_counts.empty else 0,
        "tickers_per_sub_industry_max": int(subindustry_counts.max()) if not subindustry_counts.empty else 0,
    }
