"""Candidate generation and validation ranking for P4."""

from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

from p4.backtest import backtest_candidate
from p4.cointegration import engle_granger_test, johansen_test
from p4.config import P4Config
from p4.ou_estimator import OUEstimator


def _to_json(values: list[str] | list[float]) -> str:
    return json.dumps(values)


class PairSelector:
    """Generate, filter, and validation-rank pair and basket candidates."""

    def __init__(self, config: P4Config):
        self.config = config
        self.ou_estimator = OUEstimator()

    def _equity_groups(self, metadata: pd.DataFrame) -> dict[str, list[str]]:
        equities = metadata.loc[metadata["asset_type"] == "equity"].copy()
        groups: dict[str, list[str]] = {}
        sector_fallback: dict[str, list[str]] = {}
        for (_, sub_df) in equities.groupby(["sector", "sub_industry"], dropna=False):
            tickers = sorted(sub_df["ticker"].tolist())
            if len(tickers) >= 2:
                key = f"equity_subindustry::{sub_df['sub_industry'].iloc[0]}"
                groups[key] = tickers
            else:
                sector = str(sub_df["sector"].iloc[0])
                sector_fallback.setdefault(sector, []).extend(tickers)
        for sector, tickers in sector_fallback.items():
            if len(tickers) >= 2:
                groups[f"equity_sector::{sector}"] = sorted(set(tickers))
        return groups

    def _etf_groups(self, metadata: pd.DataFrame) -> dict[str, list[str]]:
        etfs = metadata.loc[metadata["asset_type"] == "etf"].copy()
        groups = {}
        for family, family_df in etfs.groupby("family", dropna=False):
            tickers = sorted(family_df["ticker"].tolist())
            if len(tickers) >= 2:
                groups[f"etf_family::{family}"] = tickers
        return groups

    def _cross_groups(self, metadata: pd.DataFrame) -> dict[str, list[str]]:
        equities = metadata.loc[metadata["asset_type"] == "equity"].copy()
        etfs = metadata.loc[(metadata["asset_type"] == "etf") & metadata["mapped_sector"].notna()].copy()
        groups = {}
        for sector, sector_df in equities.groupby("sector", dropna=False):
            sector_etfs = etfs.loc[etfs["mapped_sector"] == sector, "ticker"].tolist()
            if len(sector_df) >= 1 and len(sector_etfs) >= 1:
                groups[f"cross::{sector}"] = sorted(set(sector_df["ticker"].tolist() + sector_etfs))
        return groups

    def _pair_relation_type(self, family_key: str) -> str:
        if family_key.startswith("equity_subindustry"):
            return "equity_equity_subindustry"
        if family_key.startswith("equity_sector"):
            return "equity_equity_sector"
        if family_key.startswith("etf_family"):
            return "etf_etf"
        return "equity_etf"

    def _rank_pair_candidates(self, tickers: list[str], formation_prices: pd.DataFrame) -> list[tuple[str, str, float]]:
        log_returns = np.log(formation_prices[tickers]).diff().dropna()
        ranked = []
        for left, right in itertools.combinations(tickers, 2):
            pair_data = log_returns[[left, right]].dropna()
            if len(pair_data) < 40:
                continue
            corr = float(pair_data[left].corr(pair_data[right]))
            if abs(corr) < self.config.selection.min_pair_correlation:
                continue
            ranked.append((left, right, corr))
        ranked.sort(key=lambda item: abs(item[2]), reverse=True)
        return ranked

    def _candidate_base_row(
        self,
        *,
        window_id: int,
        strategy_type: str,
        family_key: str,
        relation_type: str,
        tickers: list[str],
        weights: np.ndarray,
        ou_params: dict[str, float],
        extras: dict[str, float | str],
    ) -> dict[str, object]:
        candidate_id = f"{strategy_type}_w{window_id:02d}_{'_'.join(tickers)}"
        row: dict[str, object] = {
            "candidate_id": candidate_id,
            "window_id": int(window_id),
            "strategy_type": strategy_type,
            "family_key": family_key,
            "relation_type": relation_type,
            "tickers_json": _to_json(tickers),
            "weights_json": _to_json([float(weight) for weight in weights]),
            "mu": float(ou_params["mu"]),
            "stationary_sigma": float(ou_params["stationary_sigma"]),
            "kappa": float(ou_params["kappa"]),
            "sigma": float(ou_params["sigma"]),
            "half_life": float(ou_params["half_life"]),
        }
        row.update(extras)
        return row

    def _validate_candidate(self, row: dict[str, object], validation_prices: pd.DataFrame) -> dict[str, object] | None:
        try:
            result = backtest_candidate(row, validation_prices, self.config.signal, self.config.backtest)
        except Exception:
            return None
        row["validation_net_sharpe"] = float(result.metrics["net_sharpe"])
        row["validation_mean_return"] = float(result.metrics["mean_net_return"])
        row["validation_max_drawdown"] = float(result.metrics["max_drawdown"])
        row["validation_avg_turnover"] = float(result.metrics["avg_turnover"])
        row["validation_trade_count"] = int(result.metrics["trade_count"])
        row["validation_win_rate"] = float(result.metrics["win_rate"])
        return row

    def _select_pairs(self, formation_prices: pd.DataFrame, validation_prices: pd.DataFrame, metadata: pd.DataFrame, window_id: int) -> pd.DataFrame:
        pair_rows: list[dict[str, object]] = []
        family_groups = {**self._equity_groups(metadata), **self._etf_groups(metadata), **self._cross_groups(metadata)}

        for family_key, tickers in family_groups.items():
            ranked_pairs = self._rank_pair_candidates(tickers, formation_prices)
            for left, right, corr in ranked_pairs[: self.config.selection.max_pairs_per_family]:
                log_pair = np.log(formation_prices[[left, right]].dropna())
                if len(log_pair) < 60:
                    continue
                try:
                    coint_result = engle_granger_test(log_pair, alpha=self.config.cointegration.engle_granger_alpha)
                    if not bool(coint_result["pass"]):
                        continue
                    ou_params = self.ou_estimator.fit(coint_result["spread"])
                except Exception:
                    continue
                if not (self.config.ou.half_life_min_days <= ou_params["half_life"] <= self.config.ou.half_life_max_days):
                    continue
                row = self._candidate_base_row(
                    window_id=window_id,
                    strategy_type="pair",
                    family_key=family_key,
                    relation_type=self._pair_relation_type(family_key),
                    tickers=[left, right],
                    weights=np.asarray(coint_result["weights"], dtype=float),
                    ou_params=ou_params,
                    extras={
                        "correlation": float(corr),
                        "beta": float(coint_result["beta"]),
                        "coint_pvalue": float(coint_result["coint_pvalue"]),
                        "adf_pvalue": float(coint_result["adf_pvalue"]),
                    },
                )
                row = self._validate_candidate(row, validation_prices)
                if row is not None:
                    pair_rows.append(row)

        pair_df = pd.DataFrame(pair_rows)
        if pair_df.empty:
            return pair_df
        return pair_df.sort_values("validation_net_sharpe", ascending=False).reset_index(drop=True)

    def _related_assets_for_basket(self, family_key: str, metadata: pd.DataFrame, pair_tickers: list[str]) -> list[str]:
        tickers = set(pair_tickers)
        if family_key.startswith("equity_"):
            sector = family_key.split("::", 1)[1] if "::" in family_key else None
            if family_key.startswith("equity_subindustry"):
                family_assets = metadata.loc[metadata["family"] == metadata.loc[metadata["ticker"] == pair_tickers[0], "family"].iloc[0], "ticker"].tolist()
            else:
                family_assets = metadata.loc[metadata["sector"] == sector, "ticker"].tolist()
        elif family_key.startswith("etf_family"):
            family = family_key.split("::", 1)[1]
            family_assets = metadata.loc[metadata["family"] == family, "ticker"].tolist()
        else:
            sector = family_key.split("::", 1)[1]
            family_assets = metadata.loc[
                (metadata["sector"] == sector) | ((metadata["asset_type"] == "etf") & (metadata["mapped_sector"] == sector)),
                "ticker",
            ].tolist()
        return sorted([ticker for ticker in family_assets if ticker not in tickers])

    def _select_baskets(
        self,
        formation_prices: pd.DataFrame,
        validation_prices: pd.DataFrame,
        metadata: pd.DataFrame,
        pair_df: pd.DataFrame,
        window_id: int,
    ) -> pd.DataFrame:
        if pair_df.empty:
            return pd.DataFrame()

        basket_rows: list[dict[str, object]] = []
        seen: set[tuple[str, ...]] = set()
        log_returns = np.log(formation_prices).diff().dropna()

        for family_key, family_pairs in pair_df.groupby("family_key", sort=False):
            family_basket_count = 0
            for _, pair_row in family_pairs.iterrows():
                if family_basket_count >= self.config.selection.max_baskets_per_family:
                    break
                pair_tickers = json.loads(pair_row["tickers_json"])
                candidate_thirds = self._related_assets_for_basket(family_key, metadata, pair_tickers)
                third_rank = []
                for ticker in candidate_thirds:
                    corr_values = []
                    for base in pair_tickers:
                        if base in log_returns.columns and ticker in log_returns.columns:
                            corr = log_returns[base].corr(log_returns[ticker])
                            if np.isfinite(corr):
                                corr_values.append(abs(float(corr)))
                    if corr_values:
                        third_rank.append((ticker, float(np.mean(corr_values))))
                third_rank.sort(key=lambda item: item[1], reverse=True)

                for third_ticker, _ in third_rank:
                    if family_basket_count >= self.config.selection.max_baskets_per_family:
                        break
                    tickers = sorted(set(pair_tickers + [third_ticker]))
                    if len(tickers) != 3:
                        continue
                    ticker_key = tuple(tickers)
                    if ticker_key in seen:
                        continue
                    log_basket = np.log(formation_prices[tickers].dropna())
                    if len(log_basket) < 90:
                        continue
                    try:
                        coint_result = johansen_test(log_basket, alpha=self.config.cointegration.johansen_alpha)
                        if not bool(coint_result["pass"]):
                            continue
                        ou_params = self.ou_estimator.fit(coint_result["spread"])
                    except Exception:
                        continue
                    if not (self.config.ou.half_life_min_days <= ou_params["half_life"] <= self.config.ou.half_life_max_days):
                        continue
                    row = self._candidate_base_row(
                        window_id=window_id,
                        strategy_type="basket",
                        family_key=family_key,
                        relation_type="basket",
                        tickers=tickers,
                        weights=np.asarray(coint_result["weights"], dtype=float),
                        ou_params=ou_params,
                        extras={
                            "source_pair_id": pair_row["candidate_id"],
                            "trace_stat": float(coint_result["trace_stat"]),
                            "critical_value": float(coint_result["critical_value"]),
                            "eigenvalue": float(coint_result["eigenvalue"]),
                        },
                    )
                    row = self._validate_candidate(row, validation_prices)
                    if row is not None:
                        basket_rows.append(row)
                        seen.add(ticker_key)
                        family_basket_count += 1

        basket_df = pd.DataFrame(basket_rows)
        if basket_df.empty:
            return basket_df
        return basket_df.sort_values("validation_net_sharpe", ascending=False).reset_index(drop=True)

    def select(
        self,
        *,
        formation_prices: pd.DataFrame,
        validation_prices: pd.DataFrame,
        metadata: pd.DataFrame,
        window_id: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        pair_df = self._select_pairs(formation_prices=formation_prices, validation_prices=validation_prices, metadata=metadata, window_id=window_id)
        basket_df = self._select_baskets(
            formation_prices=formation_prices,
            validation_prices=validation_prices,
            metadata=metadata,
            pair_df=pair_df,
            window_id=window_id,
        )
        return pair_df, basket_df
