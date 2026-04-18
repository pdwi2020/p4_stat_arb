"""PCA eigenportfolio statistical arbitrage following Avellaneda-Lee (2010).

This module implements the correlation-based PCA factor decomposition and
residual mean-reversion workflow described in Avellaneda and Lee, "Statistical
arbitrage in the US equities market" (Quantitative Finance, 2010). The factor
model follows the common/idiosyncratic return split in equation (10), models
the residual state with an Ornstein-Uhlenbeck process as in equation (12), and
uses the dimensionless residual score from equation (15) for signal generation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from p4.backtest import BacktestResult, backtest_candidate, build_portfolio_returns
from p4.config import BacktestConfig, SignalConfig
from p4.ou_estimator import OUEstimator
from p4.signal import ZScoreSignal

OUParams: TypeAlias = dict[str, float]

EIGENPORTFOLIO_COLUMNS = [
    "candidate_id",
    "window_id",
    "strategy_type",
    "family_key",
    "relation_type",
    "target_ticker",
    "tickers_json",
    "weights_json",
    "mu",
    "stationary_sigma",
    "kappa",
    "sigma",
    "half_life",
    "pca_components",
    "explained_variance_ratio",
    "residual_volatility",
    "validation_net_sharpe",
    "validation_mean_return",
    "validation_max_drawdown",
    "validation_avg_turnover",
    "validation_trade_count",
    "validation_win_rate",
    "formation_start",
    "formation_end",
    "validation_start",
    "validation_end",
    "test_start",
    "test_end",
    "test_net_sharpe",
    "test_mean_return",
    "test_max_drawdown",
    "test_avg_turnover",
    "test_trade_count",
    "test_win_rate",
]


def _to_json(values: list[str] | list[float]) -> str:
    return json.dumps(values)


def _signal_config(strategy: "EigenportfolioStrategy") -> SignalConfig:
    return SignalConfig(entry_z=strategy.z_entry, exit_z=strategy.z_exit, stop_z=strategy.z_stop)


def _pseudo_prices(returns: pd.DataFrame) -> pd.DataFrame:
    clean = pd.DataFrame(returns, dtype=float).sort_index().fillna(0.0)
    return (1.0 + clean).cumprod()


@dataclass(slots=True)
class BacktestResults:
    candidate_eigenportfolios: pd.DataFrame
    daily_strategy_returns: pd.DataFrame
    portfolio_returns: pd.DataFrame
    n_windows: int

    @property
    def strategy_metrics(self) -> pd.DataFrame:
        return self.candidate_eigenportfolios


class PCAFactorModel(BaseModel):
    """Correlation-PCA factor model for standardized stock returns."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    n_components: int = Field(default=15, ge=1)
    standardize: bool = True
    lookback_days: int | None = Field(default=252, ge=30)

    _tickers: list[str] = PrivateAttr(default_factory=list)
    _means: pd.Series = PrivateAttr(default_factory=lambda: pd.Series(dtype=float))
    _scales: pd.Series = PrivateAttr(default_factory=lambda: pd.Series(dtype=float))
    _loadings: pd.DataFrame = PrivateAttr(default_factory=pd.DataFrame)
    _explained_variance: pd.Series = PrivateAttr(default_factory=lambda: pd.Series(dtype=float))
    _explained_variance_ratio: pd.Series = PrivateAttr(default_factory=lambda: pd.Series(dtype=float))
    _projection: pd.DataFrame = PrivateAttr(default_factory=pd.DataFrame)
    _weight_matrix: pd.DataFrame = PrivateAttr(default_factory=pd.DataFrame)

    def _require_fit(self) -> None:
        if not self._tickers:
            raise ValueError("PCAFactorModel must be fit before use.")

    def _prepare_fit_panel(self, returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        frame = pd.DataFrame(returns, dtype=float).sort_index()
        if self.lookback_days is not None:
            frame = frame.tail(self.lookback_days)
        frame = frame.dropna(axis=0, how="any").dropna(axis=1, how="any")
        if frame.shape[0] < 30 or frame.shape[1] < 2:
            raise ValueError("Not enough fully observed returns to fit PCA factor model.")

        means = frame.mean(axis=0)
        raw_scales = frame.std(axis=0, ddof=1)
        valid_columns = raw_scales[raw_scales > 1e-12].index.tolist()
        frame = frame[valid_columns]
        means = means.loc[valid_columns]
        scales = raw_scales.loc[valid_columns]
        if frame.shape[1] < 2:
            raise ValueError("PCA factor model requires at least two non-degenerate assets.")

        if self.standardize:
            normalized = (frame - means) / scales
        else:
            normalized = frame - means
            scales = pd.Series(1.0, index=frame.columns, dtype=float)
        return frame, normalized, means.astype(float), scales.astype(float)

    def _prepare_inference_panel(self, returns: pd.DataFrame) -> pd.DataFrame:
        self._require_fit()
        frame = pd.DataFrame(returns, dtype=float).sort_index().reindex(columns=self._tickers)
        clean = frame.dropna(axis=0, how="any")
        if clean.empty:
            return pd.DataFrame(columns=self._tickers, dtype=float)
        if self.standardize:
            return (clean - self._means) / self._scales
        return clean - self._means

    @property
    def tickers(self) -> list[str]:
        self._require_fit()
        return list(self._tickers)

    @property
    def loadings(self) -> pd.DataFrame:
        self._require_fit()
        return self._loadings.copy()

    @property
    def explained_variance(self) -> pd.Series:
        self._require_fit()
        return self._explained_variance.copy()

    @property
    def explained_variance_ratio(self) -> pd.Series:
        self._require_fit()
        return self._explained_variance_ratio.copy()

    def fit(self, returns: pd.DataFrame) -> "PCAFactorModel":
        _, normalized, means, scales = self._prepare_fit_panel(returns)
        covariance = np.cov(normalized.to_numpy(dtype=float), rowvar=False, ddof=1)
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.asarray(eigenvalues[order], dtype=float)
        eigenvectors = np.asarray(eigenvectors[:, order], dtype=float)

        positive = eigenvalues > 1e-10
        rank = int(positive.sum())
        n_components = min(self.n_components, rank)
        if n_components < 1:
            raise ValueError("Correlation matrix is rank deficient; no principal components remain.")

        tickers = normalized.columns.tolist()
        component_labels = [f"PC{idx + 1}" for idx in range(n_components)]
        loadings = pd.DataFrame(eigenvectors[:, :n_components], index=tickers, columns=component_labels, dtype=float)
        explained = pd.Series(eigenvalues[:n_components], index=component_labels, dtype=float, name="explained_variance")
        total_variance = float(np.clip(eigenvalues, 0.0, None).sum())
        explained_ratio = explained / total_variance if total_variance > 0 else explained * 0.0

        projection = loadings @ loadings.T
        annihilator = np.eye(len(tickers), dtype=float) - projection.to_numpy(dtype=float)
        scale_ratio = scales.to_numpy(dtype=float)[:, None] / scales.to_numpy(dtype=float)[None, :]
        weight_matrix = pd.DataFrame(annihilator * scale_ratio, index=tickers, columns=tickers, dtype=float)

        self._tickers = tickers
        self._means = means.loc[tickers]
        self._scales = scales.loc[tickers]
        self._loadings = loadings
        self._explained_variance = explained
        self._explained_variance_ratio = explained_ratio.rename("explained_variance_ratio")
        self._projection = projection
        self._weight_matrix = weight_matrix
        return self

    def transform(self, returns: pd.DataFrame) -> pd.DataFrame:
        normalized = self._prepare_inference_panel(returns)
        if normalized.empty:
            return pd.DataFrame(columns=self._loadings.columns, dtype=float)
        factor_returns = normalized @ self._loadings
        factor_returns.columns = self._loadings.columns
        return factor_returns

    def residuals(self, returns: pd.DataFrame) -> pd.DataFrame:
        normalized = self._prepare_inference_panel(returns)
        if normalized.empty:
            return pd.DataFrame(columns=self._tickers, dtype=float)
        factor_returns = normalized @ self._loadings
        reconstructed = factor_returns @ self._loadings.T
        residual_standardized = normalized - reconstructed
        if self.standardize:
            return residual_standardized.mul(self._scales, axis=1)
        return residual_standardized

    def candidate_weights(self, ticker: str, *, normalize: bool = True, weight_floor: float = 1e-8) -> pd.Series:
        self._require_fit()
        if ticker not in self._weight_matrix.index:
            raise KeyError(f"{ticker} is not available in the fitted PCA universe.")
        weights = self._weight_matrix.loc[ticker].astype(float).copy()
        weights = weights.where(weights.abs() >= weight_floor, 0.0)
        active = weights[weights != 0.0]
        if active.empty:
            raise ValueError(f"No residual basket weights remain for {ticker}.")
        if normalize:
            norm = float(active.abs().sum())
            if norm <= 0.0:
                raise ValueError(f"Cannot normalize residual basket for {ticker}.")
            active = active / norm
        return active


class EigenportfolioStrategy(BaseModel):
    """OU-filtered PCA residual strategy on per-stock eigenportfolio residuals."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    pca_model: PCAFactorModel
    min_half_life: float = Field(default=5.0, ge=1.0)
    max_half_life: float = Field(default=30.0, gt=1.0)
    z_entry: float = Field(default=1.5, gt=0.0)
    z_exit: float = Field(default=0.5, ge=0.0)
    z_stop: float = Field(default=3.0, gt=0.0)

    _ou_estimator: OUEstimator = PrivateAttr(default_factory=OUEstimator)

    @model_validator(mode="after")
    def _validate_half_life_bounds(self) -> "EigenportfolioStrategy":
        if self.max_half_life <= self.min_half_life:
            raise ValueError("max_half_life must exceed min_half_life.")
        return self

    @property
    def ou_estimator(self) -> OUEstimator:
        return self._ou_estimator

    def fit_residual_ou(self, residuals: pd.DataFrame) -> dict[str, OUParams]:
        params: dict[str, OUParams] = {}
        panel = pd.DataFrame(residuals, dtype=float).sort_index()
        for ticker in panel.columns:
            residual = panel[ticker].dropna()
            if len(residual) < 30 or float(residual.std(ddof=1)) <= 1e-12:
                continue
            try:
                params[ticker] = self._ou_estimator.fit(residual.cumsum())
            except Exception:
                continue
        return params

    def filter_candidates(self, ou_params_dict: dict[str, OUParams]) -> list[str]:
        candidates = []
        for ticker, params in ou_params_dict.items():
            half_life = float(params.get("half_life", np.nan))
            stationary_sigma = float(params.get("stationary_sigma", np.nan))
            if not np.isfinite(half_life) or not np.isfinite(stationary_sigma):
                continue
            if stationary_sigma <= 0.0:
                continue
            if self.min_half_life <= half_life <= self.max_half_life:
                candidates.append(str(ticker))
        return sorted(candidates)

    def generate_signals(
        self,
        residuals_ou_params: dict[str, OUParams],
        current_z_score: pd.DataFrame | pd.Series,
    ) -> pd.DataFrame | pd.Series:
        signal_engine = ZScoreSignal(entry_z=self.z_entry, exit_z=self.z_exit, stop_z=self.z_stop)
        candidates = self.filter_candidates(residuals_ou_params)

        if isinstance(current_z_score, pd.DataFrame):
            signals = {
                ticker: signal_engine.generate_positions(pd.Series(current_z_score[ticker], dtype=float))
                for ticker in candidates
                if ticker in current_z_score.columns
            }
            return pd.DataFrame(signals, index=current_z_score.index, dtype=float)

        zscore = pd.Series(current_z_score, dtype=float)
        signals = {
            ticker: float(signal_engine.generate_positions(pd.Series([zscore[ticker]], dtype=float)).iloc[-1])
            for ticker in candidates
            if ticker in zscore.index
        }
        return pd.Series(signals, name="position", dtype=float)


class EigenportfolioBacktest:
    """Walk-forward backtest for PCA eigenportfolio residual strategies."""

    def __init__(
        self,
        pca_model: PCAFactorModel,
        strategy: EigenportfolioStrategy,
        cost_model: BacktestConfig,
        capacity_model: BacktestConfig | None = None,
    ) -> None:
        self.pca_model = pca_model
        self.strategy = strategy
        self.cost_model = cost_model
        self.capacity_model = capacity_model or cost_model

    def _window_rows(
        self,
        *,
        window_id: int,
        formation_returns: pd.DataFrame,
        validation_prices: pd.DataFrame,
        test_prices: pd.DataFrame,
        test_adv: pd.DataFrame,
    ) -> tuple[list[dict[str, object]], list[pd.DataFrame], pd.DataFrame]:
        model = self.pca_model.model_copy(deep=True)
        model.fit(formation_returns)

        residuals = model.residuals(formation_returns)
        residual_ou = self.strategy.fit_residual_ou(residuals)
        candidate_tickers = self.strategy.filter_candidates(residual_ou)
        if not candidate_tickers:
            return [], [], pd.DataFrame()

        signal_config = _signal_config(self.strategy)
        strategy_results: dict[str, BacktestResult] = {}
        daily_frames: list[pd.DataFrame] = []
        rows: list[dict[str, object]] = []

        formation_prices = _pseudo_prices(formation_returns[model.tickers])
        explained_ratio = float(model.explained_variance_ratio.sum())

        for ticker in candidate_tickers:
            try:
                weights = model.candidate_weights(ticker)
            except Exception:
                continue

            basket_prices = formation_prices[weights.index]
            spread = pd.Series(
                np.log(basket_prices).to_numpy(dtype=float) @ weights.to_numpy(dtype=float),
                index=basket_prices.index,
                name="spread",
                dtype=float,
            )
            try:
                spread_ou = self.strategy.ou_estimator.fit(spread)
            except Exception:
                continue

            row: dict[str, object] = {
                "candidate_id": f"eigenportfolio_w{window_id:02d}_{ticker}",
                "window_id": window_id,
                "strategy_type": "eigenportfolio",
                "family_key": "pca_factor_model",
                "relation_type": "eigenportfolio_residual",
                "target_ticker": ticker,
                "tickers_json": _to_json([str(name) for name in weights.index.tolist()]),
                "weights_json": _to_json([float(value) for value in weights.tolist()]),
                "mu": float(spread_ou["mu"]),
                "stationary_sigma": float(spread_ou["stationary_sigma"]),
                "kappa": float(spread_ou["kappa"]),
                "sigma": float(spread_ou["sigma"]),
                "half_life": float(spread_ou["half_life"]),
                "pca_components": int(model.loadings.shape[1]),
                "explained_variance_ratio": explained_ratio,
                "residual_volatility": float(residuals[ticker].std(ddof=1)),
                "formation_start": pd.Timestamp(formation_returns.index[0]),
                "formation_end": pd.Timestamp(formation_returns.index[-1]),
                "validation_start": pd.Timestamp(validation_prices.index[0]),
                "validation_end": pd.Timestamp(validation_prices.index[-1]),
                "test_start": pd.Timestamp(test_prices.index[0]),
                "test_end": pd.Timestamp(test_prices.index[-1]),
            }

            try:
                validation_result = backtest_candidate(row, validation_prices[weights.index], signal_config, self.cost_model)
                test_result = backtest_candidate(row, test_prices[weights.index], signal_config, self.cost_model)
            except Exception:
                continue

            row["validation_net_sharpe"] = float(validation_result.metrics["net_sharpe"])
            row["validation_mean_return"] = float(validation_result.metrics["mean_net_return"])
            row["validation_max_drawdown"] = float(validation_result.metrics["max_drawdown"])
            row["validation_avg_turnover"] = float(validation_result.metrics["avg_turnover"])
            row["validation_trade_count"] = int(validation_result.metrics["trade_count"])
            row["validation_win_rate"] = float(validation_result.metrics["win_rate"])

            row["test_net_sharpe"] = float(test_result.metrics["net_sharpe"])
            row["test_mean_return"] = float(test_result.metrics["mean_net_return"])
            row["test_max_drawdown"] = float(test_result.metrics["max_drawdown"])
            row["test_avg_turnover"] = float(test_result.metrics["avg_turnover"])
            row["test_trade_count"] = int(test_result.metrics["trade_count"])
            row["test_win_rate"] = float(test_result.metrics["win_rate"])

            strategy_results[str(row["candidate_id"])] = test_result
            daily = test_result.daily.copy().reset_index(drop=True)
            daily["window_id"] = window_id
            daily["strategy_type"] = "eigenportfolio"
            daily["target_ticker"] = ticker
            daily_frames.append(daily)
            rows.append(row)

        validated_ids = [str(row["candidate_id"]) for row in rows if float(row["validation_net_sharpe"]) > 0.0]
        portfolio = build_portfolio_returns(strategy_results, validated_ids, test_adv, self.capacity_model)
        if not portfolio.empty:
            portfolio["window_id"] = window_id

        return rows, daily_frames, portfolio

    def walk_forward(
        self,
        returns: pd.DataFrame,
        formation_window: int = 252,
        validation_window: int = 126,
        test_window: int = 126,
        *,
        step_window: int | None = None,
        adv_30d: pd.DataFrame | None = None,
    ) -> BacktestResults:
        panel = pd.DataFrame(returns, dtype=float).sort_index()
        if panel.shape[0] < formation_window + validation_window + test_window:
            raise ValueError("Not enough observations for eigenportfolio walk-forward backtest.")

        step = step_window or validation_window
        pseudo_prices = _pseudo_prices(panel)
        if adv_30d is None:
            adv_panel = pd.DataFrame(1e18, index=panel.index, columns=panel.columns, dtype=float)
        else:
            adv_panel = pd.DataFrame(adv_30d, dtype=float).reindex(index=panel.index, columns=panel.columns).fillna(1e18)

        candidate_rows: list[dict[str, object]] = []
        strategy_daily_frames: list[pd.DataFrame] = []
        portfolio_frames: list[pd.DataFrame] = []

        n_windows = 0
        start = 0
        window_id = 0
        while start + formation_window + validation_window + test_window <= len(panel.index):
            formation_slice = panel.iloc[start : start + formation_window]
            validation_slice = panel.iloc[start + formation_window : start + formation_window + validation_window]
            test_slice = panel.iloc[
                start + formation_window + validation_window : start + formation_window + validation_window + test_window
            ]
            full_slice = panel.iloc[start : start + formation_window + validation_window + test_window]

            eligible = full_slice.columns[full_slice.notna().all(axis=0)]
            formation_std = formation_slice[eligible].std(ddof=1)
            eligible = formation_std[formation_std > 1e-12].index.tolist()
            if len(eligible) >= 2:
                rows, daily_frames, portfolio = self._window_rows(
                    window_id=window_id,
                    formation_returns=formation_slice[eligible],
                    validation_prices=pseudo_prices.loc[validation_slice.index, eligible],
                    test_prices=pseudo_prices.loc[test_slice.index, eligible],
                    test_adv=adv_panel.loc[test_slice.index, eligible],
                )
                candidate_rows.extend(rows)
                strategy_daily_frames.extend(daily_frames)
                if not portfolio.empty:
                    portfolio_frames.append(portfolio)

            n_windows += 1
            start += step
            window_id += 1

        candidate_df = pd.DataFrame(candidate_rows)
        if candidate_df.empty:
            candidate_df = pd.DataFrame(columns=EIGENPORTFOLIO_COLUMNS)
        else:
            candidate_df = candidate_df.sort_values(["window_id", "validation_net_sharpe"], ascending=[True, False]).reset_index(drop=True)

        daily_df = (
            pd.concat(strategy_daily_frames, ignore_index=True).sort_values(["date", "candidate_id"])
            if strategy_daily_frames
            else pd.DataFrame()
        )
        portfolio_df = (
            pd.concat(portfolio_frames, ignore_index=True).sort_values("date")
            if portfolio_frames
            else pd.DataFrame(columns=["date", "net_return", "scale_factor", "n_validated", "cumulative_return", "drawdown"])
        )

        return BacktestResults(
            candidate_eigenportfolios=candidate_df,
            daily_strategy_returns=daily_df,
            portfolio_returns=portfolio_df,
            n_windows=n_windows,
        )
