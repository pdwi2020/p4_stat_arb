"""Backtesting helpers for P4 spread strategies."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from p4.config import BacktestConfig, SignalConfig
from p4.signal import ZScoreSignal


@dataclass(slots=True)
class BacktestResult:
    daily: pd.DataFrame
    exposures: pd.DataFrame
    metrics: dict[str, float | int]


def _decode_list(value: str | list[str] | list[float]) -> list:
    if isinstance(value, list):
        return value
    return json.loads(value)


def _max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def _sharpe(returns: pd.Series) -> float:
    clean = returns.dropna()
    std = float(clean.std()) if len(clean) > 1 else 0.0
    if std <= 0:
        return 0.0
    return float(clean.mean() / std * np.sqrt(252.0))


def backtest_candidate(
    candidate: pd.Series | dict,
    prices: pd.DataFrame,
    signal_config: SignalConfig,
    backtest_config: BacktestConfig,
) -> BacktestResult:
    candidate_series = pd.Series(candidate)
    tickers = [str(ticker) for ticker in _decode_list(candidate_series["tickers_json"])]
    weights = np.asarray(_decode_list(candidate_series["weights_json"]), dtype=float)
    price_slice = prices[tickers].dropna().copy()
    if price_slice.empty:
        raise ValueError(f"No prices available for candidate {candidate_series['candidate_id']}.")

    log_prices = np.log(price_slice)
    spread = pd.Series(log_prices.to_numpy(dtype=float) @ weights, index=log_prices.index, name="spread")
    signal_engine = ZScoreSignal(
        entry_z=signal_config.entry_z,
        exit_z=signal_config.exit_z,
        stop_z=signal_config.stop_z,
    )
    zscore = signal_engine.compute(
        spread=spread,
        mu=float(candidate_series["mu"]),
        stationary_sigma=float(candidate_series["stationary_sigma"]),
    )
    position = signal_engine.generate_positions(zscore)

    asset_returns = price_slice.pct_change().fillna(0.0)
    normalized_exposures = pd.DataFrame(
        np.outer(position.shift(1).fillna(0.0).to_numpy(dtype=float), weights),
        index=price_slice.index,
        columns=tickers,
    )
    gross_return = (normalized_exposures * asset_returns).sum(axis=1)
    turnover = normalized_exposures.diff().abs().sum(axis=1).fillna(normalized_exposures.abs().sum(axis=1))
    trading_cost = turnover * ((backtest_config.cost_halfspread_bps + backtest_config.cost_slippage_bps) / 10_000.0)
    short_exposure = normalized_exposures.clip(upper=0.0).abs().sum(axis=1)
    borrow_cost = short_exposure * (backtest_config.borrow_cost_annual_bps / 10_000.0) / 252.0
    net_return = gross_return - trading_cost - borrow_cost

    daily = pd.DataFrame(
        {
            "date": price_slice.index,
            "candidate_id": candidate_series["candidate_id"],
            "position": position.to_numpy(dtype=float),
            "zscore": zscore.to_numpy(dtype=float),
            "turnover": turnover.to_numpy(dtype=float),
            "gross_return": gross_return.to_numpy(dtype=float),
            "net_return": net_return.to_numpy(dtype=float),
            "borrow_cost": borrow_cost.to_numpy(dtype=float),
            "short_exposure": short_exposure.to_numpy(dtype=float),
        },
        index=price_slice.index,
    )
    trade_count = int(((position != 0) & (position.shift(1).fillna(0.0) == 0)).sum())
    metrics = {
        "net_sharpe": _sharpe(net_return),
        "mean_net_return": float(net_return.mean()),
        "max_drawdown": _max_drawdown(net_return),
        "avg_turnover": float(turnover.mean()),
        "trade_count": trade_count,
        "win_rate": float((net_return > 0).mean()),
    }
    return BacktestResult(daily=daily, exposures=normalized_exposures, metrics=metrics)


def build_portfolio_returns(
    strategy_results: dict[str, BacktestResult],
    validated_ids: list[str],
    adv_30d: pd.DataFrame,
    backtest_config: BacktestConfig,
) -> pd.DataFrame:
    if not validated_ids:
        return pd.DataFrame(columns=["date", "net_return", "scale_factor", "n_validated", "cumulative_return", "drawdown"])

    all_dates = sorted({date for strategy_id in validated_ids for date in strategy_results[strategy_id].daily.index})
    rows: list[dict[str, float | int | pd.Timestamp]] = []
    capital_per_strategy = backtest_config.portfolio_capital_usd / len(validated_ids)

    for date in all_dates:
        returns = []
        aggregate_exposure: dict[str, float] = {}
        for strategy_id in validated_ids:
            result = strategy_results[strategy_id]
            if date not in result.daily.index:
                continue
            returns.append(float(result.daily.loc[date, "net_return"]))
            if date in result.exposures.index:
                day_exposure = result.exposures.loc[date].abs() * capital_per_strategy
                for ticker, exposure in day_exposure.items():
                    aggregate_exposure[ticker] = aggregate_exposure.get(ticker, 0.0) + float(exposure)

        if not returns:
            continue

        scale_factor = 1.0
        if date in adv_30d.index and aggregate_exposure:
            capacity = adv_30d.loc[date] * backtest_config.capacity_adv_fraction
            ratios = []
            for ticker, exposure in aggregate_exposure.items():
                limit = float(capacity.get(ticker, np.nan))
                if exposure > 0 and np.isfinite(limit) and limit > 0:
                    ratios.append(limit / exposure)
            if ratios:
                scale_factor = float(min(1.0, min(ratios)))

        rows.append(
            {
                "date": pd.Timestamp(date),
                "net_return": float(np.mean(returns) * scale_factor),
                "scale_factor": scale_factor,
                "n_validated": len(validated_ids),
            }
        )

    portfolio = pd.DataFrame(rows).sort_values("date")
    if portfolio.empty:
        return portfolio
    portfolio["cumulative_return"] = (1.0 + portfolio["net_return"]).cumprod() - 1.0
    equity = 1.0 + portfolio["cumulative_return"]
    peak = equity.cummax()
    portfolio["drawdown"] = equity / peak - 1.0
    return portfolio


def run_backtest(
    candidate: pd.Series | dict,
    prices: pd.DataFrame,
    signal_config: SignalConfig,
    backtest_config: BacktestConfig,
) -> BacktestResult:
    return backtest_candidate(candidate=candidate, prices=prices, signal_config=signal_config, backtest_config=backtest_config)
