from __future__ import annotations

import json

import numpy as np
import pandas as pd

from p4.backtest import BacktestResult, backtest_candidate, build_portfolio_returns
from p4.config import BacktestConfig, load_config
from p4.data_loader import load_market_inputs
from p4.ou_estimator import OUEstimator

from .conftest import write_config


def test_backtest_candidate_applies_costs_and_produces_metrics(tmp_path) -> None:
    config = load_config(write_config(tmp_path))
    _, _, prices, _ = load_market_inputs(config)
    price_slice = prices[["TCHA", "TCHB"]].iloc[:220]
    spread = np.log(price_slice["TCHA"]) - np.log(price_slice["TCHB"])
    ou_params = OUEstimator().fit(spread)
    candidate = {
        "candidate_id": "pair_fixture",
        "tickers_json": json.dumps(["TCHA", "TCHB"]),
        "weights_json": json.dumps([1.0, -1.0]),
        "mu": float(ou_params["mu"]),
        "stationary_sigma": float(ou_params["stationary_sigma"]),
    }

    result = backtest_candidate(candidate, price_slice, config.signal, config.backtest)
    assert not result.daily.empty
    assert result.metrics["avg_turnover"] >= 0.0
    assert result.metrics["trade_count"] >= 0
    assert (result.daily["net_return"] <= result.daily["gross_return"] + 1e-12).all()


def test_build_portfolio_returns_scales_to_adv_capacity() -> None:
    dates = pd.bdate_range("2024-01-02", periods=3)
    result_a = BacktestResult(
        daily=pd.DataFrame({"date": dates, "net_return": [0.010, 0.012, 0.011]}, index=dates),
        exposures=pd.DataFrame({"A": [1.0, 1.0, 1.0], "B": [0.5, 0.5, 0.5]}, index=dates),
        metrics={"net_sharpe": 1.0, "mean_net_return": 0.0, "max_drawdown": 0.0, "avg_turnover": 0.0, "trade_count": 1, "win_rate": 1.0},
    )
    result_b = BacktestResult(
        daily=pd.DataFrame({"date": dates, "net_return": [0.008, 0.009, 0.010]}, index=dates),
        exposures=pd.DataFrame({"A": [0.9, 0.9, 0.9], "C": [0.4, 0.4, 0.4]}, index=dates),
        metrics={"net_sharpe": 1.0, "mean_net_return": 0.0, "max_drawdown": 0.0, "avg_turnover": 0.0, "trade_count": 1, "win_rate": 1.0},
    )
    adv_30d = pd.DataFrame({"A": [100_000.0] * 3, "B": [100_000.0] * 3, "C": [100_000.0] * 3}, index=dates)
    backtest_config = BacktestConfig(
        cost_halfspread_bps=5.0,
        cost_slippage_bps=3.0,
        borrow_cost_annual_bps=50.0,
        capacity_adv_fraction=0.01,
        portfolio_capital_usd=10_000_000.0,
    )

    portfolio = build_portfolio_returns({"s1": result_a, "s2": result_b}, ["s1", "s2"], adv_30d, backtest_config)
    assert not portfolio.empty
    assert (portfolio["scale_factor"] < 1.0).all()
    assert portfolio["net_return"].iloc[0] < (0.010 + 0.008) / 2.0
