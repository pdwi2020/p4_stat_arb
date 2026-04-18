"""End-to-end statistical arbitrage pipeline for P4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from p4.backtest import backtest_candidate, build_portfolio_returns
from p4.config import P4Config, ensure_run_directories, load_config
from p4.data_loader import load_market_inputs
from p4.eigenportfolio import EIGENPORTFOLIO_COLUMNS, EigenportfolioBacktest, EigenportfolioStrategy, PCAFactorModel
from p4.multiple_testing import bonferroni_threshold, hansen_spa_test, one_sided_mean_pvalues, white_reality_check

PAIR_COLUMNS = [
    "candidate_id",
    "window_id",
    "strategy_type",
    "family_key",
    "relation_type",
    "tickers_json",
    "weights_json",
    "mu",
    "stationary_sigma",
    "kappa",
    "sigma",
    "half_life",
    "correlation",
    "beta",
    "coint_pvalue",
    "adf_pvalue",
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

BASKET_COLUMNS = [
    "candidate_id",
    "window_id",
    "strategy_type",
    "family_key",
    "relation_type",
    "tickers_json",
    "weights_json",
    "mu",
    "stationary_sigma",
    "kappa",
    "sigma",
    "half_life",
    "source_pair_id",
    "trace_stat",
    "critical_value",
    "eigenvalue",
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

PORTFOLIO_COLUMNS = ["date", "net_return", "scale_factor", "n_validated", "cumulative_return", "drawdown"]


def _write_json(payload: dict[str, object], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str))


def _max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def _sharpe(returns: pd.Series) -> float:
    clean = returns.dropna()
    std = float(clean.std()) if len(clean) > 1 else 0.0
    if std <= 0:
        return 0.0
    return float(clean.mean() / std * np.sqrt(252.0))


def _walkforward_windows(index: pd.Index, config: P4Config) -> list[dict[str, object]]:
    dates = pd.Index(pd.to_datetime(index)).sort_values().unique()
    formation = config.walkforward.formation_days
    validation = config.walkforward.validation_days
    test = config.walkforward.test_days
    step = config.walkforward.step_days
    windows = []
    start = 0
    window_id = 0
    while start + formation + validation + test <= len(dates):
        formation_dates = dates[start : start + formation]
        validation_dates = dates[start + formation : start + formation + validation]
        test_dates = dates[start + formation + validation : start + formation + validation + test]
        windows.append(
            {
                "window_id": window_id,
                "formation_dates": formation_dates,
                "validation_dates": validation_dates,
                "test_dates": test_dates,
            }
        )
        start += step
        window_id += 1
    return windows


def _plot_cumulative(portfolio_returns: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(pd.to_datetime(portfolio_returns["date"]), portfolio_returns["cumulative_return"], color="#1f77b4")
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("P4 Portfolio Cumulative Return")
    ax.set_ylabel("Cumulative Return")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_drawdown(portfolio_returns: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(pd.to_datetime(portfolio_returns["date"]), portfolio_returns["drawdown"], color="#d62728")
    ax.set_title("P4 Portfolio Drawdown")
    ax.set_ylabel("Drawdown")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_deflation(summary: dict[str, object], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Raw Positive", "Bonferroni", "SPA"]
    values = [
        int(summary["raw_positive_strategies"]),
        int(summary["bonferroni_survivors"]),
        int(summary["spa_survivors"]),
    ]
    ax.bar(labels, values, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_title("P4 Candidate Deflation")
    ax.set_ylabel("Strategy Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _write_alpha_trace(config: P4Config, summary: dict[str, object]) -> None:
    blueprint = {
        "project": "P4 Statistical Arbitrage Mixed Universe",
        "run_name": config.run_name,
        "strategy_type": config.strategy_type,
        "mixed_universe": "Current S&P 500 constituents plus a fixed ETF list from public sources.",
        "basket_scope": "3-asset baskets only, expanded from prefiltered pair candidates.",
        "multiple_testing": "Bonferroni, White Reality Check, and Hansen SPA on out-of-sample daily net returns.",
        "summary": summary,
    }
    _write_json(blueprint, config.alpha_trace_path("blueprint.json"))
    _write_json(blueprint, config.paths.alpha_trace_dir / "blueprint.json")
    ambiguity_log = """# P4 Ambiguity Log

- Survivorship: v1 uses current S&P 500 constituents, not historical constituent membership.
- Universe mixing: equity-ETF pairs are limited to sector-matched relationships; broad-market ETFs are mainly used in ETF-only families.
- Basket scope: baskets are restricted to 3 assets and generated from prefiltered pairs rather than exhaustive triplet enumeration.
- Multiple-testing bootstrap: the family-wise bootstrap uses a common block length derived from the median `2 x half_life`, clipped to `[5, 60]`.
"""
    config.alpha_trace_path("ambiguity_log.md").write_text(ambiguity_log)
    (config.paths.alpha_trace_dir / "ambiguity_log.md").write_text(ambiguity_log)


def _apply_multiple_testing(
    config: P4Config,
    strategy_metrics: pd.DataFrame,
    daily_strategy_returns: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object], list[str], dict[str, object], dict[str, object]]:
    return_wide = daily_strategy_returns.pivot_table(index="date", columns="candidate_id", values="net_return", fill_value=0.0)
    pvalues = one_sided_mean_pvalues(return_wide)
    bonf_threshold = bonferroni_threshold(config.multiple_testing.alpha, len(pvalues))
    bonf_survivors = sorted(pvalues.index[pvalues <= bonf_threshold].tolist())

    block_lengths = np.clip((strategy_metrics["half_life"].astype(float) * 2.0).round().astype(int), 5, 60)
    block_size = int(block_lengths.median()) if not block_lengths.empty else 5
    white_report = white_reality_check(
        return_wide,
        n_bootstrap=config.multiple_testing.n_bootstrap,
        block_size=block_size,
        random_seed=config.multiple_testing.random_seed,
    )
    spa_report = hansen_spa_test(
        return_wide,
        n_bootstrap=config.multiple_testing.n_bootstrap,
        block_size=block_size,
        alpha=config.multiple_testing.alpha,
        random_seed=config.multiple_testing.random_seed,
    )

    enriched = strategy_metrics.copy()
    enriched["one_sided_pvalue"] = enriched["candidate_id"].map(pvalues.to_dict())
    enriched["bonferroni_survivor"] = enriched["candidate_id"].isin(bonf_survivors)
    enriched["spa_survivor"] = enriched["candidate_id"].isin(spa_report["survivors"])
    enriched["white_rc_best_strategy"] = enriched["candidate_id"] == white_report.get("best_strategy")

    multiple_testing_report = {
        "alpha": config.multiple_testing.alpha,
        "n_tested_strategies": int(len(enriched)),
        "bonferroni_threshold": bonf_threshold,
        "bonferroni_survivors": bonf_survivors,
        "white_reality_check": white_report,
        "hansen_spa": spa_report,
    }
    return enriched, multiple_testing_report, bonf_survivors, white_report, spa_report


def _finalize_run(
    *,
    config: P4Config,
    run_dir: Path,
    universe: pd.DataFrame,
    metadata: pd.DataFrame,
    n_windows: int,
    candidate_frames: dict[str, pd.DataFrame],
    strategy_metrics: pd.DataFrame,
    daily_strategy_returns: pd.DataFrame,
    portfolio_returns: pd.DataFrame,
    summary_counts: dict[str, object],
) -> dict[str, object]:
    if strategy_metrics.empty or daily_strategy_returns.empty:
        raise ValueError("No viable strategies were produced by the current configuration.")

    strategy_metrics, multiple_testing_report, bonf_survivors, white_report, spa_report = _apply_multiple_testing(
        config,
        strategy_metrics,
        daily_strategy_returns,
    )
    _write_json(multiple_testing_report, run_dir / "multiple_testing_report.json")

    for file_name, frame in candidate_frames.items():
        frame.to_csv(run_dir / file_name, index=False)

    strategy_metrics.to_csv(run_dir / "strategy_metrics.csv", index=False)
    daily_strategy_returns.to_csv(run_dir / "daily_strategy_returns.csv", index=False)
    portfolio_returns.to_csv(run_dir / "portfolio_returns.csv", index=False)

    if not portfolio_returns.empty:
        _plot_cumulative(portfolio_returns, run_dir / "portfolio_cumulative_return.png")
        _plot_drawdown(portfolio_returns, run_dir / "portfolio_drawdown.png")

    summary: dict[str, object] = {
        "run_name": config.run_name,
        "data_mode": config.data_mode,
        "n_assets": int(len(universe)),
        "n_equities": int((metadata["asset_type"] == "equity").sum()),
        "n_etfs": int((metadata["asset_type"] == "etf").sum()),
        "n_windows": int(n_windows),
    }
    summary.update(summary_counts)
    summary.update(
        {
            "raw_positive_strategies": int((strategy_metrics["test_net_sharpe"] > 0).sum()),
            "bonferroni_survivors": int(len(bonf_survivors)),
            "spa_survivors": int(len(spa_report["survivors"])),
            "white_rc_pvalue": float(white_report["pvalue"]),
            "portfolio_net_sharpe": _sharpe(portfolio_returns["net_return"]) if not portfolio_returns.empty else 0.0,
            "portfolio_max_drawdown": _max_drawdown(portfolio_returns["net_return"]) if not portfolio_returns.empty else 0.0,
            "portfolio_total_return": float(portfolio_returns["cumulative_return"].iloc[-1]) if not portfolio_returns.empty else 0.0,
            "results_dir": str(run_dir),
        }
    )
    _write_json(summary, run_dir / "summary.json")
    _plot_deflation(summary, run_dir / "candidate_deflation.png")
    _write_alpha_trace(config, summary)

    status_parts = [f"mode={config.data_mode}"]
    if "n_pair_candidates" in summary:
        status_parts.append(f"pair_candidates={summary['n_pair_candidates']}")
    if "n_basket_candidates" in summary:
        status_parts.append(f"basket_candidates={summary['n_basket_candidates']}")
    if "n_eigenportfolio_candidates" in summary:
        status_parts.append(f"eigen_candidates={summary['n_eigenportfolio_candidates']}")
    status_parts.append(f"bonferroni={summary['bonferroni_survivors']}")
    status_parts.append(f"spa={summary['spa_survivors']}")
    print("P4 complete | " + " | ".join(status_parts))
    return summary


def _run_cointegration(
    *,
    config: P4Config,
    run_dir: Path,
    universe: pd.DataFrame,
    metadata: pd.DataFrame,
    prices: pd.DataFrame,
    adv_30d: pd.DataFrame,
) -> dict[str, object]:
    from p4.pair_selector import PairSelector

    tickers = metadata["ticker"].tolist()
    prices = prices.reindex(columns=tickers).sort_index()
    adv_30d = adv_30d.reindex(columns=tickers).sort_index()
    selector = PairSelector(config)
    windows = _walkforward_windows(prices.index, config)
    if not windows:
        raise ValueError("Not enough history for the configured walk-forward schedule.")

    pair_rows: list[dict[str, object]] = []
    basket_rows: list[dict[str, object]] = []
    strategy_daily_frames: list[pd.DataFrame] = []
    portfolio_frames: list[pd.DataFrame] = []

    for window in windows:
        window_id = int(window["window_id"])
        formation_prices = prices.loc[window["formation_dates"]].copy()
        validation_prices = prices.loc[window["validation_dates"]].copy()
        test_prices = prices.loc[window["test_dates"]].copy()
        test_adv = adv_30d.loc[window["test_dates"]].copy()

        candidate_pairs, candidate_baskets = selector.select(
            formation_prices=formation_prices,
            validation_prices=validation_prices,
            metadata=metadata,
            window_id=window_id,
        )
        if candidate_pairs.empty and candidate_baskets.empty:
            continue

        combined = pd.concat([candidate_pairs, candidate_baskets], ignore_index=True)
        combined["formation_start"] = pd.Timestamp(window["formation_dates"][0])
        combined["formation_end"] = pd.Timestamp(window["formation_dates"][-1])
        combined["validation_start"] = pd.Timestamp(window["validation_dates"][0])
        combined["validation_end"] = pd.Timestamp(window["validation_dates"][-1])
        combined["test_start"] = pd.Timestamp(window["test_dates"][0])
        combined["test_end"] = pd.Timestamp(window["test_dates"][-1])

        strategy_results = {}
        window_rows = []
        for _, row in combined.iterrows():
            result = backtest_candidate(row, test_prices, config.signal, config.backtest)
            strategy_results[str(row["candidate_id"])] = result
            daily = result.daily.copy().reset_index(drop=True)
            daily["window_id"] = window_id
            daily["strategy_type"] = row["strategy_type"]
            strategy_daily_frames.append(daily)

            row_dict = row.to_dict()
            row_dict["test_net_sharpe"] = float(result.metrics["net_sharpe"])
            row_dict["test_mean_return"] = float(result.metrics["mean_net_return"])
            row_dict["test_max_drawdown"] = float(result.metrics["max_drawdown"])
            row_dict["test_avg_turnover"] = float(result.metrics["avg_turnover"])
            row_dict["test_trade_count"] = int(result.metrics["trade_count"])
            row_dict["test_win_rate"] = float(result.metrics["win_rate"])
            window_rows.append(row_dict)

        validated_ids = [row["candidate_id"] for row in window_rows if float(row["validation_net_sharpe"]) > 0.0]
        portfolio = build_portfolio_returns(strategy_results, validated_ids, test_adv, config.backtest)
        if not portfolio.empty:
            portfolio["window_id"] = window_id
            portfolio_frames.append(portfolio)

        pair_rows.extend([row for row in window_rows if row["strategy_type"] == "pair"])
        basket_rows.extend([row for row in window_rows if row["strategy_type"] == "basket"])

    candidate_pairs = pd.DataFrame(pair_rows)
    if not candidate_pairs.empty:
        candidate_pairs = candidate_pairs.sort_values(["window_id", "validation_net_sharpe"], ascending=[True, False]).reset_index(drop=True)
    else:
        candidate_pairs = pd.DataFrame(columns=PAIR_COLUMNS)

    candidate_baskets = pd.DataFrame(basket_rows)
    if not candidate_baskets.empty:
        candidate_baskets = candidate_baskets.sort_values(["window_id", "validation_net_sharpe"], ascending=[True, False]).reset_index(drop=True)
    else:
        candidate_baskets = pd.DataFrame(columns=BASKET_COLUMNS)

    strategy_metrics = (
        pd.concat([candidate_pairs, candidate_baskets], ignore_index=True)
        if (not candidate_pairs.empty or not candidate_baskets.empty)
        else pd.DataFrame()
    )
    daily_strategy_returns = (
        pd.concat(strategy_daily_frames, ignore_index=True).sort_values(["date", "candidate_id"])
        if strategy_daily_frames
        else pd.DataFrame()
    )
    portfolio_returns = (
        pd.concat(portfolio_frames, ignore_index=True).sort_values("date")
        if portfolio_frames
        else pd.DataFrame(columns=PORTFOLIO_COLUMNS)
    )

    return _finalize_run(
        config=config,
        run_dir=run_dir,
        universe=universe,
        metadata=metadata,
        n_windows=len(windows),
        candidate_frames={
            "candidate_pairs.csv": candidate_pairs,
            "candidate_baskets.csv": candidate_baskets,
        },
        strategy_metrics=strategy_metrics,
        daily_strategy_returns=daily_strategy_returns,
        portfolio_returns=portfolio_returns,
        summary_counts={
            "n_pair_candidates": int(len(candidate_pairs)),
            "n_basket_candidates": int(len(candidate_baskets)),
        },
    )


def _run_eigenportfolio(
    *,
    config: P4Config,
    run_dir: Path,
    universe: pd.DataFrame,
    metadata: pd.DataFrame,
    prices: pd.DataFrame,
    adv_30d: pd.DataFrame,
) -> dict[str, object]:
    tickers = metadata["ticker"].tolist()
    prices = prices.reindex(columns=tickers).sort_index()
    adv_30d = adv_30d.reindex(columns=tickers).sort_index()
    returns = prices.pct_change().dropna(how="all")
    if returns.empty:
        raise ValueError("Not enough price history to compute eigenportfolio returns.")

    pca_model = PCAFactorModel(
        n_components=config.eigenportfolio.pca_components,
        standardize=True,
        lookback_days=config.eigenportfolio.formation_window_days,
    )
    strategy = EigenportfolioStrategy(
        pca_model=pca_model,
        min_half_life=config.eigenportfolio.min_half_life_days,
        max_half_life=config.eigenportfolio.max_half_life_days,
        z_entry=config.eigenportfolio.z_entry,
        z_exit=config.eigenportfolio.z_exit,
        z_stop=config.eigenportfolio.z_stop,
    )
    backtester = EigenportfolioBacktest(
        pca_model=pca_model,
        strategy=strategy,
        cost_model=config.backtest,
        capacity_model=config.backtest,
    )
    results = backtester.walk_forward(
        returns=returns,
        formation_window=config.eigenportfolio.formation_window_days,
        validation_window=config.walkforward.validation_days,
        test_window=config.walkforward.test_days,
        step_window=config.walkforward.step_days,
        adv_30d=adv_30d.reindex(index=returns.index),
    )

    candidate_df = results.candidate_eigenportfolios
    if candidate_df.empty:
        candidate_df = pd.DataFrame(columns=EIGENPORTFOLIO_COLUMNS)

    return _finalize_run(
        config=config,
        run_dir=run_dir,
        universe=universe,
        metadata=metadata,
        n_windows=results.n_windows,
        candidate_frames={"candidate_eigenportfolios.csv": candidate_df},
        strategy_metrics=candidate_df,
        daily_strategy_returns=results.daily_strategy_returns,
        portfolio_returns=results.portfolio_returns,
        summary_counts={
            "strategy_type": "eigenportfolio",
            "n_eigenportfolio_candidates": int(len(candidate_df)),
        },
    )


def run(config_path: str | Path | None = None) -> dict[str, object]:
    config = load_config(config_path)
    run_dir = ensure_run_directories(config)
    universe, metadata, prices, adv_30d = load_market_inputs(config)

    if config.strategy_type == "eigenportfolio":
        return _run_eigenportfolio(
            config=config,
            run_dir=run_dir,
            universe=universe,
            metadata=metadata,
            prices=prices,
            adv_30d=adv_30d,
        )

    return _run_cointegration(
        config=config,
        run_dir=run_dir,
        universe=universe,
        metadata=metadata,
        prices=prices,
        adv_30d=adv_30d,
    )


def main(config_path: str | Path | None = None) -> dict[str, object]:
    return run(config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
