"""Extended regime-aware statistical arbitrage pipeline for P4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from p4.capacity import pair_capacity
from p4.config import P4Config, ensure_run_directories, load_config
from p4.data_loader import load_market_inputs
from p4.multiple_testing import benjamini_hochberg, benjamini_yekutieli, one_sided_mean_pvalues, storey_qvalue
from p4.pair_selector import PairSelector
from p4.pipeline import _walkforward_windows
from p4.regime_switch import fit_pair_regime, regime_filtered_trading_signal


def _resolve_config(config: P4Config | str | Path | None) -> P4Config:
    if isinstance(config, P4Config):
        return config
    return load_config(config)


def _decode_candidate(row: pd.Series) -> tuple[list[str], np.ndarray]:
    tickers = [str(ticker) for ticker in json.loads(row["tickers_json"])]
    weights = np.asarray(json.loads(row["weights_json"]), dtype=float)
    return tickers, weights


def _build_spread(price_panel: pd.DataFrame, tickers: list[str], weights: np.ndarray) -> pd.Series:
    slice_ = price_panel[tickers].dropna().copy()
    if slice_.empty:
        raise ValueError("No overlapping prices for the candidate.")
    log_prices = np.log(slice_)
    return pd.Series(log_prices.to_numpy(dtype=float) @ weights, index=log_prices.index, name="spread")


def _sharpe(returns: pd.Series) -> float:
    clean = pd.Series(returns, dtype=float).dropna()
    std = float(clean.std(ddof=1)) if len(clean) > 1 else 0.0
    if std <= 0.0:
        return 0.0
    return float(clean.mean() / std * np.sqrt(252.0))


def _max_drawdown(returns: pd.Series) -> float:
    clean = pd.Series(returns, dtype=float).fillna(0.0)
    if clean.empty:
        return 0.0
    equity = (1.0 + clean).cumprod()
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def _regime_backtest(
    row: pd.Series,
    *,
    formation_prices: pd.DataFrame,
    validation_prices: pd.DataFrame,
    test_prices: pd.DataFrame,
    test_adv: pd.DataFrame,
    config: P4Config,
) -> tuple[dict[str, object], pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    tickers, weights = _decode_candidate(row)
    train_prices = pd.concat([formation_prices[tickers], validation_prices[tickers]], axis=0).dropna()
    test_price_slice = test_prices[tickers].dropna()
    test_adv_slice = test_adv[tickers].reindex(test_price_slice.index).ffill().bfill()
    if train_prices.empty or test_price_slice.empty or test_adv_slice.empty:
        raise ValueError("Insufficient data for regime-aware backtest.")

    train_spread = _build_spread(train_prices, tickers, weights)
    test_spread = _build_spread(test_price_slice, tickers, weights)
    regime_fit = fit_pair_regime(train_spread, n_regimes=2)
    hmm = regime_fit["model"]
    predicted_states = pd.Series(hmm.predict_states(test_spread.to_numpy(dtype=float)), index=test_spread.index, name="regime", dtype=int)
    trade_signal = regime_filtered_trading_signal(
        spread=test_spread,
        regime=predicted_states,
        active_regime=int(regime_fit["high_reversion_regime_idx"]),
        z_threshold=float(config.signal.entry_z),
    )

    asset_returns = test_price_slice.pct_change().fillna(0.0)
    exposures = pd.DataFrame(
        np.outer(trade_signal.shift(1).fillna(0.0).to_numpy(dtype=float), weights),
        index=test_spread.index,
        columns=tickers,
    )
    gross_return = (exposures * asset_returns.reindex(test_spread.index)).sum(axis=1)
    turnover = exposures.diff().abs().sum(axis=1).fillna(exposures.abs().sum(axis=1))
    trading_cost = turnover * ((config.backtest.cost_halfspread_bps + config.backtest.cost_slippage_bps) / 10_000.0)
    borrow_cost = exposures.clip(upper=0.0).abs().sum(axis=1) * (config.backtest.borrow_cost_annual_bps / 10_000.0) / 252.0
    net_return = gross_return - trading_cost - borrow_cost

    pair_mark = pd.Series(np.exp(test_spread - float(test_spread.iloc[0])), index=test_spread.index, name=str(row["candidate_id"]))
    pair_adv = pd.Series(test_adv_slice.min(axis=1).to_numpy(dtype=float), index=test_spread.index, name=str(row["candidate_id"]))
    daily = pd.DataFrame(
        {
            "date": test_spread.index,
            "candidate_id": str(row["candidate_id"]),
            "net_return": net_return.to_numpy(dtype=float),
            "gross_return": gross_return.to_numpy(dtype=float),
            "turnover": turnover.to_numpy(dtype=float),
            "regime": predicted_states.to_numpy(dtype=int),
            "signal": trade_signal.to_numpy(dtype=float),
        }
    )

    summary = row.to_dict()
    summary.update(
        {
            "regime_means": np.asarray(regime_fit["regime_means"], dtype=float).round(6).tolist(),
            "regime_vols": np.asarray(regime_fit["regime_vols"], dtype=float).round(6).tolist(),
            "transition_matrix": np.asarray(regime_fit["transition_matrix"], dtype=float).round(6).tolist(),
            "half_life_per_regime": np.asarray(regime_fit["half_life_per_regime"], dtype=float).round(6).tolist(),
            "active_regime": int(regime_fit["high_reversion_regime_idx"]),
            "regime_log_likelihood": float(regime_fit["log_likelihood"]),
            "test_net_sharpe": _sharpe(net_return),
            "test_mean_return": float(net_return.mean()),
            "test_max_drawdown": _max_drawdown(net_return),
            "test_avg_turnover": float(turnover.mean()),
            "test_trade_count": int((trade_signal != 0.0).sum()),
        }
    )
    return summary, daily, trade_signal.rename(str(row["candidate_id"])), pair_mark, pair_adv


def run_regime_aware_stat_arb(
    config: P4Config | str | Path | None,
    output_dir: str | Path | None,
) -> dict[str, object]:
    """Run regime-aware pair selection, FDR filtering, and capacity estimation."""

    resolved_config = _resolve_config(config)
    run_dir = Path(output_dir) if output_dir is not None else ensure_run_directories(resolved_config)
    run_dir.mkdir(parents=True, exist_ok=True)

    universe, metadata, prices, adv_30d = load_market_inputs(resolved_config)
    selector = PairSelector(resolved_config)
    windows = _walkforward_windows(prices.index, resolved_config)
    if not windows:
        raise ValueError("Not enough history for the configured walk-forward schedule.")

    candidate_rows: list[dict[str, object]] = []
    daily_frames: list[pd.DataFrame] = []
    signal_map: dict[str, pd.Series] = {}
    mark_map: dict[str, pd.Series] = {}
    adv_map: dict[str, pd.Series] = {}

    for window in windows:
        formation_prices = prices.loc[window["formation_dates"]].copy()
        validation_prices = prices.loc[window["validation_dates"]].copy()
        test_prices = prices.loc[window["test_dates"]].copy()
        test_adv = adv_30d.loc[window["test_dates"]].copy()

        pair_df, _ = selector.select(
            formation_prices=formation_prices,
            validation_prices=validation_prices,
            metadata=metadata,
            window_id=int(window["window_id"]),
        )
        if pair_df.empty:
            continue

        for _, row in pair_df.iterrows():
            try:
                summary, daily, signal, pair_mark, pair_adv = _regime_backtest(
                    row,
                    formation_prices=formation_prices,
                    validation_prices=validation_prices,
                    test_prices=test_prices,
                    test_adv=test_adv,
                    config=resolved_config,
                )
            except Exception:
                continue
            summary["window_id"] = int(window["window_id"])
            candidate_rows.append(summary)
            daily_frames.append(daily)
            signal_map[str(row["candidate_id"])] = signal
            mark_map[str(row["candidate_id"])] = pair_mark
            adv_map[str(row["candidate_id"])] = pair_adv

    candidate_df = pd.DataFrame(candidate_rows)
    daily_df = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame(columns=["date", "candidate_id", "net_return"])
    if candidate_df.empty or daily_df.empty:
        payload = {
            "run_name": resolved_config.run_name,
            "n_windows": len(windows),
            "n_pair_candidates": 0,
            "bh_survivors": [],
            "bhy_survivors": [],
            "capacity": [],
        }
        output_path = run_dir / "regime_aware_stat_arb.json"
        output_path.write_text(json.dumps(payload, indent=2, default=str))
        return payload

    return_wide = daily_df.pivot_table(index="date", columns="candidate_id", values="net_return", fill_value=0.0)
    pvalues = one_sided_mean_pvalues(return_wide)
    bh_reject, bh_adj = benjamini_hochberg(pvalues.to_numpy(dtype=float), alpha=resolved_config.multiple_testing.alpha)
    bhy_reject, bhy_adj = benjamini_yekutieli(pvalues.to_numpy(dtype=float), alpha=resolved_config.multiple_testing.alpha)
    qvalues = storey_qvalue(pvalues.to_numpy(dtype=float))

    pvalue_lookup = pvalues.to_dict()
    bh_lookup = dict(zip(pvalues.index, bh_reject.tolist(), strict=True))
    bhy_lookup = dict(zip(pvalues.index, bhy_reject.tolist(), strict=True))
    bh_adj_lookup = dict(zip(pvalues.index, bh_adj.tolist(), strict=True))
    bhy_adj_lookup = dict(zip(pvalues.index, bhy_adj.tolist(), strict=True))
    qvalue_lookup = dict(zip(pvalues.index, qvalues.tolist(), strict=True))

    candidate_df["one_sided_pvalue"] = candidate_df["candidate_id"].map(pvalue_lookup)
    candidate_df["bh_fdr_survivor"] = candidate_df["candidate_id"].map(bh_lookup).fillna(False)
    candidate_df["bhy_fdr_survivor"] = candidate_df["candidate_id"].map(bhy_lookup).fillna(False)
    candidate_df["bh_adjusted_pvalue"] = candidate_df["candidate_id"].map(bh_adj_lookup)
    candidate_df["bhy_adjusted_pvalue"] = candidate_df["candidate_id"].map(bhy_adj_lookup)
    candidate_df["storey_qvalue"] = candidate_df["candidate_id"].map(qvalue_lookup)

    survivor_ids = sorted(
        candidate_df.loc[candidate_df["bh_fdr_survivor"] | candidate_df["bhy_fdr_survivor"], "candidate_id"].astype(str).tolist()
    )
    capacity_df = pd.DataFrame()
    if survivor_ids:
        signal_frame = pd.concat([signal_map[candidate_id] for candidate_id in survivor_ids], axis=1).sort_index()
        mark_frame = pd.concat([mark_map[candidate_id] for candidate_id in survivor_ids], axis=1).sort_index()
        adv_frame = pd.concat([adv_map[candidate_id] for candidate_id in survivor_ids], axis=1).sort_index()
        capacity_df = pair_capacity(signal_frame, mark_frame, adv_frame)

    payload = {
        "run_name": resolved_config.run_name,
        "results_dir": str(run_dir),
        "n_assets": int(len(universe)),
        "n_windows": len(windows),
        "n_pair_candidates": int(len(candidate_df)),
        "alpha": float(resolved_config.multiple_testing.alpha),
        "bh_survivors": sorted(candidate_df.loc[candidate_df["bh_fdr_survivor"], "candidate_id"].astype(str).tolist()),
        "bhy_survivors": sorted(candidate_df.loc[candidate_df["bhy_fdr_survivor"], "candidate_id"].astype(str).tolist()),
        "candidates": candidate_df.sort_values(["window_id", "candidate_id"]).to_dict(orient="records"),
        "capacity": capacity_df.to_dict(orient="records"),
    }
    output_path = run_dir / "regime_aware_stat_arb.json"
    output_path.write_text(json.dumps(payload, indent=2, default=str))
    return payload


def main(config_path: str | Path | None = None) -> dict[str, object]:
    return run_regime_aware_stat_arb(config_path, output_dir=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
