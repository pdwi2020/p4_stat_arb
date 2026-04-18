"""Capacity curves for pair strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_panel(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{name} must be a non-empty DataFrame.")
    if frame.columns.empty:
        raise ValueError(f"{name} must contain at least one column.")
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().all().all():
        raise ValueError(f"{name} must contain numeric values.")
    return numeric.sort_index()


def pair_capacity(
    trade_signals: pd.DataFrame,
    prices: pd.DataFrame,
    adv_series: pd.DataFrame,
    slippage_model: str = "sqrt",
    slippage_coef: float = 0.1,
) -> pd.DataFrame:
    """Estimate per-pair capacity curves across a deterministic AUM grid.

    The inputs are pair-level panels aligned on dates. `prices` should be a
    tradable mark or P&L proxy for each pair, while `adv_series` should contain
    pair-level dollar ADV, usually the weaker leg's ADV.
    """

    if slippage_model not in {"sqrt", "linear"}:
        raise ValueError("slippage_model must be 'sqrt' or 'linear'.")
    if slippage_coef < 0.0:
        raise ValueError("slippage_coef must be non-negative.")

    signals = _validate_panel("trade_signals", trade_signals)
    marks = _validate_panel("prices", prices)
    adv = _validate_panel("adv_series", adv_series)

    if set(signals.columns) != set(marks.columns) or set(signals.columns) != set(adv.columns):
        raise ValueError("trade_signals, prices, and adv_series must share identical columns.")

    common_index = signals.index.intersection(marks.index).intersection(adv.index)
    if common_index.empty:
        raise ValueError("Inputs must share at least one common date.")

    signals = signals.loc[common_index, signals.columns]
    marks = marks.loc[common_index, signals.columns]
    adv = adv.loc[common_index, signals.columns]

    returns = marks.pct_change().fillna(0.0)
    lagged_signals = signals.shift(1).fillna(0.0)
    gross_returns = lagged_signals * returns
    turnover = signals.diff().abs().fillna(signals.abs())

    median_adv = float(np.nanmedian(adv.to_numpy(dtype=float)))
    if not np.isfinite(median_adv) or median_adv <= 0.0:
        median_adv = 1_000_000.0
    aum_grid = median_adv * np.geomspace(0.01, 25.0, 16)

    records: list[dict[str, float | str]] = []
    for column in signals.columns:
        gross = gross_returns[column].to_numpy(dtype=float)
        turnover_series = turnover[column].to_numpy(dtype=float)
        adv_values = adv[column].to_numpy(dtype=float)
        valid_adv = np.where(np.isfinite(adv_values) & (adv_values > 0.0), adv_values, np.nan)
        fill_value = float(np.nanmedian(valid_adv)) if np.isfinite(np.nanmedian(valid_adv)) else median_adv
        valid_adv = np.where(np.isfinite(valid_adv), valid_adv, fill_value)

        for aum in aum_grid:
            ratio = (aum * turnover_series) / np.clip(valid_adv, 1e-6, None)
            if slippage_model == "sqrt":
                impact_cost = slippage_coef * np.sqrt(np.clip(ratio, 0.0, None))
            else:
                impact_cost = slippage_coef * np.clip(ratio, 0.0, None)
            net = gross - impact_cost
            records.append(
                {
                    "pair_id": str(column),
                    "aum": float(aum),
                    "mean_gross_return": float(np.nanmean(gross)),
                    "mean_net_return": float(np.nanmean(net)),
                    "mean_impact_cost": float(np.nanmean(impact_cost)),
                    "total_net_pnl": float(np.nansum(net) * aum),
                }
            )

    result = pd.DataFrame.from_records(records)
    return result.sort_values(["pair_id", "aum"]).reset_index(drop=True)
