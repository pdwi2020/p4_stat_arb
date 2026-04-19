"""Run a 4-way OU ablation on a top-500 Russell-3000 weight proxy universe.

The universe uses the top `top_n` names by `weight_pct` from the
Russell-3000/IWV-style holdings file as a practical S&P 500 proxy. The
underlying loader also synthesizes `sub_industry` buckets from within-sector
weight quartiles so the existing P4 `PairSelector` can group similar-size
names inside a sector, even though the source file does not contain true GICS
sub-industry labels.
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from p4.config import P4Config
from p4.kalman_ou import KalmanOU
from p4.ou_estimator import OUEstimator
from p4.pair_selector import PairSelector
from p4.sp500_universe import load_sp500_panel


METHODS = ("static", "kalman", "neural", "regime")


def compute_pair_spread(prices: pd.DataFrame, weights: Sequence[float]) -> pd.Series:
    clean = np.log(prices).dropna()
    spread = clean.to_numpy(dtype=float) @ np.asarray(weights, dtype=float)
    return pd.Series(spread, index=clean.index, name="spread", dtype=float)


def evaluate_ou_strategy(spread: pd.Series, mu: float) -> tuple[float, float, float]:
    clean = pd.Series(spread, dtype=float).dropna()
    diff = clean.diff().fillna(0.0)
    signal = np.sign(float(mu) - clean)
    pnl = signal.shift(1).fillna(0.0) * diff
    turnover = float(signal.diff().abs().fillna(signal.abs()).mean())
    gross_std = float(pnl.std(ddof=0))
    sharpe_gross = 0.0 if gross_std <= 1e-12 else float(pnl.mean() / gross_std * math.sqrt(252.0))
    sharpe_net = float(sharpe_gross - 0.0005 * turnover * 252.0)
    return sharpe_gross, sharpe_net, turnover


def _johansen_trace_stat(prices: pd.DataFrame) -> float:
    clean = np.log(prices).dropna()
    if len(clean) < 30:
        return float("nan")
    result = coint_johansen(clean.to_numpy(dtype=float), det_order=0, k_ar_diff=1)
    return float(result.lr1[0])


def _fit_neural_ou(spread: pd.Series) -> dict[str, float]:
    module = importlib.import_module("p4.neural_ou")
    result = module.fit_neural_ou(spread)
    return {
        "kappa": float(result.get("kappa", result.get("theta"))),
        "mu": float(result["mu"]),
        "sigma": float(result["sigma"]),
        "half_life": float(result["half_life"]),
    }


def _fit_regime_ou(spread: pd.Series) -> dict[str, float]:
    module = importlib.import_module("p4.regime_switch")
    result = module.fit_pair_regime(spread)
    probs = result["smoothed_probs"].to_numpy(dtype=float)
    weights = probs.mean(axis=0)
    half_lives = np.asarray(result["half_life_per_regime"], dtype=float)
    kappas = np.where(np.isfinite(half_lives) & (half_lives > 0.0), math.log(2.0) / half_lives, 0.0)
    summary_kappa = float(np.average(kappas, weights=weights))
    return {
        "kappa": summary_kappa,
        "mu": float(np.average(np.asarray(result["regime_means"], dtype=float), weights=weights)),
        "sigma": float(np.average(np.asarray(result["regime_vols"], dtype=float), weights=weights)),
        "half_life": float(math.log(2.0) / max(summary_kappa, 1e-8)),
    }


def _ablation_methods() -> dict[str, callable]:
    return {
        "static": lambda spread: OUEstimator().fit(spread),
        "kalman": lambda spread: KalmanOU().fit(spread),
        "neural": _fit_neural_ou,
        "regime": _fit_regime_ou,
    }


def _build_config() -> P4Config:
    return P4Config(run_name="sp500_ablation")


def _screenable_pair_count(metadata: pd.DataFrame) -> int:
    groups: list[list[str]] = []
    sector_fallback: dict[str, list[str]] = {}
    equities = metadata.loc[metadata["asset_type"] == "equity"].copy()
    for (_, sub_df) in equities.groupby(["sector", "sub_industry"], dropna=False):
        tickers = sorted(sub_df["ticker"].tolist())
        if len(tickers) >= 2:
            groups.append(tickers)
        elif tickers:
            sector_fallback.setdefault(str(sub_df["sector"].iloc[0]), []).extend(tickers)
    groups.extend(sorted(set(tickers)) for tickers in sector_fallback.values() if len(set(tickers)) >= 2)
    return int(sum(math.comb(len(group), 2) for group in groups))


def run_ablation(panel_dir: Path | str, universe_csv: Path | str, out_dir: Path | str, top_n: int = 500, max_pairs: int = 200) -> int:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    price_panel, metadata = load_sp500_panel(panel_dir=panel_dir, universe_csv=universe_csv, top_n=top_n)
    if price_panel.empty or metadata.empty:
        raise ValueError("No eligible tickers available for the S&P 500 proxy scan.")

    split_idx = max(60, int(len(price_panel) * 0.7))
    split_idx = min(split_idx, len(price_panel) - 20)
    if split_idx <= 0:
        raise ValueError("Not enough dates to create formation and validation windows.")
    formation = price_panel.iloc[:split_idx]
    validation = price_panel.iloc[split_idx:]

    selector = PairSelector(_build_config())
    pair_df, _ = selector.select(formation_prices=formation, validation_prices=validation, metadata=metadata, window_id=0)
    selected_pairs = pair_df.head(max_pairs).copy() if not pair_df.empty else pd.DataFrame()

    metadata.to_csv(out_path / "pair_universe.csv", index=False)
    candidate_rows: list[dict[str, object]] = []
    ablation_rows: list[dict[str, object]] = []
    fitters = _ablation_methods()

    for pair_row in selected_pairs.to_dict("records"):
        tickers = json.loads(pair_row["tickers_json"])
        weights = json.loads(pair_row["weights_json"])
        pair_id = str(pair_row["candidate_id"])
        prices = price_panel[tickers]
        spread = compute_pair_spread(prices, weights)
        static_fit = OUEstimator().fit(spread)
        candidate_rows.append(
            {
                "pair_id": pair_id,
                "ticker_a": tickers[0],
                "ticker_b": tickers[1],
                "weights": json.dumps([float(value) for value in weights]),
                "johansen_trace_stat": _johansen_trace_stat(prices),
                "kappa_static": float(static_fit["kappa"]),
            }
        )
        for method, fitter in fitters.items():
            try:
                fit = fitter(spread)
                sharpe_gross, sharpe_net, turnover = evaluate_ou_strategy(spread, float(fit["mu"]))
            except Exception as exc:
                print(f"[warn] pair={pair_id} method={method} failed: {exc}", file=sys.stderr)
                continue
            ablation_rows.append(
                {
                    "pair_id": pair_id,
                    "ticker_a": tickers[0],
                    "ticker_b": tickers[1],
                    "method": method,
                    "kappa": float(fit["kappa"]),
                    "mu": float(fit["mu"]),
                    "sigma": float(fit["sigma"]),
                    "half_life": float(fit["half_life"]),
                    "sharpe_gross": sharpe_gross,
                    "sharpe_net": sharpe_net,
                    "turnover": turnover,
                }
            )

    candidate_df = pd.DataFrame(candidate_rows, columns=["pair_id", "ticker_a", "ticker_b", "weights", "johansen_trace_stat", "kappa_static"])
    ablation_df = pd.DataFrame(
        ablation_rows,
        columns=["pair_id", "ticker_a", "ticker_b", "method", "kappa", "mu", "sigma", "half_life", "sharpe_gross", "sharpe_net", "turnover"],
    )
    candidate_df.to_csv(out_path / "pair_candidates.csv", index=False)
    ablation_df.to_csv(out_path / "ou_ablation.csv", index=False)

    summary = {
        method: {
            "median_kappa": float(group["kappa"].median()) if not group.empty else 0.0,
            "mean_sharpe_net": float(group["sharpe_net"].mean()) if not group.empty else 0.0,
            "median_sharpe_net": float(group["sharpe_net"].median()) if not group.empty else 0.0,
            "n_pairs": int(group["pair_id"].nunique()) if not group.empty else 0,
        }
        for method, group in ((name, ablation_df.loc[ablation_df["method"] == name]) for name in METHODS)
    }
    (out_path / "summary.json").write_text(json.dumps(summary, indent=2))

    print("SP500 OU ABLATION")
    print(f"universe: top-{top_n} by Russell-3000 IWV weight (proxy)")
    print(f"candidates_screened: {_screenable_pair_count(metadata)}")
    print(f"pairs_validated: {len(pair_df)}")
    print(f"pairs_in_ablation: {len(candidate_df)}")
    print("ablation_summary (median net Sharpe per method):")
    for method in METHODS:
        print(f"  {method:<9}: {summary[method]['median_sharpe_net']:.2f}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run P4 Week 6 S&P 500 proxy OU ablation.")
    parser.add_argument("--panel-dir", required=True)
    parser.add_argument("--universe-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-n", type=int, default=500)
    parser.add_argument("--max-pairs", type=int, default=200)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    try:
        return run_ablation(
            panel_dir=args.panel_dir,
            universe_csv=args.universe_csv,
            out_dir=args.out_dir,
            top_n=args.top_n,
            max_pairs=args.max_pairs,
        )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
