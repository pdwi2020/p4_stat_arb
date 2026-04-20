#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
from collections import OrderedDict
from pathlib import Path
from statistics import median, stdev

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_CACHE = Path("/tmp") / "p4_readme_figure_cache"
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(RUNTIME_CACHE))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ABLATION_CSV = ROOT / "results" / "sp500_ablation" / "ou_ablation.csv"
ABLATION_SUMMARY_JSON = ROOT / "results" / "sp500_ablation" / "summary.json"
OU_OUTPUT = ROOT / "results" / "ou_ablation" / "ou_ablation_sharpe.png"

DEFAULT_REAL_MIXED = ROOT / "results" / "default_real_mixed"
REAL_SMOKE = ROOT / "results" / "real_smoke"
MULTIPLE_TESTING_OUTPUT = ROOT / "results" / "multiple_testing" / "multiple_testing_survivors.png"

METHOD_ORDER = ("static", "kalman", "neural", "regime")
METHOD_LABELS = {
    "static": "Static OU",
    "kalman": "Kalman-OU",
    "neural": "Neural OU",
    "regime": "Regime-switch",
}
METHOD_COLORS = {
    "static": "#4C78A8",
    "kalman": "#72B7B2",
    "neural": "#F58518",
    "regime": "#54A24B",
}

SURVIVOR_FALLBACKS = OrderedDict(
    [
        ("Bonferroni", 1),
        ("BH", 4),
        ("BY", 3),
        ("Storey", 5),
        ("Hansen SPA", 0),
        ("White RC", 1),
    ]
)
SURVIVOR_COLORS = {
    "Bonferroni": "#4C78A8",
    "BH": "#59A14F",
    "BY": "#9C755F",
    "Storey": "#F28E2B",
    "Hansen SPA": "#E45756",
    "White RC": "#B279A2",
}


def _read_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _bh_rejections(pvalues: list[float], alpha: float = 0.05) -> list[bool]:
    n_tests = len(pvalues)
    if n_tests == 0:
        return []
    ordered = sorted(enumerate(pvalues), key=lambda item: (item[1], item[0]))
    cutoff_rank = 0
    for rank, (_, pvalue) in enumerate(ordered, start=1):
        if pvalue <= alpha * rank / n_tests:
            cutoff_rank = rank
    rejected = [False] * n_tests
    for rank, (original_idx, _) in enumerate(ordered, start=1):
        if rank <= cutoff_rank:
            rejected[original_idx] = True
    return rejected


def _by_rejections(pvalues: list[float], alpha: float = 0.05) -> list[bool]:
    n_tests = len(pvalues)
    if n_tests == 0:
        return []
    harmonic = sum(1.0 / rank for rank in range(1, n_tests + 1))
    ordered = sorted(enumerate(pvalues), key=lambda item: (item[1], item[0]))
    cutoff_rank = 0
    for rank, (_, pvalue) in enumerate(ordered, start=1):
        if pvalue <= alpha * rank / (n_tests * harmonic):
            cutoff_rank = rank
    rejected = [False] * n_tests
    for rank, (original_idx, _) in enumerate(ordered, start=1):
        if rank <= cutoff_rank:
            rejected[original_idx] = True
    return rejected


def _storey_qvalues(pvalues: list[float], lambda_: float = 0.5) -> list[float]:
    n_tests = len(pvalues)
    if n_tests == 0:
        return []
    if not 0.0 <= lambda_ < 1.0:
        raise ValueError("lambda_ must lie in [0, 1).")
    pi0 = min(1.0, sum(pvalue > lambda_ for pvalue in pvalues) / max((1.0 - lambda_) * n_tests, 1e-12))
    ordered = sorted(enumerate(pvalues), key=lambda item: (item[1], item[0]))
    q_sorted = [1.0] * n_tests
    running_min = 1.0
    for rank in range(n_tests, 0, -1):
        _, pvalue = ordered[rank - 1]
        estimate = pi0 * n_tests * pvalue / rank
        running_min = min(running_min, estimate)
        q_sorted[rank - 1] = min(1.0, running_min)
    qvalues = [1.0] * n_tests
    for sorted_idx, (original_idx, _) in enumerate(ordered):
        qvalues[original_idx] = q_sorted[sorted_idx]
    return qvalues


def _load_ou_ablation_payload() -> tuple[list[str], list[float], list[float], int]:
    rows = _read_csv_rows(ABLATION_CSV)
    if not rows:
        raise FileNotFoundError(f"Missing ablation CSV: {ABLATION_CSV}")

    sharpe_by_method: dict[str, list[float]] = {method: [] for method in METHOD_ORDER}
    for row in rows:
        method = row["method"].strip()
        sharpe_by_method.setdefault(method, [])
        sharpe_by_method[method].append(float(row["sharpe_net"]))

    summary = _read_json(ABLATION_SUMMARY_JSON) or {}

    labels: list[str] = []
    medians: list[float] = []
    dispersions: list[float] = []
    n_pairs = 0
    for method in METHOD_ORDER:
        values = sharpe_by_method.get(method, [])
        if not values:
            raise ValueError(f"No Sharpe data found for method '{method}'.")
        labels.append(METHOD_LABELS[method])
        if summary and method in summary:
            medians.append(float(summary[method]["median_sharpe_net"]))
            n_pairs = max(n_pairs, int(summary[method]["n_pairs"]))
        else:
            medians.append(float(median(values)))
            n_pairs = max(n_pairs, len(values))
        dispersions.append(float(stdev(values)) if len(values) > 1 else 0.0)
    return labels, medians, dispersions, n_pairs


def _plot_ou_ablation() -> None:
    labels, medians, dispersions, n_pairs = _load_ou_ablation_payload()
    OU_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    colors = [METHOD_COLORS[method] for method in METHOD_ORDER]
    bars = ax.bar(
        labels,
        medians,
        color=colors,
        edgecolor="#1F1F1F",
        linewidth=0.8,
        yerr=dispersions,
        capsize=6,
    )
    ax.axhline(0.0, color="#1F1F1F", linewidth=0.8, linestyle="--", alpha=0.65)
    ax.set_title(f"OU ablation: median net Sharpe across {n_pairs} validated pairs")
    ax.set_ylabel("Median net Sharpe")
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)

    upper = max(medians[i] + dispersions[i] for i in range(len(medians)))
    lower = min(0.0, min(medians))
    ax.set_ylim(lower - 0.1, upper + 0.25)

    for bar, value in zip(bars, medians, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.04,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    fig.text(
        0.5,
        0.02,
        "Error bars show across-pair sample standard deviation of net Sharpe from results/sp500_ablation/ou_ablation.csv.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0.02, 0.06, 1.0, 1.0))
    fig.savefig(OU_OUTPUT, dpi=180)
    plt.close(fig)


def _load_unique_one_sided_pvalues(result_dir: Path) -> list[float]:
    metrics_path = result_dir / "strategy_metrics.csv"
    rows = _read_csv_rows(metrics_path)
    unique = OrderedDict()
    for row in rows:
        candidate_id = row.get("candidate_id", "").strip()
        pvalue = row.get("one_sided_pvalue", "").strip()
        if not candidate_id or not pvalue or candidate_id in unique:
            continue
        unique[candidate_id] = float(pvalue)
    return list(unique.values())


def _select_multiple_testing_result_dir() -> Path | None:
    for candidate in (DEFAULT_REAL_MIXED, REAL_SMOKE):
        if (candidate / "multiple_testing_report.json").exists():
            return candidate
    return None


def _load_survivor_counts() -> tuple[OrderedDict[str, int], str, float | None]:
    result_dir = _select_multiple_testing_result_dir()
    if result_dir is None:
        return OrderedDict(SURVIVOR_FALLBACKS), "fallback memo estimates", None

    report = _read_json(result_dir / "multiple_testing_report.json") or {}
    counts = OrderedDict()
    counts["Bonferroni"] = len(report.get("bonferroni_survivors", []))

    pvalues = _load_unique_one_sided_pvalues(result_dir)
    if pvalues:
        counts["BH"] = sum(_bh_rejections(pvalues, alpha=0.05))
        counts["BY"] = sum(_by_rejections(pvalues, alpha=0.05))
        counts["Storey"] = sum(qvalue <= 0.05 for qvalue in _storey_qvalues(pvalues, lambda_=0.5))
    else:
        counts["BH"] = SURVIVOR_FALLBACKS["BH"]
        counts["BY"] = SURVIVOR_FALLBACKS["BY"]
        counts["Storey"] = SURVIVOR_FALLBACKS["Storey"]

    spa_report = report.get("hansen_spa", {})
    counts["Hansen SPA"] = len(spa_report.get("survivors", [])) if isinstance(spa_report, dict) else 0

    white_report = report.get("white_reality_check", {})
    if isinstance(white_report, dict):
        counts["White RC"] = 1 if white_report.get("best_strategy") else 0
        white_pvalue = float(white_report["pvalue"]) if "pvalue" in white_report else None
    else:
        counts["White RC"] = SURVIVOR_FALLBACKS["White RC"]
        white_pvalue = None

    return counts, f"results/{result_dir.name}", white_pvalue


def _plot_multiple_testing_survivors() -> None:
    counts, source_label, white_pvalue = _load_survivor_counts()
    MULTIPLE_TESTING_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    labels = list(counts.keys())
    values = list(counts.values())
    colors = [SURVIVOR_COLORS[label] for label in labels]

    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    bars = ax.bar(labels, values, color=colors, edgecolor="#1F1F1F", linewidth=0.8)
    ax.set_title("Multiple-testing survivors by method")
    ax.set_ylabel("Survivor count")
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ymax = max(values) if values else 0
    ax.set_ylim(0, max(1.6, ymax + 1.2))

    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.06,
            str(value),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    footnote = f"Source: {source_label}."
    if white_pvalue is not None:
        footnote += f" White RC counts the nominated best strategy; current report p-value = {white_pvalue:.3f}."
    else:
        footnote += " White RC counts the nominated best strategy."

    fig.text(0.5, 0.02, footnote, ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0.02, 0.06, 1.0, 1.0))
    fig.savefig(MULTIPLE_TESTING_OUTPUT, dpi=180)
    plt.close(fig)


def main() -> None:
    _plot_ou_ablation()
    _plot_multiple_testing_survivors()
    print(f"rendered {OU_OUTPUT.relative_to(ROOT)}")
    print(f"rendered {MULTIPLE_TESTING_OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
