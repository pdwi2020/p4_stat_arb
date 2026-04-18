"""Typed configuration helpers for P4."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


REPO_ROOT = Path(__file__).resolve().parents[2]


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cache_dir: Path = Path("/Volumes/Crucial X9/data/market_data/p4")
    results_dir: Path = Path("results")
    fixture_dir: Path = Path("tests/fixtures")
    alpha_trace_dir: Path = Path("docs/alpha_engine_trace")


class MarketConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_start: date = date(2016, 1, 1)
    data_end: date = date(2025, 12, 31)
    snapshot_period: str = "3mo"
    price_batch_size: int = Field(default=80, ge=10)
    etf_universe: list[str] = Field(
        default_factory=lambda: [
            "SPY",
            "QQQ",
            "IWM",
            "DIA",
            "XLK",
            "XLF",
            "XLE",
            "XLU",
            "XLP",
            "XLY",
            "XLI",
            "XLV",
            "XLB",
            "XLC",
            "VNQ",
            "XBI",
            "SMH",
            "KRE",
            "HYG",
            "TLT",
        ]
    )


class SelectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_equities: int = Field(default=30, ge=5)
    min_price: float = Field(default=5.0, ge=0.0)
    min_adtv_usd: float = Field(default=25_000_000.0, ge=0.0)
    min_pair_correlation: float = Field(default=0.6, ge=0.0, le=1.0)
    max_pairs_per_family: int = Field(default=6, ge=1)
    max_baskets_per_family: int = Field(default=3, ge=1)


class CointegrationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engle_granger_alpha: float = Field(default=0.05, gt=0.0, lt=0.25)
    johansen_alpha: float = Field(default=0.05, gt=0.0, lt=0.25)


class OUConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    half_life_min_days: float = Field(default=5.0, ge=1.0)
    half_life_max_days: float = Field(default=30.0, gt=5.0)


class SignalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entry_z: float = Field(default=2.0, gt=0.0)
    exit_z: float = Field(default=0.5, ge=0.0)
    stop_z: float = Field(default=4.0, gt=0.0)


class BacktestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cost_halfspread_bps: float = Field(default=5.0, ge=0.0)
    cost_slippage_bps: float = Field(default=3.0, ge=0.0)
    borrow_cost_annual_bps: float = Field(default=50.0, ge=0.0)
    capacity_adv_fraction: float = Field(default=0.05, gt=0.0, le=1.0)
    portfolio_capital_usd: float = Field(default=10_000_000.0, gt=0.0)


class WalkforwardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    formation_days: int = Field(default=504, ge=100)
    validation_days: int = Field(default=126, ge=21)
    test_days: int = Field(default=126, ge=21)
    step_days: int = Field(default=126, ge=21)


class EigenportfolioConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pca_components: int = Field(default=15, ge=1)
    formation_window_days: int = Field(default=252, ge=60)
    min_half_life_days: float = Field(default=5.0, ge=1.0)
    max_half_life_days: float = Field(default=30.0, gt=1.0)
    z_entry: float = Field(default=1.5, gt=0.0)
    z_exit: float = Field(default=0.5, ge=0.0)
    z_stop: float = Field(default=3.0, gt=0.0)

    @model_validator(mode="after")
    def _validate_half_life_bounds(self) -> "EigenportfolioConfig":
        if self.max_half_life_days <= self.min_half_life_days:
            raise ValueError("eigenportfolio.max_half_life_days must exceed min_half_life_days.")
        return self


class MultipleTestingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_bootstrap: int = Field(default=300, ge=50)
    block_length_mode: str = "ou_half_life"
    alpha: float = Field(default=0.05, gt=0.0, lt=0.25)
    random_seed: int = 42


class P4Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_name: str = "default_real_mixed"
    data_mode: str = "real"
    strategy_type: Literal["cointegration", "eigenportfolio"] = "cointegration"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    market: MarketConfig = Field(default_factory=MarketConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    cointegration: CointegrationConfig = Field(default_factory=CointegrationConfig)
    ou: OUConfig = Field(default_factory=OUConfig)
    signal: SignalConfig = Field(default_factory=SignalConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    walkforward: WalkforwardConfig = Field(default_factory=WalkforwardConfig)
    eigenportfolio: EigenportfolioConfig = Field(default_factory=EigenportfolioConfig)
    multiple_testing: MultipleTestingConfig = Field(default_factory=MultipleTestingConfig)

    @model_validator(mode="after")
    def _resolve_relative_paths(self) -> "P4Config":
        self.data_mode = self.data_mode.lower()
        self.strategy_type = self.strategy_type.lower()
        self.paths.cache_dir = resolve_path(self.paths.cache_dir)
        self.paths.results_dir = resolve_path(self.paths.results_dir)
        self.paths.fixture_dir = resolve_path(self.paths.fixture_dir)
        self.paths.alpha_trace_dir = resolve_path(self.paths.alpha_trace_dir)
        self.market.etf_universe = [ticker.upper() for ticker in self.market.etf_universe]
        return self

    def results_path(self, *parts: str) -> Path:
        return self.paths.results_dir.joinpath(self.run_name, *parts)

    def alpha_trace_path(self, *parts: str) -> Path:
        return self.paths.alpha_trace_dir.joinpath(self.run_name, *parts)


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def default_config_path() -> Path:
    return REPO_ROOT / "configs" / "p4_config.yaml"


def load_config(path: str | Path | None = None) -> P4Config:
    config_path = resolve_path(path) if path is not None else default_config_path()
    raw: dict[str, Any] = {}
    if config_path.exists():
        raw = yaml.safe_load(config_path.read_text()) or {}
    return P4Config.model_validate(raw)


def ensure_run_directories(config: P4Config) -> Path:
    run_dir = config.results_path()
    run_dir.mkdir(parents=True, exist_ok=True)
    config.paths.cache_dir.mkdir(parents=True, exist_ok=True)
    config.paths.alpha_trace_dir.mkdir(parents=True, exist_ok=True)
    config.alpha_trace_path().mkdir(parents=True, exist_ok=True)
    return run_dir
