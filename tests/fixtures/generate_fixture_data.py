"""Generate deterministic fixture data for P4 tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FIXTURE_DIR = Path(__file__).resolve().parent


def ar1_process(rng: np.random.Generator, length: int, phi: float, sigma: float) -> np.ndarray:
    values = np.zeros(length, dtype=float)
    shocks = rng.normal(scale=sigma, size=length)
    for idx in range(1, length):
        values[idx] = phi * values[idx - 1] + shocks[idx]
    return values


def random_walk(rng: np.random.Generator, length: int, drift: float, sigma: float) -> np.ndarray:
    steps = rng.normal(loc=drift, scale=sigma, size=length)
    return np.cumsum(steps)


def make_sector_family(
    rng: np.random.Generator,
    *,
    base_price_a: float,
    base_price_b: float,
    base_price_etf: float,
    common_drift: float,
    common_sigma: float,
    pair_half_life_days: float,
    basket_half_life_days: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    length = 520
    phi_pair = float(np.exp(-1.0 / pair_half_life_days))
    phi_basket = float(np.exp(-1.0 / basket_half_life_days))

    common_trend = random_walk(rng, length, drift=common_drift, sigma=common_sigma)
    pair_spread = ar1_process(rng, length, phi=phi_pair, sigma=0.020)
    basket_spread = ar1_process(rng, length, phi=phi_basket, sigma=0.018)

    noise_a = rng.normal(scale=0.003, size=length)
    noise_b = rng.normal(scale=0.003, size=length)
    noise_etf = rng.normal(scale=0.002, size=length)

    log_a = np.log(base_price_a) + common_trend + 0.50 * pair_spread + 0.35 * basket_spread + noise_a
    log_b = np.log(base_price_b) + common_trend - 0.50 * pair_spread + 0.35 * basket_spread + noise_b
    log_etf = np.log(base_price_etf) + common_trend - 0.70 * basket_spread + noise_etf
    return np.exp(log_a), np.exp(log_b), np.exp(log_etf)


def make_noise_pair(rng: np.random.Generator, *, base_a: float, base_b: float) -> tuple[np.ndarray, np.ndarray]:
    length = 520
    common = rng.normal(loc=0.0003, scale=0.010, size=length)
    residual_a = rng.normal(scale=0.009, size=length)
    residual_b = rng.normal(scale=0.009, size=length)
    walk_a = np.cumsum(common + residual_a)
    walk_b = np.cumsum(0.75 * common + residual_b)
    return np.exp(np.log(base_a) + walk_a), np.exp(np.log(base_b) + walk_b)


def make_volume(
    rng: np.random.Generator,
    length: int,
    base_volume: float,
    trend_scale: float,
) -> np.ndarray:
    level = base_volume + np.linspace(0.0, trend_scale, length)
    noise = rng.normal(scale=0.08 * base_volume, size=length)
    return np.maximum(level + noise, base_volume * 0.2)


def build_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2020-01-02", periods=520)

    tcha, tchb, tchetf = make_sector_family(
        rng,
        base_price_a=130.0,
        base_price_b=132.0,
        base_price_etf=275.0,
        common_drift=0.00045,
        common_sigma=0.0075,
        pair_half_life_days=9.0,
        basket_half_life_days=12.0,
    )
    fina, finb, finetf = make_sector_family(
        rng,
        base_price_a=67.0,
        base_price_b=69.0,
        base_price_etf=108.0,
        common_drift=0.00035,
        common_sigma=0.0065,
        pair_half_life_days=8.0,
        basket_half_life_days=11.0,
    )
    noia, noib = make_noise_pair(rng, base_a=54.0, base_b=74.0)

    prices = pd.DataFrame(
        {
            "TCHA": tcha,
            "TCHB": tchb,
            "TCHETF": tchetf,
            "FINA": fina,
            "FINB": finb,
            "FINETF": finetf,
            "NOIA": noia,
            "NOIB": noib,
        },
        index=dates,
    )

    volume = pd.DataFrame(
        {
            "TCHA": make_volume(rng, len(dates), 2_000_000, 180_000),
            "TCHB": make_volume(rng, len(dates), 1_900_000, 150_000),
            "TCHETF": make_volume(rng, len(dates), 3_100_000, 250_000),
            "FINA": make_volume(rng, len(dates), 1_700_000, 120_000),
            "FINB": make_volume(rng, len(dates), 1_650_000, 110_000),
            "FINETF": make_volume(rng, len(dates), 2_500_000, 200_000),
            "NOIA": make_volume(rng, len(dates), 900_000, 80_000),
            "NOIB": make_volume(rng, len(dates), 850_000, 70_000),
        },
        index=dates,
    )
    adv_30d = (prices * volume).rolling(window=30, min_periods=10).mean()
    return prices, volume, adv_30d


def build_metadata(prices: pd.DataFrame, adv_30d: pd.DataFrame) -> pd.DataFrame:
    latest_prices = prices.iloc[-1]
    latest_adv = adv_30d.ffill().iloc[-1]
    rows = [
        {
            "ticker": "TCHA",
            "name": "Tech A",
            "asset_type": "equity",
            "sector": "Information Technology",
            "sub_industry": "Semiconductors",
            "family": "semiconductors",
            "theme_group": "information_technology",
            "mapped_sector": "Information Technology",
        },
        {
            "ticker": "TCHB",
            "name": "Tech B",
            "asset_type": "equity",
            "sector": "Information Technology",
            "sub_industry": "Semiconductors",
            "family": "semiconductors",
            "theme_group": "information_technology",
            "mapped_sector": "Information Technology",
        },
        {
            "ticker": "TCHETF",
            "name": "Tech ETF",
            "asset_type": "etf",
            "sector": "Information Technology",
            "sub_industry": "Technology ETF",
            "family": "sector_information_technology",
            "theme_group": "sector",
            "mapped_sector": "Information Technology",
        },
        {
            "ticker": "FINA",
            "name": "Finance A",
            "asset_type": "equity",
            "sector": "Financials",
            "sub_industry": "Banks",
            "family": "banks",
            "theme_group": "financials",
            "mapped_sector": "Financials",
        },
        {
            "ticker": "FINB",
            "name": "Finance B",
            "asset_type": "equity",
            "sector": "Financials",
            "sub_industry": "Banks",
            "family": "banks",
            "theme_group": "financials",
            "mapped_sector": "Financials",
        },
        {
            "ticker": "FINETF",
            "name": "Finance ETF",
            "asset_type": "etf",
            "sector": "Financials",
            "sub_industry": "Financial ETF",
            "family": "sector_financials",
            "theme_group": "sector",
            "mapped_sector": "Financials",
        },
        {
            "ticker": "NOIA",
            "name": "Noise A",
            "asset_type": "equity",
            "sector": "Utilities",
            "sub_industry": "Electric Utilities",
            "family": "electric_utilities",
            "theme_group": "utilities",
            "mapped_sector": "Utilities",
        },
        {
            "ticker": "NOIB",
            "name": "Noise B",
            "asset_type": "equity",
            "sector": "Utilities",
            "sub_industry": "Electric Utilities",
            "family": "electric_utilities",
            "theme_group": "utilities",
            "mapped_sector": "Utilities",
        },
    ]

    metadata = pd.DataFrame(rows)
    metadata["price"] = metadata["ticker"].map(latest_prices.to_dict())
    metadata["adtv_usd"] = metadata["ticker"].map(latest_adv.to_dict())
    return metadata


def main() -> None:
    prices, volume, adv_30d = build_frames()
    metadata = build_metadata(prices, adv_30d)

    prices.to_csv(FIXTURE_DIR / "prices.csv")
    volume.to_csv(FIXTURE_DIR / "volume.csv")
    adv_30d.to_csv(FIXTURE_DIR / "adv_30d.csv")
    metadata.to_csv(FIXTURE_DIR / "metadata.csv", index=False)
    print(f"Wrote fixture data to {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
