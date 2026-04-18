from __future__ import annotations

import json

from p4.config import load_config
from p4.data_loader import load_market_inputs
from p4.pair_selector import PairSelector

from .conftest import write_config


def test_pair_selector_returns_pairs_and_baskets_from_fixture(tmp_path) -> None:
    config = load_config(write_config(tmp_path))
    _, metadata, prices, _ = load_market_inputs(config)
    formation = prices.iloc[:180]
    validation = prices.iloc[180:240]

    pair_df, basket_df = PairSelector(config).select(
        formation_prices=formation,
        validation_prices=validation,
        metadata=metadata,
        window_id=0,
    )

    assert not pair_df.empty
    assert not basket_df.empty
    assert set(pair_df["strategy_type"]) == {"pair"}
    assert set(basket_df["strategy_type"]) == {"basket"}
    assert pair_df["half_life"].between(config.ou.half_life_min_days, config.ou.half_life_max_days).all()
    assert basket_df["half_life"].between(config.ou.half_life_min_days, config.ou.half_life_max_days).all()
    assert basket_df["tickers_json"].map(lambda raw: len(json.loads(raw)) == 3).all()
