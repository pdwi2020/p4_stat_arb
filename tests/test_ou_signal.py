from __future__ import annotations

import numpy as np
import pandas as pd

from p4.ou_estimator import OUEstimator
from p4.signal import ZScoreSignal


def test_ou_estimator_recovers_reasonable_half_life() -> None:
    rng = np.random.default_rng(21)
    phi = float(np.exp(-1.0 / 7.0))
    values = np.zeros(300, dtype=float)
    shocks = rng.normal(scale=0.05, size=len(values))
    for idx in range(1, len(values)):
        values[idx] = phi * values[idx - 1] + shocks[idx]

    params = OUEstimator().fit(pd.Series(values))
    assert 4.0 <= params["half_life"] <= 10.0
    assert params["stationary_sigma"] > 0.0
    assert params["kappa"] > 0.0


def test_zscore_signal_entries_exits_and_stop_logic() -> None:
    signal = ZScoreSignal(entry_z=1.5, exit_z=0.25, stop_z=4.0)
    zscore = pd.Series([0.0, -1.6, -1.2, -0.1, 0.0, 1.7, 4.1, 0.0])
    positions = signal.generate_positions(zscore)
    assert positions.tolist() == [0.0, 1.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0]
