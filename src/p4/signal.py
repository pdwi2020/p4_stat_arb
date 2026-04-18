"""Z-score spread signals for P4."""

from __future__ import annotations

import numpy as np
import pandas as pd


class ZScoreSignal:
    """Generate mean-reversion spread positions from OU-centered z-scores."""

    def __init__(self, *, entry_z: float = 2.0, exit_z: float = 0.5, stop_z: float = 4.0):
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z

    def compute(self, spread: pd.Series, mu: float, stationary_sigma: float) -> pd.Series:
        sigma = max(float(stationary_sigma), 1e-8)
        zscore = (pd.Series(spread, dtype=float) - float(mu)) / sigma
        zscore.name = "zscore"
        return zscore

    def generate_positions(self, zscore: pd.Series) -> pd.Series:
        position = 0
        positions = []
        for value in pd.Series(zscore, dtype=float).fillna(0.0):
            if position == 0:
                if value <= -self.entry_z:
                    position = 1
                elif value >= self.entry_z:
                    position = -1
            elif position == 1:
                if abs(value) >= self.stop_z or abs(value) <= self.exit_z:
                    position = 0
            elif position == -1:
                if abs(value) >= self.stop_z or abs(value) <= self.exit_z:
                    position = 0
            positions.append(position)
        return pd.Series(positions, index=zscore.index, name="position", dtype=float)
