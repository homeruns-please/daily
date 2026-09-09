"""Prediction targets."""

from __future__ import annotations

import pandas as pd


def make_hr_target(game_batter: pd.DataFrame) -> pd.Series:
    """Return 1 when a batter hits at least one HR in the target game."""
    return (game_batter["home_runs"] >= 1).astype("int8")
