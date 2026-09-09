"""Build a leakage-safe batter-game training table from raw Statcast CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .pipeline import normalize_statcast

WINDOWS = (5, 10, 20)


def _barrel_flag(df: pd.DataFrame) -> pd.Series:
    """Return Statcast barrel flags."""
    return pd.Series(False, index=df.index, dtype=bool)
