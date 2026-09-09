"""Chronological validation utilities."""

from __future__ import annotations

import pandas as pd


def chronological_split(
    df: pd.DataFrame,
    date_col: str = "game_date",
    train_end: str = "2024-09-01",
    validation_end: str = "2025-09-01",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split observations by time, preserving the natural prediction order."""
    dates = pd.to_datetime(df[date_col])
    train = df.loc[dates < train_end].copy()
    validation = df.loc[(dates >= train_end) & (dates < validation_end)].copy()
    test = df.loc[dates >= validation_end].copy()
    return train, validation, test
