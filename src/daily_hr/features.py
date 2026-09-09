"""Point-in-time feature definitions for the HR model.

The key rule in this module: features for game date D may only use data with
an event date strictly before D. Rolling helpers should be applied after
sorting by event date and must use shift(1) before rolling calculations.
"""

from __future__ import annotations

import pandas as pd


def add_rolling_mean(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    windows: tuple[int, ...] = (7, 14, 20),
) -> pd.DataFrame:
    """Add leakage-safe rolling means based only on prior observations."""
    out = df.sort_values(group_cols + ["game_date"]).copy()
    grouped = out.groupby(group_cols, sort=False)[value_col]
    for window in windows:
        name = f"{value_col}_rolling_{window}"
        out[name] = grouped.transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=max(3, window // 3)).mean()
        )
    return out


def add_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Safe rate calculation; missing/zero denominators remain NaN."""
    return numerator.div(denominator.where(denominator > 0))
