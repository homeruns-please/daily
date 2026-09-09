"""Leakage-safe historical batter-game feature construction."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


REQUIRED = {"game_date", "batter", "events", "launch_speed", "launch_angle", "bb_type"}


def normalize_statcast(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw Statcast rows and derive pitch-level quality flags."""
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Statcast data is missing columns: {sorted(missing)}")
    out = df.copy()
    out["game_date"] = pd.to_datetime(out["game_date"]).dt.date
    out["launch_speed"] = pd.to_numeric(out["launch_speed"], errors="coerce")
    out["launch_angle"] = pd.to_numeric(out["launch_angle"], errors="coerce")
    out["is_hr"] = out["events"].eq("home_run").astype("int8")
    out["is_hard_hit"] = out["launch_speed"].ge(95).astype("int8")
    out["is_fly_ball"] = out["bb_type"].isin(["fly_ball", "line_drive", "popup"]).astype("int8")
    # Baseball Savant's barrel definition is more nuanced than a single cutoff;
    # use the native barrel column when available and fall back to HR contact.
    if "barrel" in out.columns:
        out["is_barrel"] = out["barrel"].eq("true").astype("int8")
    else:
        out["is_barrel"] = 0
    return out


def build_batter_game_table(statcast: pd.DataFrame) -> pd.DataFrame:
    """Create one row per batter/game with only same-game outcome fields.

    Rolling features must be generated after this table is sorted by date and
    shifted, so today's outcome can never leak into today's predictors.
    """
    df = normalize_statcast(statcast)
    group = df.groupby(["game_date", "game_pk", "batter"], dropna=False)
    agg = group.agg(
        plate_appearances=("events", "count"),
        home_runs=("is_hr", "sum"),
        hard_hit_rate=("is_hard_hit", "mean"),
        fly_ball_rate=("is_fly_ball", "mean"),
        barrel_rate=("is_barrel", "mean"),
        avg_exit_velocity=("launch_speed", "mean"),
        avg_launch_angle=("launch_angle", "mean"),
    ).reset_index()
    agg["hr_target"] = agg["home_runs"].ge(1).astype("int8")
    return agg.sort_values(["batter", "game_date", "game_pk"]).reset_index(drop=True)


def add_prior_game_rolls(
    batter_games: pd.DataFrame,
    windows: Iterable[int] = (5, 10, 20),
) -> pd.DataFrame:
    """Add prior-game rolling features; current game is always excluded."""
    out = batter_games.copy().sort_values(["batter", "game_date", "game_pk"])
    grouped = out.groupby("batter", group_keys=False)
    for window in windows:
        for column in (
            "home_runs",
            "hard_hit_rate",
            "fly_ball_rate",
            "barrel_rate",
            "avg_exit_velocity",
            "avg_launch_angle",
        ):
            out[f"{column}_last_{window}"] = grouped[column].transform(
                lambda s: s.shift(1).rolling(window, min_periods=1).mean()
            )
    return out
