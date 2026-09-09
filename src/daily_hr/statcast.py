"""Leakage-safe Statcast aggregation for batter and pitcher game features."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

REQUIRED = {"game_date", "batter", "pitcher", "events"}


def _check(df: pd.DataFrame) -> None:
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Statcast data missing columns: {sorted(missing)}")


def batter_game_table(pitches: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pitch-level Statcast rows into one row per batter/game."""
    _check(pitches)
    df = pitches.copy()
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    df["is_barrel"] = (df.get("launch_speed_angle") == 6).astype("int8")
    df["is_hard_hit"] = (pd.to_numeric(df.get("launch_speed"), errors="coerce") >= 95).astype("int8")
    df["is_fly_ball"] = df.get("bb_type", pd.Series(index=df.index)).isin(["fly_ball", "popup"]).astype("int8")
    df["is_hr"] = df["events"].eq("home_run").astype("int8")

    pa = df[df["events"].notna()].groupby(["game_date", "batter"], as_index=False).agg(
        plate_appearances=("events", "size"),
        home_runs=("is_hr", "sum"),
        barrel_rate=("is_barrel", "mean"),
        hard_hit_rate=("is_hard_hit", "mean"),
        fly_ball_rate=("is_fly_ball", "mean"),
        exit_velocity=("launch_speed", "mean"),
        launch_angle=("launch_angle", "mean"),
    )
    return pa.sort_values(["game_date", "batter"]).reset_index(drop=True)


def pitcher_game_table(pitches: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pitch-level Statcast rows into one row per pitcher/game."""
    _check(pitches)
    df = pitches.copy()
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    df["is_barrel"] = (df.get("launch_speed_angle") == 6).astype("int8")
    df["is_hr"] = df["events"].eq("home_run").astype("int8")
    df["is_hard_hit"] = (pd.to_numeric(df.get("launch_speed"), errors="coerce") >= 95).astype("int8")

    out = df.groupby(["game_date", "pitcher"], as_index=False).agg(
        pitches=("pitch_type", "size"),
        home_runs_allowed=("is_hr", "sum"),
        barrel_rate_allowed=("is_barrel", "mean"),
        hard_hit_rate_allowed=("is_hard_hit", "mean"),
        exit_velocity_allowed=("launch_speed", "mean"),
    )
    return out.sort_values(["game_date", "pitcher"]).reset_index(drop=True)


def pitch_mix(pitches: pd.DataFrame, pitcher_ids: Iterable[int] | None = None) -> pd.DataFrame:
    """Calculate pitch-type usage by pitcher from prior data."""
    _check(pitches)
    df = pitches.copy()
    if pitcher_ids is not None:
        df = df[df["pitcher"].isin(list(pitcher_ids))]
    return (
        df.dropna(subset=["pitch_type"])
        .groupby(["pitcher", "pitch_type"])
        .size()
        .groupby(level=0)
        .transform(lambda x: x / x.sum())
        .rename("usage")
        .reset_index()
    )
