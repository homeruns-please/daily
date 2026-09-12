"""Build a leakage-safe batter-game training table from raw Statcast CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .pipeline import normalize_statcast

WINDOWS = (5, 10, 20)
ROLL_COLUMNS = (
    "home_runs", "plate_appearances", "barrel_pct", "hard_hit_pct",
    "fly_ball_pct", "launch_angle_avg", "exit_velocity_avg",
)
PREGAME_FEATURES = [
    f"{column}_last_{window}" for window in WINDOWS for column in ROLL_COLUMNS
] + [
    "hr_per_pa_last_20", "hr_per_pa_last_5", "hr_rate_change_5_vs_20",
    "barrel_rate_change_5_vs_20", "hard_hit_rate_change_5_vs_20",
    "exit_velocity_change_5_vs_20", "fly_ball_change_5_vs_20", "barrel_to_hr_gap_20",
]


def _barrel_flag(df: pd.DataFrame) -> pd.Series:
    if "barrel" in df:
        return df["barrel"].astype(str).str.lower().eq("true").astype("int8")
    if "launch_speed_angle" in df:
        return pd.to_numeric(df["launch_speed_angle"], errors="coerce").eq(6).astype("int8")
    return pd.Series(0, index=df.index, dtype="int8")


def build_batter_games(statcast: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw pitches and add prior-game rolling predictors."""
    df = normalize_statcast(statcast)
    df["is_barrel"] = _barrel_flag(df)
    df["is_fly_ball"] = df["bb_type"].isin(["fly_ball", "line_drive", "popup"]).astype("int8")

    grouped = df.groupby(["game_date", "game_pk", "batter"], dropna=False)
    out = grouped.agg(
        plate_appearances=("events", "count"),
        home_runs=("is_hr", "sum"),
        barrel_pct=("is_barrel", "mean"),
        hard_hit_pct=("is_hard_hit", "mean"),
        fly_ball_pct=("is_fly_ball", "mean"),
        launch_angle_avg=("launch_angle", "mean"),
        exit_velocity_avg=("launch_speed", "mean"),
    ).reset_index()

    out["hr_target"] = out["home_runs"].ge(1).astype("int8")
    out = out.sort_values(["batter", "game_date", "game_pk"]).reset_index(drop=True)

    for window in WINDOWS:
        grouped_out = out.groupby("batter", group_keys=False)
        for column in (
            "home_runs",
            "plate_appearances",
            "barrel_pct",
            "hard_hit_pct",
            "fly_ball_pct",
            "launch_angle_avg",
            "exit_velocity_avg",
        ):
            out[f"{column}_last_{window}"] = grouped_out[column].transform(
                lambda s, window=window: s.shift(1).rolling(window, min_periods=1).mean()
            )

    out["hr_per_pa_last_20"] = out["home_runs_last_20"].div(
        out["plate_appearances_last_20"].replace(0, pd.NA)
    )
    out["hr_per_pa_last_5"] = out["home_runs_last_5"].div(
        out["plate_appearances_last_5"].replace(0, pd.NA)
    )

    out["hr_rate_change_5_vs_20"] = out["hr_per_pa_last_5"] - out["hr_per_pa_last_20"]
    out["barrel_rate_change_5_vs_20"] = out["barrel_pct_last_5"] - out["barrel_pct_last_20"]
    out["hard_hit_rate_change_5_vs_20"] = out["hard_hit_pct_last_5"] - out["hard_hit_pct_last_20"]
    out["exit_velocity_change_5_vs_20"] = out["exit_velocity_avg_last_5"] - out["exit_velocity_avg_last_20"]
    out["fly_ball_change_5_vs_20"] = out["fly_ball_pct_last_5"] - out["fly_ball_pct_last_20"]
    out["barrel_to_hr_gap_20"] = out["barrel_pct_last_20"] - out["hr_per_pa_last_20"]
    # Both games of a doubleheader use the same prior-day snapshot.
    first = out.drop_duplicates(["batter", "game_date"], keep="first")
    out = out.drop(columns=PREGAME_FEATURES).merge(
        first[["batter", "game_date", *PREGAME_FEATURES]],
        on=["batter", "game_date"], how="left", validate="many_to_one",
    )
    return out


def prediction_features(raw: pd.DataFrame, day) -> pd.DataFrame:
    """Build target-day rolls, including the latest completed prior game."""
    history = raw.loc[pd.to_datetime(raw["game_date"]).dt.date < pd.Timestamp(day).date()].copy()
    if history.empty:
        raise ValueError("No history strictly before target day")
    targets = pd.DataFrame({
        "batter": history["batter"].unique(),
        "game_date": pd.Timestamp(day).date(),
        "game_pk": -1,
        "events": None,
        "launch_speed": float("nan"),
        "launch_angle": float("nan"),
        "bb_type": None,
    })
    table = build_batter_games(pd.concat([history, targets], ignore_index=True))
    return table.loc[table["game_pk"].eq(-1), ["batter", *PREGAME_FEATURES]]


def build_dataset(input_csv: Path, output_parquet: Path) -> Path:
    """Read raw Statcast CSV and write the processed training parquet."""
    raw = pd.read_csv(input_csv)
    dataset = build_batter_games(raw)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_parquet, index=False)
    return output_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-safe HR training data")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_parquet", type=Path)
    args = parser.parse_args()
    output = build_dataset(args.input_csv, args.output_parquet)
    print(f"wrote={output}")


if __name__ == "__main__":
    main()
