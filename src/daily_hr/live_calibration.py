"""Live same-day HR calibration for the remaining MLB slate."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .daily_board import MLB_API, build_board

LEADING_FEATURES = {
    "hr_per_pa_last_5": 1.0,
    "barrel_pct_last_20": 1.0,
    "hard_hit_pct_last_20": 0.8,
    "exit_velocity_avg_last_20": 0.8,
}
MAX_LIVE_ADJUSTMENT = 0.05
MIN_HR_HITTERS = 2
SHRINKAGE_PRIOR = 4.0


def _live_games(day: date) -> list[dict]:
    response = requests.get(
        f"{MLB_API}/schedule",
        params={"sportId": 1, "date": day.isoformat(), "hydrate": "linescore"},
        timeout=30,
    )
    response.raise_for_status()
    return [game for block in response.json().get("dates", []) for game in block["games"]]


def _hr_hitters(day: date) -> set[int]:
    """Return MLBAM batter IDs with a home run recorded so far today."""
    hitters: set[int] = set()
    for game in _live_games(day):
        if game.get("status", {}).get("abstractGameState") not in {"Live", "Final"}:
            continue
        game_pk = game.get("gamePk")
        if not game_pk:
            continue
        response = requests.get(f"{MLB_API}/game/{game_pk}/feed/live", timeout=30)
        if response.status_code == 404:
            continue
        response.raise_for_status()
        data = response.json().get("liveData", {})
        for play in data.get("plays", {}).get("allPlays", []):
            if play.get("result", {}).get("eventType") != "home_run":
                continue
            batter = play.get("matchup", {}).get("batter", {}).get("id")
            if batter:
                hitters.add(int(batter))
    return hitters


def _percentiles(board: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = pd.DataFrame(index=board.index)
    for column in columns:
        values = pd.to_numeric(board[column], errors="coerce")
        result[column] = values.rank(pct=True).fillna(0.5)
    return result


def apply_live_calibration(
    board: pd.DataFrame,
    hr_hitters: set[int],
    eligible_game_pks: set[int] | None = None,
) -> pd.DataFrame:
    """Adjust remaining-slate rankings from the observed same-day HR profile.

    Observed HR hitters are compared with the pregame board on leading power
    indicators. Empirical-Bayes shrinkage prevents a small early sample from
    swinging the slate, and the final adjustment is bounded to +/- 5%.
    """
    columns = [column for column in LEADING_FEATURES if column in board.columns]
    board["live_hr_count"] = len(hr_hitters)
    board["live_calibration_confidence"] = "None"
    board["live_calibration_score"] = 0.5
    board["live_calibration_adjustment"] = 0.0

    if len(hr_hitters) < MIN_HR_HITTERS or not columns:
        return board

    feature_pct = _percentiles(board, columns)
    observed = board[board["batter"].isin(hr_hitters)]
    if observed.empty:
        return board

    observed_pct = feature_pct.loc[observed.index]
    slate_mean = feature_pct.mean()
    hr_mean = observed_pct.mean()
    n = len(observed_pct)
    shrink = n / (n + SHRINKAGE_PRIOR)
    effect = (hr_mean - slate_mean) * shrink
    feature_weights = pd.Series(LEADING_FEATURES).reindex(columns)
    effect_strength = effect.abs() * feature_weights
    if effect_strength.sum() <= 0:
        return board

    weighted_effect = effect * feature_weights
    effect_total = float(effect_strength.sum())
    feature_signal = (feature_pct.sub(0.5) * weighted_effect).sum(axis=1) / effect_total
    feature_signal = feature_signal.clip(-0.5, 0.5)

    model_pct = board["hr_probability"].rank(pct=True).fillna(0.5)
    observed_model_pct = model_pct.loc[observed.index].mean()
    model_confirmation = np.clip(observed_model_pct - 0.5, -0.5, 0.5)
    live_signal = 0.75 * feature_signal + 0.25 * model_confirmation
    adjustment = (
        live_signal * MAX_LIVE_ADJUSTMENT * min(1.0, n / 5.0)
    ).clip(-MAX_LIVE_ADJUSTMENT, MAX_LIVE_ADJUSTMENT)

    if eligible_game_pks is None:
        eligible = pd.Series(True, index=board.index)
    else:
        eligible = board["game_pk"].isin(eligible_game_pks)
    already_hit = board["batter"].isin(hr_hitters)
    adjustment.loc[~eligible | already_hit] = 0.0

    board["live_calibration_score"] = (0.5 + live_signal).clip(0.0, 1.0).round(3)
    board["live_calibration_adjustment"] = adjustment.round(4)
    board["live_calibration_confidence"] = (
        "High" if n >= 6 else "Moderate" if n >= 3 else "Low"
    )
    board["ranking_score"] = board["ranking_score"] + adjustment
    board["live_hr_count"] = n
    return board


def build_live_board(
    raw_csv: Path,
    model_path: Path,
    features_path: Path,
    day: date,
) -> pd.DataFrame:
    """Build the normal board, then apply live same-day HR calibration."""
    board = build_board(raw_csv, model_path, features_path, day)
    board["baseline_ranking_score"] = board["ranking_score"]
    board["baseline_model_rank"] = (
        board["baseline_ranking_score"].rank(method="first", ascending=False).astype(int)
    )
    games = _live_games(day)
    hr_hitters = _hr_hitters(day)
    eligible_game_pks = {
        int(game["gamePk"])
        for game in games
        if game.get("status", {}).get("abstractGameState") != "Final"
        and game.get("gamePk")
    }
    board = apply_live_calibration(board, hr_hitters, eligible_game_pks)
    board = board.sort_values("ranking_score", ascending=False).reset_index(drop=True)
    board["model_rank"] = board.index + 1
    if len(board) > 1:
        ranks = board["ranking_score"].rank(method="first", ascending=False)
        board["hr_rating"] = (10.0 - 9.0 * (ranks - 1) / (len(board) - 1)).round(1)
    board["snapshot_at_utc"] = datetime.now(UTC).isoformat()
    board["live_eligible_game"] = board["game_pk"].isin(eligible_game_pks)
    return board


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build the live same-day HR calibration board")
    parser.add_argument("raw_csv", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("features", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--date", default=datetime.now(UTC).date().isoformat())
    args = parser.parse_args()
    board = build_live_board(
        args.raw_csv,
        args.model,
        args.features,
        date.fromisoformat(args.date),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    board.to_csv(args.output, index=False)
    columns = [
        "model_rank",
        "batter_name",
        "hr_rating",
        "live_hr_count",
        "live_calibration_adjustment",
    ]
    print(board[columns].head(25).to_string(index=False))


if __name__ == "__main__":
    main()
