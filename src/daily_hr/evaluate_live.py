"""Evaluate a saved intraday HR calibration snapshot against final HR outcomes."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from .daily_board import MLB_API


def _final_hr_hitters(day: date) -> dict[int, set[int]]:
    """Return final HR batter IDs grouped by gamePk."""
    response = requests.get(
        f"{MLB_API}/schedule",
        params={"sportId": 1, "date": day.isoformat()},
        timeout=30,
    )
    response.raise_for_status()
    result: dict[int, set[int]] = {}
    for block in response.json().get("dates", []):
        for game in block.get("games", []):
            game_pk = game.get("gamePk")
            if not game_pk or game.get("status", {}).get("abstractGameState") != "Final":
                continue
            feed = requests.get(f"{MLB_API}/game/{game_pk}/feed/live", timeout=30)
            if feed.status_code == 404:
                continue
            feed.raise_for_status()
            hitters: set[int] = set()
            for play in feed.json().get("liveData", {}).get("plays", {}).get("allPlays", []):
                if play.get("result", {}).get("eventType") != "home_run":
                    continue
                batter = play.get("matchup", {}).get("batter", {}).get("id")
                if batter:
                    hitters.add(int(batter))
            result[int(game_pk)] = hitters
    return result


def evaluate_snapshot(
    snapshot: pd.DataFrame,
    final_hr_by_game: dict[int, set[int]],
) -> dict[str, float | int]:
    """Measure baseline and live rank capture for games still active at snapshot time."""
    eligible = snapshot[snapshot["live_eligible_game"].astype(bool)].copy()
    if eligible.empty:
        return {"eligible_players": 0, "actual_hr_hitters": 0}

    actual_hr: set[int] = set()
    for game_pk in eligible["game_pk"].dropna().astype(int).unique():
        actual_hr.update(final_hr_by_game.get(game_pk, set()))

    if not actual_hr:
        return {"eligible_players": int(len(eligible)), "actual_hr_hitters": 0}

    actual = eligible[eligible["batter"].isin(actual_hr)].copy()
    top10_baseline = set(eligible.nsmallest(10, "baseline_model_rank")["batter"])
    top20_baseline = set(eligible.nsmallest(20, "baseline_model_rank")["batter"])
    top10_live = set(eligible.nsmallest(10, "model_rank")["batter"])
    top20_live = set(eligible.nsmallest(20, "model_rank")["batter"])

    return {
        "eligible_players": int(len(eligible)),
        "actual_hr_hitters": int(len(actual)),
        "baseline_top10_hr": int(len(actual_hr.intersection(top10_baseline))),
        "live_top10_hr": int(len(actual_hr.intersection(top10_live))),
        "baseline_top20_hr": int(len(actual_hr.intersection(top20_baseline))),
        "live_top20_hr": int(len(actual_hr.intersection(top20_live))),
        "baseline_avg_hr_rank": float(actual["baseline_model_rank"].mean()),
        "live_avg_hr_rank": float(actual["model_rank"].mean()),
        "avg_rank_improvement": float(
            actual["baseline_model_rank"].mean() - actual["model_rank"].mean()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a live HR calibration snapshot")
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--date", required=True)
    args = parser.parse_args()

    snapshot = pd.read_csv(args.snapshot)
    metrics = evaluate_snapshot(snapshot, _final_hr_hitters(date.fromisoformat(args.date)))
    for key, value in metrics.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
