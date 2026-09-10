"""Evaluate a pregame HR board against final MLB home-run results."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from .daily_board import MLB_API


def _final_hr_names(day: date) -> set[str]:
    """Return batter names that hit a home run on a completed MLB slate."""
    response = requests.get(
        f"{MLB_API}/schedule",
        params={"sportId": 1, "date": day.isoformat()},
        timeout=30,
    )
    response.raise_for_status()
    hitters: set[str] = set()
    for block in response.json().get("dates", []):
        for game in block.get("games", []):
            if game.get("status", {}).get("abstractGameState") != "Final":
                continue
            game_pk = game.get("gamePk")
            if not game_pk:
                continue
            feed = requests.get(f"{MLB_API}/game/{game_pk}/feed/live", timeout=30)
            if feed.status_code == 404:
                continue
            feed.raise_for_status()
            payload = feed.json()
            players = payload.get("gameData", {}).get("players", {})
            for play in payload.get("liveData", {}).get("plays", {}).get("allPlays", []):
                if play.get("result", {}).get("eventType") != "home_run":
                    continue
                batter_id = play.get("matchup", {}).get("batter", {}).get("id")
                if batter_id:
                    player = players.get(f"ID{batter_id}", {})
                    if player.get("fullName"):
                        hitters.add(player["fullName"])
    return hitters


def evaluate_board(board: pd.DataFrame, actual_hr_names: set[str]) -> pd.DataFrame:
    """Annotate a pregame board with actual HR outcomes without changing model ranks."""
    result = board.copy()
    result["actual_hr"] = result["batter_name"].isin(actual_hr_names)
    result["actual_hr_rank"] = pd.NA
    hits = result[result["actual_hr"]].sort_values("model_rank")
    for rank, index in enumerate(hits.index, start=1):
        result.loc[index, "actual_hr_rank"] = rank
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a pregame HR board against final results")
    parser.add_argument("board", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--date", required=True)
    args = parser.parse_args()

    board = pd.read_csv(args.board)
    actual = _final_hr_names(date.fromisoformat(args.date))
    evaluated = evaluate_board(board, actual)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    evaluated.to_csv(args.output, index=False)

    print(f"actual_hr_hitters={len(actual)}")
    for cutoff in (10, 20, 50, 100):
        hits = int(evaluated.nsmallest(cutoff, "model_rank")["actual_hr"].sum())
        print(f"hits_at_{cutoff}={hits}")
    print("actual_hr_names=" + ", ".join(sorted(actual)))


if __name__ == "__main__":
    main()
