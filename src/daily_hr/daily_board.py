"""Build today's MLB HR board from the trained market-blind model."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import requests

from .dataset import build_batter_games

MLB_API = "https://statsapi.mlb.com/api/v1"


def _today_games(day: date) -> list[dict]:
    response = requests.get(
        f"{MLB_API}/schedule",
        params={"sportId": 1, "date": day.isoformat(), "hydrate": "probablePitcher"},
        timeout=30,
    )
    response.raise_for_status()
    return [game for date_block in response.json().get("dates", []) for game in date_block["games"]]


def _lineup(game_pk: int) -> list[dict]:
    response = requests.get(f"{MLB_API}/game/{game_pk}/feed/live", timeout=30)
    response.raise_for_status()
    data = response.json()
    teams = data.get("liveData", {}).get("boxscore", {}).get("teams", {})
    rows: list[dict] = []
    for side in ("away", "home"):
        team = teams.get(side, {})
        team_name = team.get("team", {}).get("name", side.title())
        for slot, player_id in enumerate(team.get("battingOrder", []), start=1):
            player = team.get("players", {}).get(f"ID{player_id}", {})
            person = player.get("person", {})
            rows.append(
                {
                    "batter": int(player_id),
                    "batter_name": person.get("fullName", f"Player {player_id}"),
                    "team": team_name,
                    "lineup_slot": slot,
                    "opponent": teams.get("home" if side == "away" else "away", {})
                    .get("team", {})
                    .get("name", ""),
                }
            )
    return rows


def build_board(raw_csv: Path, model_path: Path, day: date) -> pd.DataFrame:
    raw = pd.read_csv(raw_csv)
    historical = build_batter_games(raw)
    model = joblib.load(model_path)
    features = model.features if hasattr(model, "features") else None
    if features is None:
        raise ValueError("Saved model does not expose its feature list")

    latest = historical.sort_values(["batter", "game_date", "game_pk"]).groupby("batter").tail(1)
    games = _today_games(day)
    candidates: list[dict] = []
    for game in games:
        probable = {
            item.get("team", {}).get("id"): item.get("probablePitcher", {}).get("fullName")
            for item in (game.get("teams", {}).get("away", {}), game.get("teams", {}).get("home", {}))
        }
        rows = _lineup(game["gamePk"])
        for row in rows:
            row["game_time"] = game.get("gameDate", "")
            row["opposing_pitcher"] = probable.get(
                next(
                    (
                        team_id
                        for team_id, team in (
                            (item.get("id"), item) for item in []
                        )
                    ),
                    None,
                ),
                "TBD",
            )
            candidates.append(row)

    # Re-map probable pitchers by team name because the schedule response nests them by side.
    for game in games:
        teams = game.get("teams", {})
        away_name = teams.get("away", {}).get("team", {}).get("name", "")
        home_name = teams.get("home", {}).get("team", {}).get("name", "")
        away_pitcher = teams.get("away", {}).get("probablePitcher", {}).get("fullName", "TBD")
        home_pitcher = teams.get("home", {}).get("probablePitcher", {}).get("fullName", "TBD")
        for row in candidates:
            if row["team"] == away_name and row["opponent"] == home_name:
                row["opposing_pitcher"] = home_pitcher
            elif row["team"] == home_name and row["opponent"] == away_name:
                row["opposing_pitcher"] = away_pitcher

    board = pd.DataFrame(candidates)
    if board.empty:
        return board
    board = board.merge(latest[["batter", *features]], on="batter", how="left")
    X = board[features].replace([float("inf"), float("-inf")], pd.NA).fillna(0)
    board["hr_probability"] = model.predict_proba(X)
    board["model_rank"] = board["hr_probability"].rank(method="first", ascending=False).astype(int)
    return board.sort_values("hr_probability", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build today's HR probability board")
    parser.add_argument("raw_csv", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--date", default=date.today().isoformat())
    args = parser.parse_args()
    board = build_board(args.raw_csv, args.model, date.fromisoformat(args.date))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    board[
        [
            "model_rank",
            "batter_name",
            "team",
            "opponent",
            "opposing_pitcher",
            "lineup_slot",
            "hr_probability",
        ]
    ].to_csv(args.output, index=False)
    print(board.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
