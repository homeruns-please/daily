"""Build today's MLB HR board from the trained market-blind model."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import joblib
import pandas as pd
import requests

from .dataset import build_batter_games

MLB_API = "https://statsapi.mlb.com/api/v1"


def _today_games(day: date) -> list[dict]:
    response = requests.get(
        f"{MLB_API}/schedule",
        params={
            "sportId": 1,
            "date": day.isoformat(),
            "hydrate": "probablePitcher,lineups",
        },
        timeout=30,
    )
    response.raise_for_status()
    return [game for block in response.json().get("dates", []) for game in block["games"]]


def _schedule_lineup(game: dict) -> list[dict]:
    """Read announced lineups directly from the schedule hydration."""
    lineups = game.get("lineups", {})
    if not isinstance(lineups, dict):
        return []
    rows: list[dict] = []
    teams = game.get("teams", {})
    for side in ("away", "home"):
        team_name = teams.get(side, {}).get("team", {}).get("name", "")
        opponent = teams.get("home" if side == "away" else "away", {}).get("team", {}).get("name", "")
        players = lineups.get(f"{side}Players", [])
        for slot, player in enumerate(players, start=1):
            if not isinstance(player, dict) or not player.get("id"):
                continue
            rows.append(
                {
                    "batter": int(player["id"]),
                    "batter_name": player.get("fullName", f"Player {player['id']}"),
                    "team": team_name,
                    "opponent": opponent,
                    "lineup_slot": slot,
                    "expected_lineup": False,
                }
            )
    return rows


def _feed_lineup(game_pk: int) -> list[dict]:
    response = requests.get(f"{MLB_API}/game/{game_pk}/feed/live", timeout=30)
    if response.status_code == 404:
        return []
    response.raise_for_status()
    teams = response.json().get("liveData", {}).get("boxscore", {}).get("teams", {})
    rows: list[dict] = []
    for side in ("away", "home"):
        team = teams.get(side, {})
        team_name = team.get("team", {}).get("name", side.title())
        opponent = teams.get("home" if side == "away" else "away", {}).get("team", {}).get("name", "")
        for slot, player_id in enumerate(team.get("battingOrder", []), start=1):
            player = team.get("players", {}).get(f"ID{player_id}", {})
            person = player.get("person", {})
            rows.append(
                {
                    "batter": int(player_id),
                    "batter_name": person.get("fullName", f"Player {player_id}"),
                    "team": team_name,
                    "opponent": opponent,
                    "lineup_slot": slot,
                    "expected_lineup": False,
                }
            )
    return rows


def _expected_lineup(team_name: str, day: date) -> list[dict]:
    """Use the team's most recent announced batting order as an expected lineup."""
    for days_back in range(1, 8):
        games = _today_games(day - timedelta(days=days_back))
        for game in games:
            teams = game.get("teams", {})
            for side in ("away", "home"):
                team = teams.get(side, {}).get("team", {})
                if team.get("name") != team_name:
                    continue
                rows = _schedule_lineup(game) or _feed_lineup(game.get("gamePk", 0))
                team_rows = [row for row in rows if row["team"] == team_name]
                if team_rows:
                    for row in team_rows:
                        row["expected_lineup"] = True
                    return team_rows
    return []


def _lineup(game: dict) -> list[dict]:
    """Prefer schedule-hydrated/live lineups; otherwise use the recent expected lineup."""
    rows = _schedule_lineup(game) or _feed_lineup(game["gamePk"])
    if rows:
        return rows
    teams = game.get("teams", {})
    rows = []
    for side in ("away", "home"):
        team_name = teams.get(side, {}).get("team", {}).get("name", "")
        opponent = teams.get("home" if side == "away" else "away", {}).get("team", {}).get("name", "")
        for row in _expected_lineup(team_name, date.fromisoformat(game["gameDate"][:10])):
            row["opponent"] = opponent
            rows.append(row)
    return rows


def build_board(raw_csv: Path, model_path: Path, features_path: Path, day: date) -> pd.DataFrame:
    raw = pd.read_csv(raw_csv)
    historical = build_batter_games(raw)
    model = joblib.load(model_path)
    features = json.loads(features_path.read_text())
    latest = historical.sort_values(["batter", "game_date", "game_pk"]).groupby("batter").tail(1)

    candidates: list[dict] = []
    for game in _today_games(day):
        teams = game.get("teams", {})
        away = teams.get("away", {})
        home = teams.get("home", {})
        away_name = away.get("team", {}).get("name", "")
        away_pitcher = away.get("probablePitcher", {}).get("fullName", "TBD")
        home_pitcher = home.get("probablePitcher", {}).get("fullName", "TBD")
        for row in _lineup(game):
            row["game_time"] = game.get("gameDate", "")
            row["opposing_pitcher"] = home_pitcher if row["team"] == away_name else away_pitcher
            candidates.append(row)

    board = pd.DataFrame(candidates)
    if board.empty:
        raise RuntimeError(
            f"No lineup candidates were returned for {day.isoformat()}; "
            "MLB schedule/live lineup data is unavailable."
        )
    board = board.merge(latest[["batter", *features]], on="batter", how="left")
    X = board[features].replace([float("inf"), float("-inf")], pd.NA).fillna(0)
    board["hr_probability"] = model.predict_proba(X)
    board["model_rank"] = board["hr_probability"].rank(method="first", ascending=False).astype(int)
    return board.sort_values("hr_probability", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build today's HR probability board")
    parser.add_argument("raw_csv", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("features", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--date", default=datetime.now(UTC).date().isoformat())
    args = parser.parse_args()
    board = build_board(
        raw_csv=args.raw_csv,
        model_path=args.model,
        features_path=args.features,
        day=date.fromisoformat(args.date),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "model_rank",
        "batter_name",
        "team",
        "opponent",
        "opposing_pitcher",
        "lineup_slot",
        "hr_probability",
        "expected_lineup",
    ]
    board[columns].to_csv(args.output, index=False)
    print(board[columns].head(25).to_string(index=False))


if __name__ == "__main__":
    main()
