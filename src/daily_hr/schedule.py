"""MLB schedule and probable-pitcher ingestion."""

from __future__ import annotations

from datetime import date
from typing import Any

import requests

from .schemas import GameContext

MLB_API_URL = "https://statsapi.mlb.com/api/v1"


def get_schedule(game_date: date) -> list[GameContext]:
    """Return normalized MLB games for a date, including probable pitchers."""
    response = requests.get(
        f"{MLB_API_URL}/schedule",
        params={
            "sportId": 1,
            "date": game_date.isoformat(),
            "hydrate": "team,probablePitcher,venue",
        },
        timeout=30,
    )
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    games: list[GameContext] = []

    for date_block in payload.get("dates", []):
        for game in date_block.get("games", []):
            teams = game.get("teams", {})
            away = teams.get("away", {})
            home = teams.get("home", {})
            away_team = away.get("team", {})
            home_team = home.get("team", {})
            away_pitcher = away.get("probablePitcher", {})
            home_pitcher = home.get("probablePitcher", {})
            venue = game.get("venue", {})

            games.append(
                GameContext(
                    game_date=game_date,
                    game_pk=game.get("gamePk"),
                    home_team=home_team.get("abbreviation") or home_team.get("name"),
                    away_team=away_team.get("abbreviation") or away_team.get("name"),
                    home_pitcher_id=home_pitcher.get("id"),
                    away_pitcher_id=away_pitcher.get("id"),
                    venue=venue.get("name"),
                )
            )

    return games
