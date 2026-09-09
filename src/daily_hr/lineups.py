"""MLB game-feed lineup ingestion."""

from __future__ import annotations

from typing import Any

import requests

MLB_API_URL = "https://statsapi.mlb.com/api/v1"


def get_game_lineups(game_pk: int) -> list[dict[str, Any]]:
    """Return announced/confirmed batting orders from an MLB live feed.

    If lineups have not been posted yet, the returned list is empty. This keeps
    the daily scorer from silently treating projected lineups as confirmed.
    """
    response = requests.get(f"{MLB_API_URL}/game/{game_pk}/feed/live", timeout=30)
    response.raise_for_status()
    data = response.json()
    boxscore = data.get("liveData", {}).get("boxscore", {})
    teams = boxscore.get("teams", {})

    rows: list[dict[str, Any]] = []
    for side in ("away", "home"):
        team = teams.get(side, {})
        team_name = (team.get("team") or {}).get("abbreviation") or (team.get("team") or {}).get("name")
        batting_order = team.get("battingOrder", [])
        players = team.get("players", {})
        for slot, player_id in enumerate(batting_order, start=1):
            player = players.get(f"ID{player_id}", {})
            person = player.get("person", {})
            rows.append(
                {
                    "game_pk": game_pk,
                    "team": team_name,
                    "lineup_slot": slot,
                    "player_id": person.get("id", player_id),
                    "player_name": person.get("fullName"),
                    "bat_side": (player.get("batSide") or {}).get("code"),
                    "confirmed": True,
                }
            )
    return rows
