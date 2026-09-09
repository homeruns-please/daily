"""Data-source adapters for Statcast and the MLB Stats API."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests

STATCAST_URL = "https://baseballsavant.mlb.com/statcast_search/csv"
MLB_API_URL = "https://statsapi.mlb.com/api/v1"


@dataclass(frozen=True)
class DateRange:
    start: date
    end: date


def download_statcast(start: date, end: date, destination: Path) -> Path:
    """Download pitch-level Statcast data for an inclusive date range."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    params = {
        "all": "true",
        "hfGT": "R|",
        "game_date_gt": start.isoformat(),
        "game_date_lt": end.isoformat(),
        "group_by": "name",
        "min_pitches": 0,
        "min_results": 0,
        "player_type": "batter",
        "type": "details",
    }
    response = requests.get(STATCAST_URL, params=params, timeout=180)
    response.raise_for_status()
    output = destination.with_suffix(".csv")
    output.write_bytes(response.content)
    return output


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def save_parquet(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def get_schedule(game_date: date) -> list[dict[str, Any]]:
    """Return MLB games and probable pitchers for a date."""
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
    data = response.json()
    games: list[dict[str, Any]] = []
    for day in data.get("dates", []):
        for game in day.get("games", []):
            away = game["teams"]["away"]
            home = game["teams"]["home"]
            games.append(
                {
                    "game_pk": game.get("gamePk"),
                    "game_date": game_date.isoformat(),
                    "game_time": game.get("gameDate"),
                    "away_team": away["team"]["abbreviation"],
                    "home_team": home["team"]["abbreviation"],
                    "away_pitcher_id": (away.get("probablePitcher") or {}).get("id"),
                    "away_pitcher": (away.get("probablePitcher") or {}).get("fullName"),
                    "home_pitcher_id": (home.get("probablePitcher") or {}).get("id"),
                    "home_pitcher": (home.get("probablePitcher") or {}).get("fullName"),
                    "venue": (game.get("venue") or {}).get("name"),
                }
            )
    return games
