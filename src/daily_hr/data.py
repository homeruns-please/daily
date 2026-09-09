"""Data-source adapters and normalized schemas.

Raw provider-specific fields should be mapped into these stable tables before
feature engineering. Keeping ingestion separate makes provider changes much
safer and keeps the model code provider-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import requests

STATCAST_URL = "https://baseballsavant.mlb.com/leaderboard/statcast-search-csv"
MLB_API_URL = "https://statsapi.mlb.com/api/v1"


@dataclass(frozen=True)
class DateRange:
    start: date
    end: date


def download_statcast(start: date, end: date, destination: Path) -> Path:
    """Download Statcast search data for a date range and save as Parquet.

    The endpoint is intentionally isolated here; downstream code consumes the
    saved normalized data rather than depending on an external API at runtime.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    params = {
        "type": "details",
        "player_type": "batter",
        "game_date_gt": start.isoformat(),
        "game_date_lt": end.isoformat(),
        "hfSea": "",
        "hfPT": "",
        "hfBBT": "",
        "hfPR": "",
        "hfZ": "",
        "hfStadium": "",
        "hfBBL": "",
        "hfNewZones": "",
        "hfPull": "",
        "hfC": "",
        "hfSea": "",
        "group_by": "name-year",
        "min_pitches": 0,
        "min_results": 0,
        "type": "details",
    }
    response = requests.get(STATCAST_URL, params=params, timeout=120)
    response.raise_for_status()
    destination.with_suffix(".csv").write_bytes(response.content)
    return destination.with_suffix(".csv")


def load_csv(path: Path) -> pd.DataFrame:
    """Load a downloaded CSV into a dataframe."""
    return pd.read_csv(path)


def save_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Persist an intermediate/processed dataframe as Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path
