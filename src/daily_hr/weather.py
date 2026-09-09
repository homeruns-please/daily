"""Weather adapter for MLB game-time context."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import requests

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"


# Approximate park coordinates. Kept here so weather is reproducible and
# independent of third-party venue geocoding at prediction time.
PARK_COORDS: dict[str, tuple[float, float]] = {
    "Comerica Park": (42.3390, -83.0485),
    "Sutter Health Park": (38.5800, -121.5138),
    "Oracle Park": (37.7786, -122.3893),
    "Petco Park": (32.7073, -117.1573),
    "T-Mobile Park": (47.5914, -122.3325),
    "Oriole Park at Camden Yards": (39.2839, -76.6219),
    "Citizens Bank Park": (39.9061, -75.1665),
    "loanDepot park": (25.7781, -80.2197),
    "Fenway Park": (42.3467, -71.0972),
    "Yankee Stadium": (40.8296, -73.9262),
    "Truist Park": (33.8907, -84.4677),
    "American Family Field": (43.0280, -87.9712),
    "Kauffman Stadium": (39.0517, -94.4803),
    "Guaranteed Rate Field": (41.8300, -87.6338),
    "Dodger Stadium": (34.0739, -118.2400),
    "Wrigley Field": (41.9484, -87.6553),
    "Minute Maid Park": (29.7573, -95.3555),
    "Globe Life Field": (32.7473, -97.0847),
    "Target Field": (44.9817, -93.2776),
    "Progressive Field": (41.4962, -81.6852),
    "Rogers Centre": (43.6414, -79.3894),
    "Nationals Park": (38.8730, -77.0074),
    "Busch Stadium": (38.6226, -90.1928),
    "Tropicana Field": (27.7683, -82.6534),
    "Chase Field": (33.4453, -112.0667),
    "Citi Field": (40.7571, -73.8458),
    "Great American Ball Park": (39.0975, -84.5060),
    "PNC Park": (40.4469, -80.0057),
    "Coors Field": (39.7559, -104.9942),
}


def _request(url: str, game_date: date, latitude: float, longitude: float) -> dict[str, Any]:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": game_date.isoformat(),
        "end_date": game_date.isoformat(),
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,pressure_msl",
        "timezone": "UTC",
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def get_weather(game_date: date, venue: str) -> dict[str, float | None]:
    """Return daily hourly weather series for a ballpark.

    The caller can select the hour closest to first pitch. Historical dates use
    Open-Meteo's archive endpoint; current/future dates use its forecast API.
    """
    if venue not in PARK_COORDS:
        raise ValueError(f"Unknown venue coordinates: {venue}")
    lat, lon = PARK_COORDS[venue]
    url = (
        OPEN_METEO_ARCHIVE
        if game_date < datetime.now(timezone.utc).date()
        else OPEN_METEO_FORECAST
    )
    payload = _request(url, game_date, lat, lon)
    hourly = payload.get("hourly", {})
    return {
        "latitude": lat,
        "longitude": lon,
        "temperature_f": float(hourly.get("temperature_2m", [None])[0] * 9 / 5 + 32)
        if hourly.get("temperature_2m") and hourly["temperature_2m"][0] is not None
        else None,
        "wind_mph": float(hourly.get("wind_speed_10m", [None])[0] * 0.621371)
        if hourly.get("wind_speed_10m") and hourly["wind_speed_10m"][0] is not None
        else None,
        "wind_direction_deg": float(hourly.get("wind_direction_10m", [None])[0])
        if hourly.get("wind_direction_10m") and hourly["wind_direction_10m"][0] is not None
        else None,
        "relative_humidity": float(hourly.get("relative_humidity_2m", [None])[0])
        if hourly.get("relative_humidity_2m") and hourly["relative_humidity_2m"][0] is not None
        else None,
        "pressure_msl_hpa": float(hourly.get("pressure_msl", [None])[0])
        if hourly.get("pressure_msl") and hourly["pressure_msl"][0] is not None
        else None,
    }
