"""Canonical schemas used by the feature pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BatterGame:
    game_date: str
    game_id: int
    batter_id: int
    pitcher_id: int
    batter_hand: str | None
    pitcher_hand: str | None
    plate_appearances: int
    home_runs: int


@dataclass(frozen=True)
class PitchEvent:
    game_date: str
    game_id: int
    batter_id: int
    pitcher_id: int
    pitch_type: str | None
    launch_speed: float | None
    launch_angle: float | None
    barrel: int | None
    estimated_woba: float | None
    pitch_velocity: float | None
    spin_rate: float | None


@dataclass(frozen=True)
class GameContext:
    game_date: str
    game_id: int
    home_team: str
    away_team: str
    venue: str
    home_pitcher_id: int | None
    away_pitcher_id: int | None
