"""Build today's MLB HR board from the trained market-blind model."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import joblib
import pandas as pd
import requests

from .dataset import PREGAME_FEATURES, prediction_features
from .matchups import build_matchup_features
from .weather import get_weather

MLB_API = "https://statsapi.mlb.com/api/v1"

# Statcast HR park factors, 2024-2026 rolling values. 100 is neutral.
# Source: Baseball Savant Statcast Park Factors leaderboard.
PARK_HR_FACTORS = {
    "Angel Stadium": 101,
    "American Family Field": 88,
    "Busch Stadium": 70,
    "Chase Field": 213,
    "Citi Field": 101,
    "Citizens Bank Park": 115,
    "Comerica Park": 100,
    "Coors Field": 207,
    "Daikin Park": 111,
    "Fenway Park": 92,
    "Globe Life Field": 94,
    "Great American Ball Park": 81,
    "Kauffman Stadium": 168,
    "loanDepot park": 135,
    "Nationals Park": 105,
    "Oracle Park": 139,
    "Oriole Park at Camden Yards": 110,
    "Petco Park": 105,
    "PNC Park": 74,
    "Progressive Field": 60,
    "Rate Field": 69,
    "Rogers Centre": 69,
    "T-Mobile Park": 37,
    "Target Field": 113,
    "Tropicana Field": 107,
    "Truist Park": 100,
    "Wrigley Field": 107,
    "Yankee Stadium": 117,
    "UNIQLO Field at Dodger Stadium": 124,
}


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


def _lineup(game: dict, confirmed_only: bool = False) -> list[dict]:
    """Prefer schedule-hydrated/live lineups; otherwise use the recent expected lineup."""
    rows = _schedule_lineup(game) or _feed_lineup(game["gamePk"])
    if confirmed_only:
        # Require both complete announced lineups before scoring a game.
        teams = game.get("teams", {})
        names = [teams.get(side, {}).get("team", {}).get("name", "") for side in ("away", "home")]
        complete = all(
            len([r for r in rows if r["team"] == name]) == 9
            and len({r["batter"] for r in rows if r["team"] == name}) == 9
            for name in names
        )
        return rows if complete else []
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


def _rating_label(rating: float) -> str:
    """Return the public-facing qualitative label for a 1-10 HR rating."""
    labels = {
        10: "Elite",
        9: "Excellent",
        8: "Strong",
        7: "Above average",
        6: "Solid",
        5: "Neutral",
        4: "Below average",
        3: "Weak",
        2: "Poor",
        1: "Avoid",
    }
    return labels[round(rating)]


def _assign_ratings(board: pd.DataFrame) -> pd.DataFrame:
    """Convert the internal composite ranking into slate-relative ratings."""
    n = len(board)
    if n == 1:
        board["hr_rating"] = 10.0
    else:
        ranks = board["ranking_score"].rank(method="first", ascending=False)
        board["hr_rating"] = (10.0 - 9.0 * (ranks - 1) / (n - 1)).round(1)
    board["rating"] = board["hr_rating"].map(_rating_label)
    return board


def _add_matchup_context(board: pd.DataFrame, historical_raw: pd.DataFrame, day: date) -> pd.DataFrame:
    """Add separate batter-edge and pitcher-vulnerability matchup signals."""
    if "pitch_type" not in historical_raw.columns or "pitcher" not in historical_raw.columns:
        board["matchup_score"] = 0.5
        board["pitch_matchup_hr_rate"] = float("nan")
        board["pitch_matchup_pitcher_hr_allowed_rate"] = float("nan")
        board["baseline_rating"] = (1.0 + 9.0 * board["hr_probability"].rank(pct=True)).round(1)
        board["matchup_rating"] = 5.5
        return board

    before = pd.Timestamp(day)
    batter_edges: list[float] = []
    pitcher_vulnerability: list[float] = []
    combined: list[float] = []
    for row in board.itertuples(index=False):
        pitcher_id = getattr(row, "opposing_pitcher_id", None)
        if pd.isna(pitcher_id) or pitcher_id in (None, ""):
            batter_edges.append(float("nan"))
            pitcher_vulnerability.append(float("nan"))
            combined.append(float("nan"))
            continue
        try:
            matchup = build_matchup_features(
                historical_raw,
                int(row.batter),
                int(pitcher_id),
                before,
            )
        except (KeyError, TypeError, ValueError):
            batter_edges.append(float("nan"))
            pitcher_vulnerability.append(float("nan"))
            combined.append(float("nan"))
            continue
        batter_edges.append(matchup.get("pitch_matchup_hr_rate", float("nan")))
        pitcher_vulnerability.append(matchup.get("pitch_matchup_pitcher_hr_allowed_rate", float("nan")))
        combined.append(matchup.get("pitch_matchup_combined_risk", float("nan")))

    board["pitch_matchup_hr_rate"] = pd.to_numeric(batter_edges, errors="coerce")
    board["pitch_matchup_pitcher_hr_allowed_rate"] = pd.to_numeric(pitcher_vulnerability, errors="coerce")
    board["matchup_score"] = pd.to_numeric(combined, errors="coerce")
    baseline_rank = board["hr_probability"].rank(pct=True)
    board["baseline_rating"] = (1.0 + 9.0 * baseline_rank).round(1)
    board["matchup_rating"] = (
        1.0 + 9.0 * board["pitch_matchup_hr_rate"].rank(pct=True).fillna(0.5)
    ).round(1)
    return board


def _park_score(venue: str) -> float:
    """Return a bounded HR environment score from Statcast park factors."""
    factor = PARK_HR_FACTORS.get(venue)
    if factor is None:
        return 0.5
    return max(0.0, min(1.0, 0.5 + (factor - 100.0) / 200.0))


def _add_park_context(board: pd.DataFrame) -> pd.DataFrame:
    """Add ballpark HR context."""
    board["park_hr_factor"] = board["venue"].map(PARK_HR_FACTORS).fillna(100.0)
    board["park_score"] = board["venue"].map(_park_score)
    board["park_rating"] = (1.0 + 9.0 * board["park_score"]).round(1)
    return board


def _weather_score(weather: dict[str, float | None]) -> float:
    """Return a conservative 0-1 game-environment score."""
    temperature = weather.get("temperature_f")
    wind = weather.get("wind_mph")
    score = 0.5
    if temperature is not None:
        if 70 <= temperature <= 85:
            score += 0.10
        elif 60 <= temperature < 70 or 85 < temperature <= 92:
            score += 0.03
        elif temperature < 45 or temperature > 100:
            score -= 0.10
        else:
            score -= 0.03
    if wind is not None:
        score += min(wind, 15.0) / 15.0 * 0.05
    return max(0.0, min(1.0, score))


def _add_weather_context(board: pd.DataFrame, games: list[dict], day: date) -> pd.DataFrame:
    """Add a bounded game-time weather adjustment and display fields."""
    weather_by_game: dict[int, dict[str, float | None]] = {}
    for game in games:
        game_pk = int(game.get("gamePk", 0))
        venue = game.get("venue", {}).get("name")
        if not game_pk or not venue:
            continue
        try:
            game_time = datetime.fromisoformat(game["gameDate"])
            weather_by_game[game_pk] = get_weather(day, venue, game_time)
        except (KeyError, TypeError, ValueError, requests.RequestException):
            continue

    if not weather_by_game:
        board["weather_score"] = 0.5
        board["weather_rating"] = 5.5
        board["weather_temperature_f"] = pd.NA
        board["weather_wind_mph"] = pd.NA
        board["weather_wind_direction_deg"] = pd.NA
        return board

    scores: list[float] = []
    temperatures: list[float | None] = []
    winds: list[float | None] = []
    directions: list[float | None] = []
    for row in board.itertuples(index=False):
        weather = weather_by_game.get(int(getattr(row, "game_pk", 0)), {})
        scores.append(_weather_score(weather))
        temperatures.append(weather.get("temperature_f"))
        winds.append(weather.get("wind_mph"))
        directions.append(weather.get("wind_direction_deg"))

    board["weather_score"] = scores
    board["weather_rating"] = (1.0 + 9.0 * board["weather_score"]).round(1)
    board["weather_temperature_f"] = temperatures
    board["weather_wind_mph"] = winds
    board["weather_wind_direction_deg"] = directions
    return board


def _add_signal_breakdown(board: pd.DataFrame) -> pd.DataFrame:
    """Add transparent descriptive signals and the production ranking components."""
    def percentile(series: pd.Series) -> pd.Series:
        return series.rank(pct=True).fillna(0.5)

    components = {
        "recent_power_rating": percentile(board["hr_per_pa_last_5"]),
        "barrel_rating": percentile(board["barrel_pct_last_20"]),
        "contact_quality_rating": percentile(board["hard_hit_pct_last_20"]),
        "exit_velocity_rating": percentile(board["exit_velocity_avg_last_20"]),
    }
    for name, values in components.items():
        board[name] = (1.0 + 9.0 * values).round(1)

    baseline_rank = percentile(board["hr_probability"])
    pitcher_rank = percentile(board["pitch_matchup_pitcher_hr_allowed_rate"])
    batter_matchup_rank = percentile(board["pitch_matchup_hr_rate"])
    park_rank = percentile(board["park_score"])
    weather_rank = percentile(board["weather_score"])
    recent_rank = percentile(board["hr_per_pa_last_5"])
    lineup_rank = (10.0 - pd.to_numeric(board["lineup_slot"], errors="coerce").fillna(5.0)) / 9.0
    lineup_rank = lineup_rank.clip(0.0, 1.0)

    board["pitcher_vulnerability_rating"] = (1.0 + 9.0 * pitcher_rank).round(1)
    board["pitch_matchup_edge_rating"] = (1.0 + 9.0 * batter_matchup_rank).round(1)
    board["lineup_context_rating"] = (1.0 + 9.0 * lineup_rank).round(1)
    board["ranking_score"] = (
        0.35 * baseline_rank
        + 0.25 * pitcher_rank
        + 0.15 * batter_matchup_rank
        + 0.10 * park_rank
        + 0.05 * weather_rank
        + 0.05 * recent_rank
        + 0.05 * lineup_rank
    )

    def factor(row: pd.Series) -> str:
        labels = {
            "baseline_rating": "Hitter skill",
            "pitcher_vulnerability_rating": "Pitcher vulnerability",
            "pitch_matchup_edge_rating": "Pitch-type matchup",
            "park_rating": "Ballpark",
            "weather_rating": "Weather",
            "recent_power_rating": "Recent power",
            "lineup_context_rating": "Lineup context",
        }
        values = {key: row.get(key, 5.5) for key in labels}
        top = sorted(values, key=values.get, reverse=True)[:2]
        return " + ".join(labels[key] for key in top)

    board["key_factors"] = board.apply(factor, axis=1)
    return board


def build_board(raw_csv: Path, model_path: Path, features_path: Path, day: date,
                confirmed_only: bool = True) -> pd.DataFrame:
    raw = pd.read_csv(raw_csv)
    raw = raw.loc[pd.to_datetime(raw["game_date"], errors="raise").dt.date < day].copy()
    if raw.empty:
        raise ValueError("No history strictly before target day")
    features = json.loads(features_path.read_text())
    if set(features) != set(PREGAME_FEATURES) or len(features) != len(PREGAME_FEATURES):
        raise ValueError("Unsafe or obsolete model feature manifest; retrain with pregame-only features")
    model = joblib.load(model_path)
    latest = prediction_features(raw, day)

    candidates: list[dict] = []
    games = _today_games(day)
    eligible_games = []
    for game in games:
        if confirmed_only and (
            game.get("status", {}).get("abstractGameState") != "Preview"
            or datetime.fromisoformat(game["gameDate"]) <= datetime.now(UTC)
        ):
            continue
        teams = game.get("teams", {})
        away = teams.get("away", {})
        home = teams.get("home", {})
        away_name = away.get("team", {}).get("name", "")
        away_pitcher = away.get("probablePitcher", {}).get("fullName", "TBD")
        home_pitcher = home.get("probablePitcher", {}).get("fullName", "TBD")
        away_pitcher_id = away.get("probablePitcher", {}).get("id")
        home_pitcher_id = home.get("probablePitcher", {}).get("id")
        if confirmed_only and (not away_pitcher_id or not home_pitcher_id):
            continue
        lineup = _lineup(game, confirmed_only=confirmed_only)
        if lineup:
            eligible_games.append(game)
        for row in lineup:
            row["game_time"] = game.get("gameDate", "")
            row["game_pk"] = game.get("gamePk")
            row["venue"] = game.get("venue", {}).get("name", "")
            row["opposing_pitcher"] = home_pitcher if row["team"] == away_name else away_pitcher
            row["opposing_pitcher_id"] = home_pitcher_id if row["team"] == away_name else away_pitcher_id
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
    board = _add_matchup_context(board, raw, day)
    board = _add_park_context(board)
    board = _add_weather_context(board, eligible_games, day)
    board = _add_signal_breakdown(board)
    board = board.sort_values("ranking_score", ascending=False).reset_index(drop=True)
    board["model_rank"] = board.index + 1
    board["target_date"] = day.isoformat()
    board["generated_at_utc"] = datetime.now(UTC).isoformat()
    board["history_through"] = str(pd.to_datetime(raw["game_date"]).max().date())
    board["board_mode"] = "confirmed_pregame" if confirmed_only else "projected_or_reconstructed"
    board["weather_source"] = "unavailable_historical_forecast" if day < datetime.now(UTC).date() else "forecast_or_neutral"
    board.attrs["coverage"] = {"scheduled_games": len(games), "included_games": len(eligible_games),
                               "omitted_game_ids": [g["gamePk"] for g in games if g not in eligible_games]}
    return _assign_ratings(board)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build today's HR rating board")
    parser.add_argument("raw_csv", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("features", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--date", default=datetime.now(UTC).date().isoformat())
    parser.add_argument("--allow-projected", action="store_true",
                        help="Explicitly allow projected/reconstructed lineups; not a confirmed pregame board")
    args = parser.parse_args()
    board = build_board(
        raw_csv=args.raw_csv,
        model_path=args.model,
        features_path=args.features,
        day=date.fromisoformat(args.date),
        confirmed_only=not args.allow_projected,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "target_date", "generated_at_utc", "history_through", "board_mode", "weather_source",
        "batter", "game_pk", "opposing_pitcher_id",
        "model_rank",
        "batter_name",
        "team",
        "opponent",
        "opposing_pitcher",
        "lineup_slot",
        "venue",
        "park_hr_factor",
        "hr_rating",
        "rating",
        "baseline_rating",
        "matchup_rating",
        "pitcher_vulnerability_rating",
        "pitch_matchup_edge_rating",
        "park_rating",
        "weather_rating",
        "lineup_context_rating",
        "weather_temperature_f",
        "weather_wind_mph",
        "weather_wind_direction_deg",
        "recent_power_rating",
        "barrel_rating",
        "contact_quality_rating",
        "exit_velocity_rating",
        "key_factors",
        "expected_lineup",
    ]
    board[columns].to_csv(args.output, index=False)
    args.output.with_suffix(".coverage.json").write_text(json.dumps(board.attrs.get("coverage", {}), indent=2))


if __name__ == "__main__":
    main()
