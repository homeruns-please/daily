"""Walk-forward evaluation of daily HR composite weights."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .daily_board import _add_matchup_context, _add_weather_context, _lineup, _today_games
from .dataset import build_batter_games
from .model import fit_calibrated_model

WEIGHTS = {
    "baseline_80_matchup_15_weather_5": (0.80, 0.15, 0.05),
    "baseline_85_matchup_10_weather_5": (0.85, 0.10, 0.05),
    "baseline_75_matchup_20_weather_5": (0.75, 0.20, 0.05),
    "baseline_80_matchup_10_weather_10": (0.80, 0.10, 0.10),
    "baseline_70_matchup_20_weather_10": (0.70, 0.20, 0.10),
    "baseline_only": (1.00, 0.00, 0.00),
}


def _training_cutoffs(history: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    dates = pd.to_datetime(history["game_date"])
    train_cut = dates.quantile(0.70)
    cal_cut = dates.quantile(0.85)
    return train_cut, cal_cut


def _build_day_board(
    raw: pd.DataFrame,
    history: pd.DataFrame,
    model,
    features: list[str],
    day: date,
) -> pd.DataFrame:
    games = _today_games(day)
    candidates: list[dict] = []
    for game in games:
        teams = game.get("teams", {})
        away = teams.get("away", {})
        home = teams.get("home", {})
        away_name = away.get("team", {}).get("name", "")
        away_pitcher = away.get("probablePitcher", {}).get("fullName", "TBD")
        home_pitcher = home.get("probablePitcher", {}).get("fullName", "TBD")
        away_pitcher_id = away.get("probablePitcher", {}).get("id")
        home_pitcher_id = home.get("probablePitcher", {}).get("id")
        for row in _lineup(game):
            row["game_time"] = game.get("gameDate", "")
            row["game_pk"] = game.get("gamePk")
            row["venue"] = game.get("venue", {}).get("name", "")
            row["opposing_pitcher"] = home_pitcher if row["team"] == away_name else away_pitcher
            row["opposing_pitcher_id"] = home_pitcher_id if row["team"] == away_name else away_pitcher_id
            candidates.append(row)
    board = pd.DataFrame(candidates)
    if board.empty:
        return board

    cutoff = pd.Timestamp(day)
    prior = history[pd.to_datetime(history["game_date"]) < cutoff]
    latest = prior.sort_values(["batter", "game_date", "game_pk"]).groupby("batter").tail(1)
    board = board.merge(latest[["batter", *features]], on="batter", how="left")
    X = board[features].replace([float("inf"), float("-inf")], pd.NA).fillna(0)
    board["hr_probability"] = model.predict_proba(X)
    board = _add_matchup_context(board, raw[raw["game_date"] < day.isoformat()], day)
    board = _add_weather_context(board, games, day)
    return board


def _score_weights(board: pd.DataFrame, weights: tuple[float, float, float]) -> pd.Series:
    baseline = board["hr_probability"].rank(pct=True)
    matchup = board["matchup_score"].rank(pct=True).fillna(0.5)
    weather = board["weather_score"].fillna(0.5)
    return weights[0] * baseline + weights[1] * matchup + weights[2] * weather


def _daily_metrics(board: pd.DataFrame, actuals: pd.Series, name: str) -> dict:
    ordered = board.assign(actual_hr=actuals).sort_values("ranking_score", ascending=False)
    slate_rate = ordered["actual_hr"].mean()
    row: dict[str, float | int | str] = {"weight_set": name, "players": len(ordered), "slate_hr_rate": slate_rate}
    for k in (5, 10, 20, 50):
        top = ordered.head(min(k, len(ordered)))
        row[f"top{k}_hr"] = int(top["actual_hr"].sum())
        row[f"top{k}_hr_rate"] = float(top["actual_hr"].mean()) if len(top) else 0.0
        row[f"top{k}_lift"] = float(top["actual_hr"].mean() / slate_rate) if slate_rate else np.nan
    return row


def run_backtest(raw_csv: Path, start: date, end: date) -> pd.DataFrame:
    raw = pd.read_csv(raw_csv)
    history = build_batter_games(raw)
    history_dates = pd.to_datetime(history["game_date"])
    rows: list[dict] = []
    for day_offset in range((end - start).days + 1):
        day = start + timedelta(days=day_offset)
        prior = history[history_dates < pd.Timestamp(day)].copy()
        if prior.empty:
            continue
        train_cut, cal_cut = _training_cutoffs(prior)
        dates = pd.to_datetime(prior["game_date"])
        excluded = {"game_date", "hr_target", "batter", "game_pk", "home_runs"}
        features = [c for c in prior.columns if c not in excluded and pd.api.types.is_numeric_dtype(prior[c])]
        X = prior[features].replace([float("inf"), float("-inf")], pd.NA).fillna(0)
        y = prior["hr_target"].astype(int)
        train = dates < train_cut
        cal = (dates >= train_cut) & (dates < cal_cut)
        if y.loc[train].nunique() < 2 or y.loc[cal].nunique() < 2:
            continue
        model = fit_calibrated_model(X.loc[train], y.loc[train], X.loc[cal], y.loc[cal])
        board = _build_day_board(raw, history, model, features, day)
        if board.empty:
            continue
        outcomes = history[history["game_date"] == pd.Timestamp(day)].set_index(["game_pk", "batter"])["hr_target"]
        actuals = pd.Series(
            [int(outcomes.get((int(row.game_pk), int(row.batter)), 0)) for row in board.itertuples(index=False)],
            index=board.index,
        )
        for name, weights in WEIGHTS.items():
            scored = board.copy()
            scored["ranking_score"] = _score_weights(scored, weights)
            rows.append(_daily_metrics(scored, actuals, name) | {"date": day.isoformat()})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a walk-forward HR composite backtest")
    parser.add_argument("raw_csv", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()
    result = run_backtest(args.raw_csv, date.fromisoformat(args.start), date.fromisoformat(args.end))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    summary = result.groupby("weight_set").agg(
        days=("date", "nunique"),
        slate_hr_rate=("slate_hr_rate", "mean"),
        top5_lift=("top5_lift", "mean"),
        top10_lift=("top10_lift", "mean"),
        top20_lift=("top20_lift", "mean"),
        top50_lift=("top50_lift", "mean"),
        top5_hr=("top5_hr", "sum"),
        top10_hr=("top10_hr", "sum"),
        top20_hr=("top20_hr", "sum"),
    ).sort_values("top10_lift", ascending=False)
    summary.to_csv(args.output.with_name(args.output.stem + "_summary.csv"))
    print(summary.to_string())


if __name__ == "__main__":
    main()
