"""Experimental V2 HR re-ranking layer.

V2 is intentionally separate from production_scorer.py so the original
production ranking remains unchanged and can be compared directly.

V2 changes:
- Baseline remains at 30% (down from the original 35%).
- Park is reduced from 10% to 7.5% so it acts as a modifier rather than a
  major driver.
- The freed 2.5% is split between pitch matchup and recent power upside.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


LABELS = {
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

# V2 is experimental and deliberately separate from production_scorer.WEIGHTS.
WEIGHTS_V2 = {
    "baseline": 0.30,
    "pitcher_vulnerability": 0.25,
    "pitch_matchup": 0.175,
    "power_upside": 0.175,
    "park": 0.075,
    "weather": 0.05,
    "lineup": 0.05,
}


def _unit_rating(series: pd.Series) -> pd.Series:
    return ((pd.to_numeric(series, errors="coerce") - 1.0) / 9.0).clip(0.0, 1.0).fillna(0.5)


def _column_or_fallback(board: pd.DataFrame, column: str, fallback: float) -> pd.Series:
    if column in board:
        return board[column]
    return pd.Series(fallback, index=board.index, dtype=float)


def score_board_v2(board: pd.DataFrame) -> pd.DataFrame:
    """Apply the experimental V2 ranking to an already-built daily board."""
    required = {
        "baseline_rating",
        "park_rating",
        "weather_rating",
        "lineup_slot",
        "recent_power_rating",
        "barrel_rating",
        "contact_quality_rating",
        "exit_velocity_rating",
    }
    missing = required - set(board.columns)
    if missing:
        raise ValueError(f"Board is missing V2 scorer columns: {sorted(missing)}")

    board = board.copy()
    baseline = _unit_rating(board["baseline_rating"])
    pitcher = _unit_rating(
        _column_or_fallback(
            board,
            "pitcher_vulnerability_rating",
            float(board.get("matchup_rating", pd.Series(5.5, index=board.index)).mean()),
        )
    )
    matchup = _unit_rating(_column_or_fallback(board, "pitch_matchup_edge_rating", 5.5))
    power = pd.concat(
        [
            _unit_rating(board["recent_power_rating"]),
            _unit_rating(board["barrel_rating"]),
            _unit_rating(board["contact_quality_rating"]),
            _unit_rating(board["exit_velocity_rating"]),
        ],
        axis=1,
    ).mean(axis=1)
    park = _unit_rating(board["park_rating"])
    weather = _unit_rating(board["weather_rating"])
    lineup = (10.0 - pd.to_numeric(board["lineup_slot"], errors="coerce")) / 9.0
    lineup = lineup.clip(0.0, 1.0).fillna(0.5)

    board["v2_production_score"] = (
        WEIGHTS_V2["baseline"] * baseline
        + WEIGHTS_V2["pitcher_vulnerability"] * pitcher
        + WEIGHTS_V2["pitch_matchup"] * matchup
        + WEIGHTS_V2["power_upside"] * power
        + WEIGHTS_V2["park"] * park
        + WEIGHTS_V2["weather"] * weather
        + WEIGHTS_V2["lineup"] * lineup
    )
    board["v2_ranking_score"] = board["v2_production_score"]
    board = board.sort_values("v2_production_score", ascending=False).reset_index(drop=True)
    board["v2_model_rank"] = board.index + 1
    n = len(board)
    if n == 1:
        board["v2_hr_rating"] = 10.0
    else:
        board["v2_hr_rating"] = (10.0 - 9.0 * board.index / (n - 1)).round(1)
    board["v2_rating"] = board["v2_hr_rating"].map(LABELS)
    return board


def rerank_csv_v2(input_path: Path, output_path: Path) -> pd.DataFrame:
    board = score_board_v2(pd.read_csv(input_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    board.to_csv(output_path, index=False)
    return board
