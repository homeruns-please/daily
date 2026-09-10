"""Production re-ranking layer for the daily HR board."""

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

# The model probability remains important, but it no longer dominates the board.
# The production scorer explicitly rewards recent power and contact-quality upside.
WEIGHTS = {
    "baseline": 0.30,
    "pitcher_vulnerability": 0.25,
    "pitch_matchup": 0.15,
    "power_upside": 0.15,
    "park": 0.05,
    "weather": 0.05,
    "lineup": 0.05,
}


def _unit_rating(series: pd.Series) -> pd.Series:
    return ((pd.to_numeric(series, errors="coerce") - 1.0) / 9.0).clip(0.0, 1.0).fillna(0.5)


def _column_or_fallback(board: pd.DataFrame, column: str, fallback: float) -> pd.Series:
    if column in board:
        return board[column]
    return pd.Series(fallback, index=board.index, dtype=float)


def score_board(board: pd.DataFrame, *, weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Apply the production ranking to an already-built daily board."""
    weights = WEIGHTS if weights is None else weights
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
        raise ValueError(f"Board is missing production scorer columns: {sorted(missing)}")

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

    board["production_score"] = (
        weights["baseline"] * baseline
        + weights["pitcher_vulnerability"] * pitcher
        + weights["pitch_matchup"] * matchup
        + weights["power_upside"] * power
        + weights["park"] * park
        + weights["weather"] * weather
        + weights["lineup"] * lineup
    )
    board["ranking_score"] = board["production_score"]
    board = board.sort_values("production_score", ascending=False, kind="stable").reset_index(drop=True)
    board["model_rank"] = board.index + 1
    n = len(board)
    if n == 1:
        board["hr_rating"] = 10.0
    else:
        board["hr_rating"] = (10.0 - 9.0 * board.index / (n - 1)).round(1)
    board["rating"] = board["hr_rating"].round().map(LABELS)
    return board


def rerank_csv(input_path: Path, output_path: Path) -> pd.DataFrame:
    board = score_board(pd.read_csv(input_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    board.to_csv(output_path, index=False)
    return board


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Apply V1 HR ranking to a board CSV")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()
    rerank_csv(args.input_csv, args.output_csv)


if __name__ == "__main__":
    main()

