import pandas as pd

from daily_hr.production_scorer import score_board


def _board() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "batter_name": "Baseline Star",
                "baseline_rating": 10.0,
                "pitcher_vulnerability_rating": 4.0,
                "pitch_matchup_edge_rating": 4.0,
                "recent_power_rating": 3.0,
                "barrel_rating": 3.0,
                "contact_quality_rating": 3.0,
                "exit_velocity_rating": 3.0,
                "park_rating": 5.0,
                "weather_rating": 5.0,
                "lineup_slot": 2,
            },
            {
                "batter_name": "Power Upside",
                "baseline_rating": 6.0,
                "pitcher_vulnerability_rating": 9.0,
                "pitch_matchup_edge_rating": 9.0,
                "recent_power_rating": 10.0,
                "barrel_rating": 10.0,
                "contact_quality_rating": 10.0,
                "exit_velocity_rating": 10.0,
                "park_rating": 5.0,
                "weather_rating": 5.0,
                "lineup_slot": 3,
            },
        ]
    )


def test_power_upside_can_overcome_baseline_dominance() -> None:
    board = score_board(_board())
    assert board.iloc[0]["batter_name"] == "Power Upside"
    assert board.iloc[0]["model_rank"] == 1
    assert board.iloc[0]["rating"] == "Elite"


def test_scorer_preserves_small_lineup_context() -> None:
    board = _board()
    board.loc[1, "lineup_slot"] = 8
    scored = score_board(board)
    assert scored.iloc[0]["batter_name"] == "Power Upside"
