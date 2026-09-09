import pandas as pd

from daily_hr.evaluate_live import evaluate_snapshot
from daily_hr.live_calibration import apply_live_calibration


def test_live_calibration_uses_observed_hr_profile_and_excludes_hr_hitters() -> None:
    board = pd.DataFrame(
        {
            "batter": [1, 2, 3, 4],
            "batter_name": ["HR One", "HR Two", "Match", "Other"],
            "hr_probability": [0.70, 0.65, 0.60, 0.30],
            "ranking_score": [0.90, 0.80, 0.70, 0.30],
            "hr_per_pa_last_5": [0.08, 0.07, 0.075, 0.01],
            "barrel_pct_last_20": [0.18, 0.17, 0.175, 0.03],
            "hard_hit_pct_last_20": [0.55, 0.52, 0.54, 0.25],
            "exit_velocity_avg_last_20": [93.0, 92.5, 92.8, 87.0],
        }
    )

    result = apply_live_calibration(board, {1, 2})

    assert result.loc[result["batter"] == 1, "live_calibration_adjustment"].item() == 0.0
    assert result.loc[result["batter"] == 2, "live_calibration_adjustment"].item() == 0.0
    assert result.loc[result["batter"] == 3, "live_calibration_adjustment"].item() > 0.0
    assert result.loc[result["batter"] == 4, "live_calibration_adjustment"].item() <= 0.0
    assert result["live_hr_count"].eq(2).all()
    assert result["live_calibration_confidence"].eq("Low").all()


def test_live_calibration_only_adjusts_eligible_games() -> None:
    board = pd.DataFrame(
        {
            "batter": [1, 2, 3],
            "game_pk": [100, 200, 300],
            "hr_probability": [0.70, 0.65, 0.30],
            "ranking_score": [0.90, 0.80, 0.30],
            "hr_per_pa_last_5": [0.08, 0.07, 0.01],
            "barrel_pct_last_20": [0.18, 0.17, 0.03],
            "hard_hit_pct_last_20": [0.55, 0.52, 0.25],
            "exit_velocity_avg_last_20": [93.0, 92.5, 87.0],
        }
    )

    result = apply_live_calibration(board, {1, 2}, {300})

    assert result.loc[result["batter"] == 1, "live_calibration_adjustment"].item() == 0.0
    assert result.loc[result["batter"] == 2, "live_calibration_adjustment"].item() == 0.0
    assert result.loc[result["batter"] == 3, "live_calibration_adjustment"].item() < 0.0


def test_live_snapshot_preserves_pregame_rank_columns() -> None:
    board = pd.DataFrame(
        {
            "batter": [1, 2, 3],
            "game_pk": [100, 100, 200],
            "hr_probability": [0.70, 0.65, 0.30],
            "ranking_score": [0.90, 0.80, 0.30],
            "hr_per_pa_last_5": [0.08, 0.07, 0.01],
            "barrel_pct_last_20": [0.18, 0.17, 0.03],
            "hard_hit_pct_last_20": [0.55, 0.52, 0.25],
            "exit_velocity_avg_last_20": [93.0, 92.5, 87.0],
        }
    )

    baseline = board["ranking_score"].rank(method="first", ascending=False).astype(int)
    result = apply_live_calibration(board, {1, 2}, {200})

    result["baseline_model_rank"] = baseline
    assert result["baseline_model_rank"].tolist() == [1, 2, 3]
    assert result["ranking_score"].tolist() == [0.90, 0.80, result.loc[2, "ranking_score"]]


def test_evaluate_snapshot_handles_csv_booleans_and_game_specific_hr() -> None:
    snapshot = pd.DataFrame(
        {
            "batter": [1, 2, 3, 4],
            "game_pk": [100, 100, 200, 200],
            "live_eligible_game": ["True", "True", "False", "True"],
            "baseline_model_rank": [1, 2, 1, 3],
            "model_rank": [1, 3, 1, 2],
        }
    )

    metrics = evaluate_snapshot(snapshot, {100: {2}, 200: {1}})

    assert metrics["eligible_players"] == 3
    assert metrics["actual_hr_hitters"] == 1
    assert metrics["baseline_top10_hr"] == 1
    assert metrics["live_top10_hr"] == 1
    assert metrics["baseline_avg_hr_rank"] == 2.0
    assert metrics["live_avg_hr_rank"] == 3.0
    assert metrics["avg_rank_improvement"] == -1.0
