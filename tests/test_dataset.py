import pandas as pd
import pytest

from daily_hr.dataset import build_batter_games


def test_training_features_exclude_current_game():
    rows = []
    for game_pk, game_date, exit_velocity, hr in [
        (1, "2026-04-01", 90, 0),
        (2, "2026-04-02", 100, 1),
        (3, "2026-04-03", 110, 1),
    ]:
        rows.append(
            {
                "game_date": game_date,
                "game_pk": game_pk,
                "batter": 7,
                "events": "home_run" if hr else "field_out",
                "launch_speed": exit_velocity,
                "launch_angle": 25,
                "bb_type": "fly_ball",
                "barrel": "true" if hr else "false",
            }
        )

    result = build_batter_games(pd.DataFrame(rows))
    third = result.iloc[2]

    assert third["hr_target"] == 1
    assert third["exit_velocity_avg_last_20"] == pytest.approx(95)
    assert third["home_runs_last_20"] == pytest.approx(0.5)
    assert third["hr_rate_change_5_vs_20"] == pytest.approx(0)
    assert third["barrel_rate_change_5_vs_20"] == pytest.approx(0)
    assert third["exit_velocity_change_5_vs_20"] == pytest.approx(0)
