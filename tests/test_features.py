import pandas as pd

from daily_hr.features import add_rolling_mean


def test_rolling_feature_does_not_use_current_game():
    df = pd.DataFrame(
        {
            "player_id": [1, 1, 1],
            "game_date": pd.to_datetime(["2026-04-01", "2026-04-02", "2026-04-03"]),
            "barrel_pct": [0.10, 0.20, 0.90],
        }
    )

    result = add_rolling_mean(df, ["player_id"], "barrel_pct", windows=(2,))

    assert pd.isna(result.iloc[0]["barrel_pct_rolling_2"])
    assert result.iloc[1]["barrel_pct_rolling_2"] == 0.10
    assert result.iloc[2]["barrel_pct_rolling_2"] == 0.15
