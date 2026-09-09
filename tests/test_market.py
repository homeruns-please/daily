import pandas as pd
import pytest

from daily_hr.market import add_market_comparison, american_to_probability


def test_american_odds_conversion() -> None:
    assert american_to_probability(300) == pytest.approx(0.25)
    assert american_to_probability(-200) == pytest.approx(2 / 3)


def test_market_is_comparison_only() -> None:
    predictions = pd.DataFrame(
        {
            "batter": ["A"],
            "hr_probability": [0.20],
            "fanduel_odds": [400],
            "draftkings_odds": [500],
        }
    )
    result = add_market_comparison(predictions)

    assert result.loc[0, "fanduel_odds"] == 400
    assert result.loc[0, "draftkings_odds"] == 500
    assert result.loc[0, "best_market_implied"] == pytest.approx(1 / 6)
    assert result.loc[0, "model_edge"] == pytest.approx(0.20 - 1 / 6)
