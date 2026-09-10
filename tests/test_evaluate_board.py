import pandas as pd

from daily_hr.evaluate_board import evaluate_board


def test_evaluate_board_marks_actual_home_runs_without_changing_rank() -> None:
    board = pd.DataFrame(
        {
            "model_rank": [1, 2, 3],
            "batter_name": ["Alpha One", "Bravo Two", "Charlie Three"],
        }
    )

    result = evaluate_board(board, {"Bravo Two", "Charlie Three"})

    assert result["model_rank"].tolist() == [1, 2, 3]
    assert result["actual_hr"].tolist() == [False, True, True]
    assert result["actual_hr_rank"].tolist() == [pd.NA, 1, 2]
