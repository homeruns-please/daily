import subprocess
import sys

import pandas as pd
import pytest

from daily_hr.production_scorer import WEIGHTS, score_board
from daily_hr.v2_scorer import WEIGHTS_V2, score_frame


def fixture():
    return pd.DataFrame({
        "batter_name": ["A", "B", "C"],
        "model_rank": [3, 1, 2],
        **{name: [1., 5.5, 10.] for name in [
            "baseline_rating", "pitcher_vulnerability_rating", "pitch_matchup_edge_rating",
            "recent_power_rating", "barrel_rating", "contact_quality_rating",
            "exit_velocity_rating", "park_rating", "weather_rating"]},
        "lineup_slot": [9, 5, 1],
    })


def test_exact_weights_and_scores():
    assert list(WEIGHTS.values()) == [.30, .25, .15, .15, .05, .05, .05]
    assert list(WEIGHTS_V2.values()) == [.25, .275, .175, .20, .025, .05, .025]
    board = fixture()
    original = board.copy(deep=True)
    for weights, scorer, column in [(WEIGHTS, score_board, "production_score"),
                                     (WEIGHTS_V2, score_frame, "v2_score")]:
        out = scorer(board)
        assert out[column].tolist() == pytest.approx([1., .5 * (1-weights["lineup"]) + 5/9 * weights["lineup"], weights["lineup"]/9])
        assert out.model_rank.tolist() == [1, 2, 3]
        assert out.rating.notna().all()
        assert not any("actual" in c for c in out.columns)
    pd.testing.assert_frame_equal(board, original)


def test_equal_weights_equal_mechanics(monkeypatch):
    import daily_hr.v2_scorer as v2
    monkeypatch.setattr(v2, "WEIGHTS_V2", WEIGHTS)
    board = fixture()
    board.loc[1, "barrel_rating"] = float("nan")
    board = board.drop(columns=["pitcher_vulnerability_rating"])
    first = score_board(board)
    second = v2.score_frame(board)
    assert first.production_score.tolist() == second.v2_score.tolist()
    assert first.batter_name.tolist() == second.batter_name.tolist()
    assert first.rating.tolist() == second.rating.tolist()


@pytest.mark.parametrize("module,column", [("production_scorer", "production_score"), ("v2_scorer", "v2_score")])
def test_cli_writes_board(tmp_path, module, column):
    source = tmp_path / "input.csv"
    output = tmp_path / module / "board.csv"
    fixture().to_csv(source, index=False)
    subprocess.run([sys.executable, "-m", "daily_hr." + module, str(source), str(output)], check=True)
    board = pd.read_csv(output)
    assert len(board) == 3
    assert board[column].is_monotonic_decreasing
    assert board.model_rank.tolist() == [1, 2, 3]
