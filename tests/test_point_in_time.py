import json
from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest

from daily_hr.daily_board import _lineup, build_board
from daily_hr.dataset import PREGAME_FEATURES, build_batter_games, prediction_features
from daily_hr.ingest import ingest
from daily_hr.weather import get_weather


def raw():
    return pd.DataFrame([
        {"game_date": day, "game_pk": pk, "batter": 1, "events": event,
         "launch_speed": speed, "launch_angle": 25, "bb_type": "fly_ball"}
        for day, pk, event, speed in [
            ("2026-09-01", 1, "field_out", 80),
            ("2026-09-02", 2, "home_run", 110),
            ("2026-09-02", 3, "home_run", 105),
            ("2026-09-03", 4, "field_out", 90),
        ]
    ])


def test_no_same_day_or_future_data_and_latest_game_included():
    source = raw()
    target = prediction_features(source, date(2026, 9, 3))
    assert target.iloc[0]["home_runs_last_5"] == pytest.approx(2 / 3)
    changed = source.copy()
    changed.loc[changed.game_date.eq("2026-09-03"), "launch_speed"] = 999
    changed.loc[changed.game_date.eq("2026-09-03"), "events"] = "home_run"
    pd.testing.assert_frame_equal(target, prediction_features(changed, date(2026, 9, 3)))
    historical = build_batter_games(source)
    actual = historical.loc[historical.game_pk.eq(4), PREGAME_FEATURES].reset_index(drop=True)
    pd.testing.assert_frame_equal(target[PREGAME_FEATURES].reset_index(drop=True), actual)


def test_doubleheader_cannot_see_first_game_results():
    table = build_batter_games(raw())
    first = table.loc[table.game_pk.eq(2), PREGAME_FEATURES].reset_index(drop=True)
    second = table.loc[table.game_pk.eq(3), PREGAME_FEATURES].reset_index(drop=True)
    pd.testing.assert_frame_equal(first, second)
    assert first.iloc[0]["home_runs_last_5"] == 0


def test_ingest_enforces_dates_even_if_provider_returns_extra(tmp_path):
    def download(start, end, path):
        raw().to_csv(path, index=False)
    with patch("daily_hr.ingest.download_statcast", side_effect=download):
        output = ingest(date(2026, 9, 2), date(2026, 9, 2), tmp_path / "out.csv")
    assert set(pd.read_csv(output).game_date) == {"2026-09-02"}


def test_legacy_model_rejected_before_loading(tmp_path):
    source = tmp_path / "raw.csv"
    raw().to_csv(source, index=False)
    manifest = tmp_path / "features.json"
    manifest.write_text(json.dumps(["barrel_pct"]))
    with pytest.raises(ValueError, match="retrain"):
        build_board(source, tmp_path / "nonexistent.joblib", manifest, date(2026, 9, 3))


def test_confirmed_mode_never_uses_expected_lineup():
    game = {"gamePk": 1, "teams": {"home": {"team": {"name": "H"}}, "away": {"team": {"name": "A"}}}}
    with patch("daily_hr.daily_board._schedule_lineup", return_value=[]), patch(
        "daily_hr.daily_board._feed_lineup", return_value=[]
    ), patch("daily_hr.daily_board._expected_lineup", side_effect=AssertionError("fallback called")):
        assert _lineup(game, confirmed_only=True) == []


def test_confirmed_mode_requires_both_complete_lineups():
    game = {"gamePk": 1, "teams": {"home": {"team": {"name": "H"}}, "away": {"team": {"name": "A"}}}}
    rows = [{"team": t, "batter": i + offset} for t, offset in [("H", 0), ("A", 10)] for i in range(9)]
    with patch("daily_hr.daily_board._schedule_lineup", return_value=rows):
        assert len(_lineup(game, confirmed_only=True)) == 18
    with patch("daily_hr.daily_board._schedule_lineup", return_value=rows[:-1]):
        assert _lineup(game, confirmed_only=True) == []


def test_historical_weather_does_not_fetch_observations():
    with patch("daily_hr.weather._request", side_effect=AssertionError("observations fetched")):
        assert get_weather(date(2000, 1, 1), "Wrigley Field")["temperature_f"] is None


def test_training_selects_only_pregame_columns(tmp_path):
    from daily_hr.train import train_file
    data = pd.concat([build_batter_games(raw()).assign(game_date=f"2026-08-{d:02d}") for d in range(1, 21)])
    path = tmp_path / "training.parquet"
    data.to_parquet(path)
    class Model:
        def predict_proba(self, X):
            return [0.2] * len(X)
    def fit(X, y, Xcal, ycal):
        assert list(X.columns) == PREGAME_FEATURES
        assert list(Xcal.columns) == PREGAME_FEATURES
        return Model()
    with patch("daily_hr.train.fit_calibrated_model", side_effect=fit), patch("daily_hr.train.evaluate"):
        _, _, columns = train_file(path)
    assert columns == PREGAME_FEATURES
