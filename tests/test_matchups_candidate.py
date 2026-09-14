import pandas as pd
import pytest

from daily_hr.matchups_candidate import build_matchup_features


def history():
    return pd.DataFrame([
        dict(game_date='2026-09-01', batter=b, pitcher=p, stand=s,
             p_throws='L', pitch_type='FF', events=e, launch_speed=100,
             launch_speed_angle=6)
        for b, p, s, e in [(1, 10, 'L', 'home_run'), (2, 10, 'L', 'out')]
        + [(3, 10, 'R', 'out')] * 20
    ])


def test_small_sample_is_shrunk_and_handedness_isolated():
    df = history()
    result = build_matchup_features(df, 1, 10, '2026-09-02')
    assert result['pitch_matchup_hr_rate'] == pytest.approx(101 / 201)
    assert result['pitch_matchup_pitcher_hr_allowed_rate'] == .5
    changed = df.copy()
    changed.loc[changed.stand == 'R', 'events'] = 'home_run'
    assert build_matchup_features(changed, 1, 10, '2026-09-02') == result


def test_target_day_excluded():
    df = history()
    future = df.copy()
    future['game_date'] = '2026-09-02'
    future['events'] = 'home_run'
    assert build_matchup_features(pd.concat([df, future]), 1, 10, '2026-09-02') == build_matchup_features(df, 1, 10, '2026-09-02')


def test_contact_denominator_excludes_unmeasured_pitches():
    df = history()
    whiff = df.iloc[[0]].copy()
    whiff['launch_speed'] = None
    whiff['launch_speed_angle'] = None
    whiff['events'] = None
    result = build_matchup_features(pd.concat([df, whiff]), 1, 10, '2026-09-02')
    assert result['pitch_matchup_hard_hit_rate'] == 1
    assert result['pitch_matchup_barrel_rate'] == 1


def test_missing_handedness_is_unavailable():
    result = build_matchup_features(history().drop(columns='stand'), 1, 10, '2026-09-02')
    assert pd.isna(result['pitch_matchup_hr_rate'])
