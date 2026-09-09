"""Central catalog of model features.

Keeping the feature names in one place makes training and daily scoring use
exactly the same columns and makes feature audits easier.
"""

BATTER_FEATURES = [
    "barrel_pct",
    "hard_hit_pct",
    "fly_ball_pct",
    "launch_angle_avg",
    "exit_velocity_avg",
    "pull_pct",
    "iso",
    "hr_per_pa",
    "xwoba",
    "xslg",
    "sweet_spot_pct",
    "pa_per_game",
]

PITCHER_FEATURES = [
    "pitcher_hr_per_pa_allowed",
    "pitcher_barrel_pct_allowed",
    "pitcher_hard_hit_pct_allowed",
    "pitcher_fly_ball_pct_allowed",
    "pitcher_xwoba_allowed",
    "pitcher_xslg_allowed",
    "pitcher_k_pct",
    "pitcher_bb_pct",
    "pitcher_fastball_usage",
    "pitcher_slider_usage",
    "pitcher_changeup_usage",
    "pitcher_curveball_usage",
    "pitcher_avg_velocity",
]

MATCHUP_FEATURES = [
    "handedness_matchup",
    "batter_barrel_vs_expected_pitch_mix",
    "batter_xslg_vs_expected_pitch_mix",
    "batter_hr_rate_vs_expected_pitch_mix",
]

CONTEXT_FEATURES = [
    "park_hr_factor",
    "temperature_f",
    "wind_out_mph",
    "air_density_proxy",
    "roof_open",
    "expected_lineup_slot",
    "expected_pa",
    "bullpen_hr_rate_allowed",
]

ROLLING_FEATURE_BASES = [
    "barrel_pct",
    "hard_hit_pct",
    "fly_ball_pct",
    "launch_angle_avg",
    "exit_velocity_avg",
    "iso",
    "hr_per_pa",
    "xwoba",
    "xslg",
]

ALL_FEATURES = BATTER_FEATURES + PITCHER_FEATURES + MATCHUP_FEATURES + CONTEXT_FEATURES
