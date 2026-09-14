"""Handedness-conditioned, regularized pregame matchup candidate.

Prior strengths are conservative starting values, not fitted performance claims.
HR rates use pitches; contact rates use measured batted balls.
"""
import pandas as pd

from .matchups import PITCH_TYPES

PRIOR_PITCHES = 200.0
PRIOR_CONTACTS = 30.0


def build_matchup_features(pitches, batter_id, pitcher_id, before):
    df = pitches.loc[pd.to_datetime(pitches.game_date) < pd.Timestamp(before)].copy()
    df = df[df.pitch_type.isin(PITCH_TYPES)].sort_values('game_date', kind='stable')
    empty = {k: float('nan') for k in (
        'pitch_matchup_hr_rate', 'pitch_matchup_hard_hit_rate',
        'pitch_matchup_barrel_rate', 'pitch_matchup_pitcher_hr_allowed_rate',
        'pitch_matchup_combined_risk')}
    empty['pitch_mix_coverage'] = 0.0
    if df.empty:
        return empty
    # Missing handedness is explicitly unavailable, never silently treated as opposite-hand.
    if not {'stand', 'p_throws'}.issubset(df.columns):
        return empty
    pitcher_history = df[df.pitcher == pitcher_id]
    hands = pitcher_history.p_throws.dropna()
    if hands.empty:
        return empty
    throws = hands.iloc[-1]
    batter_history = df[df.batter == batter_id]
    side_history = batter_history[batter_history.p_throws == throws].stand.dropna()
    if side_history.empty:
        sides = batter_history.stand.dropna().unique()
        if len(sides) != 1:
            return empty
        stand = sides[0]
    else:
        stand = side_history.iloc[-1]
    league = df[(df.p_throws == throws) & (df.stand == stand)].copy()
    if league.empty:
        return empty
    league['hr'] = league.events.eq('home_run').astype(float)
    speed = pd.to_numeric(league.launch_speed, errors='coerce')
    league['hard'] = speed.ge(95).astype(float).where(speed.notna())
    if 'launch_speed_angle' in league:
        angle = pd.to_numeric(league.launch_speed_angle, errors='coerce')
        league['barrel_metric'] = angle.eq(6).astype(float).where(angle.notna())
    else:
        league['barrel_metric'] = float('nan')
    bh = league[league.batter == batter_id]
    ph = league[league.pitcher == pitcher_id]
    # Smooth the handedness-specific mix toward this pitcher's overall mix.
    overall = pitcher_history.pitch_type.value_counts(normalize=True)
    counts = ph.pitch_type.value_counts()
    mix = (counts.reindex(overall.index, fill_value=0) + PRIOR_PITCHES * overall)
    mix = mix / mix.sum()

    def rate(sample, pool, metric, strength):
        prior = pool[metric].mean()
        if pd.isna(prior):
            return float('nan')
        values = sample[metric].dropna()
        return (values.sum() + strength * prior) / (len(values) + strength)

    values = {'hr': [], 'hard': [], 'barrel_metric': [], 'allowed': []}
    coverage = 0.0
    for pitch, usage in mix.items():
        pool = league[league.pitch_type == pitch]
        if pool.empty:
            pool = league
        b = bh[bh.pitch_type == pitch]
        p = ph[ph.pitch_type == pitch]
        coverage += usage * (len(b) / (len(b) + PRIOR_PITCHES))
        for metric in ('hr', 'hard', 'barrel_metric'):
            strength = PRIOR_PITCHES if metric == 'hr' else PRIOR_CONTACTS
            values[metric].append((usage, rate(b, pool, metric, strength)))
        values['allowed'].append((usage, rate(p, pool, 'hr', PRIOR_PITCHES)))

    def weighted(items):
        valid = [(w, v) for w, v in items if pd.notna(v)]
        return sum(w*v for w, v in valid) / sum(w for w, v in valid) if valid else float('nan')

    hr, allowed = weighted(values['hr']), weighted(values['allowed'])
    return {'pitch_matchup_hr_rate': hr,
            'pitch_matchup_pitcher_hr_allowed_rate': allowed,
            'pitch_matchup_combined_risk': hr * allowed,
            'pitch_matchup_hard_hit_rate': weighted(values['hard']),
            'pitch_matchup_barrel_rate': weighted(values['barrel_metric']),
            'pitch_mix_coverage': coverage}
