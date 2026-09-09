"""Pitch-type matchup features built only from pre-game Statcast history."""

from __future__ import annotations

import pandas as pd

PITCH_TYPES = ("FF", "SI", "FC", "SL", "CH", "CU", "KC", "ST", "FS", "SV")


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & weights.gt(0)
    if not mask.any():
        return float("nan")
    return float((values[mask] * weights[mask]).sum() / weights[mask].sum())


def batter_pitch_type_rates(pitches: pd.DataFrame, before: pd.Timestamp) -> pd.DataFrame:
    """Return batter HR/contact-quality rates by pitch type before ``before``."""
    df = pitches[pd.to_datetime(pitches["game_date"]) < pd.Timestamp(before)].copy()
    df = df[df["pitch_type"].isin(PITCH_TYPES)]
    df["is_hr"] = df["events"].eq("home_run").astype(float)
    df["is_hard_hit"] = pd.to_numeric(df["launch_speed"], errors="coerce").ge(95).astype(float)
    barrel = df["barrel"] if "barrel" in df.columns else pd.Series("false", index=df.index)
    df["is_barrel"] = barrel.astype(str).str.lower().eq("true").astype(float)
    return df.groupby(["batter", "pitch_type"], as_index=False).agg(
        batter_pitch_rate=("pitch_type", "size"),
        batter_hr_rate=("is_hr", "mean"),
        batter_hard_hit_rate=("is_hard_hit", "mean"),
        batter_barrel_rate=("is_barrel", "mean"),
    )


def pitcher_pitch_rates(pitches: pd.DataFrame, before: pd.Timestamp) -> pd.DataFrame:
    """Return pitcher HR/contact allowed by pitch type before ``before``."""
    df = pitches[pd.to_datetime(pitches["game_date"]) < pd.Timestamp(before)].copy()
    df = df[df["pitch_type"].isin(PITCH_TYPES)]
    df["is_hr"] = df["events"].eq("home_run").astype(float)
    df["is_hard_hit"] = pd.to_numeric(df["launch_speed"], errors="coerce").ge(95).astype(float)
    barrel = df["barrel"] if "barrel" in df.columns else pd.Series("false", index=df.index)
    df["is_barrel"] = barrel.astype(str).str.lower().eq("true").astype(float)
    return df.groupby(["pitcher", "pitch_type"], as_index=False).agg(
        pitcher_pitch_rate=("pitch_type", "size"),
        pitcher_hr_allowed_rate=("is_hr", "mean"),
        pitcher_hard_hit_allowed_rate=("is_hard_hit", "mean"),
        pitcher_barrel_allowed_rate=("is_barrel", "mean"),
    )


def pitcher_pitch_mix(pitches: pd.DataFrame, before: pd.Timestamp) -> pd.DataFrame:
    """Return pitcher pitch-type usage before ``before``."""
    df = pitches[pd.to_datetime(pitches["game_date"]) < pd.Timestamp(before)].copy()
    df = df[df["pitch_type"].isin(PITCH_TYPES)]
    counts = df.groupby(["pitcher", "pitch_type"]).size().rename("pitches").reset_index()
    counts["usage"] = counts["pitches"] / counts.groupby("pitcher")["pitches"].transform("sum")
    return counts


def build_matchup_features(
    pitches: pd.DataFrame, batter_id: int, pitcher_id: int, before: pd.Timestamp
) -> dict[str, float]:
    """Summarize a batter against the pitcher's expected pitch mix."""
    batter = batter_pitch_type_rates(pitches, before)
    pitcher = pitcher_pitch_rates(pitches, before)
    mix = pitcher_pitch_mix(pitches, before)
    batter = batter[batter["batter"] == batter_id]
    pitcher = pitcher[pitcher["pitcher"] == pitcher_id]
    mix = mix[mix["pitcher"] == pitcher_id]
    merged = mix.merge(batter, on="pitch_type", how="left").merge(pitcher, on=["pitcher", "pitch_type"], how="left")
    if merged.empty:
        return {
            "pitch_matchup_hr_rate": float("nan"),
            "pitch_matchup_hard_hit_rate": float("nan"),
            "pitch_matchup_barrel_rate": float("nan"),
            "pitch_matchup_pitcher_hr_allowed_rate": float("nan"),
            "pitch_matchup_combined_risk": float("nan"),
            "pitch_mix_coverage": 0.0,
        }
    batter_hr = _weighted_mean(merged["batter_hr_rate"], merged["usage"])
    pitcher_hr = _weighted_mean(merged["pitcher_hr_allowed_rate"], merged["usage"])
    return {
        "pitch_matchup_hr_rate": batter_hr,
        "pitch_matchup_hard_hit_rate": _weighted_mean(merged["batter_hard_hit_rate"], merged["usage"]),
        "pitch_matchup_barrel_rate": _weighted_mean(merged["batter_barrel_rate"], merged["usage"]),
        "pitch_matchup_pitcher_hr_allowed_rate": pitcher_hr,
        "pitch_matchup_combined_risk": batter_hr * pitcher_hr if pd.notna(batter_hr) and pd.notna(pitcher_hr) else float("nan"),
        "pitch_mix_coverage": float(merged["batter_pitch_rate"].notna().mul(merged["usage"]).sum()),
    }
