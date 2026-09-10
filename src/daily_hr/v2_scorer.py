"""Experimental V2 home-run ranking scorer.

V2 keeps the same scoring mechanics as V1 but changes the feature weights based
on the recent Sep 7-8 evaluation:
- lower park and lineup influence
- higher pitcher vulnerability, recent power, and weather influence

V1 is the original production scorer in production_scorer.py. V2 is experimental
and must not overwrite V1.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import pandas as pd


# V2 experimental weights. These intentionally differ from V1.
WEIGHTS_V2 = {
    "baseline": 0.25,
    "pitcher_vulnerability": 0.275,
    "pitch_matchup": 0.175,
    "power_upside": 0.20,
    "park": 0.025,
    "weather": 0.05,
    "lineup": 0.025,
}


def _unit_rating(value: object, default: float = 5.5) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        x = default
    if math.isnan(x):
        x = default
    return max(0.0, min(1.0, x / 10.0))


def _lineup_rating(value: object) -> float:
    try:
        slot = float(value)
    except (TypeError, ValueError):
        return 0.5
    if math.isnan(slot):
        return 0.5
    # 1st is best; slots 1-9 are mapped to 1.0-0.2.
    return max(0.0, min(1.0, (10.0 - slot) / 9.0))


def score_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Apply V2 scoring to a board dataframe and return rows ranked by score."""
    out = df.copy()

    out["v2_baseline_component"] = out["baseline_rating"].map(_unit_rating)

    pitcher = out.get("pitcher_vulnerability_rating", pd.Series(5.5, index=out.index))
    matchup = out.get("pitch_matchup_edge_rating", pd.Series(5.5, index=out.index))
    baseline_matchup = out.get("matchup_rating", pd.Series(5.5, index=out.index))
    pitcher = pitcher.fillna(baseline_matchup)
    matchup = matchup.fillna(baseline_matchup)

    out["v2_pitcher_vulnerability_component"] = pitcher.map(_unit_rating)
    out["v2_pitch_matchup_component"] = matchup.map(_unit_rating)

    power_cols = [
        "recent_power_rating",
        "barrel_rating",
        "contact_quality_rating",
        "exit_velocity_rating",
    ]
    available = [c for c in power_cols if c in out.columns]
    if available:
        power = out[available].apply(pd.to_numeric, errors="coerce").mean(axis=1).fillna(5.5)
    else:
        power = pd.Series(5.5, index=out.index)
    out["v2_power_upside_component"] = power.map(_unit_rating)

    out["v2_park_component"] = out.get("park_rating", pd.Series(5.5, index=out.index)).map(_unit_rating)
    out["v2_weather_component"] = out.get("weather_rating", pd.Series(5.5, index=out.index)).map(_unit_rating)
    out["v2_lineup_component"] = out.get("lineup_slot", pd.Series(5.5, index=out.index)).map(_lineup_rating)

    out["v2_score"] = (
        WEIGHTS_V2["baseline"] * out["v2_baseline_component"]
        + WEIGHTS_V2["pitcher_vulnerability"] * out["v2_pitcher_vulnerability_component"]
        + WEIGHTS_V2["pitch_matchup"] * out["v2_pitch_matchup_component"]
        + WEIGHTS_V2["power_upside"] * out["v2_power_upside_component"]
        + WEIGHTS_V2["park"] * out["v2_park_component"]
        + WEIGHTS_V2["weather"] * out["v2_weather_component"]
        + WEIGHTS_V2["lineup"] * out["v2_lineup_component"]
    )

    return out.sort_values(["v2_score", "player"], ascending=[False, True]).reset_index(drop=True)


def rerank_csv(input_path: str | Path, output_path: str | Path) -> None:
    df = pd.read_csv(input_path)
    scored = score_frame(df)
    scored.to_csv(output_path, index=False)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Apply experimental V2 HR ranking to a board CSV")
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    args = parser.parse_args()
    rerank_csv(args.input_csv, args.output_csv)


if __name__ == "__main__":
    main()
