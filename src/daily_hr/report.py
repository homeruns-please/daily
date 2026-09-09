"""Human-friendly formatting for daily HR model output."""

from __future__ import annotations

import pandas as pd


def format_daily_report(predictions: pd.DataFrame, date_label: str) -> str:
    """Render a concise Markdown report from ranked predictions."""
    required = {"batter", "opposing_pitcher", "hr_probability"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing report columns: {sorted(missing)}")

    ranked = predictions.sort_values("hr_probability", ascending=False).head(25)
    lines = [
        f"# 🔥 Daily Home Run Board — {date_label}",
        "",
        "> Model ranking of the strongest HR candidates. Probabilities are model estimates, not guarantees.",
        "",
        "| Rank | Batter | Opposing Pitcher | HR Probability | Key Factors |",
        "|---:|---|---|---:|---|",
    ]
    for rank, (_, row) in enumerate(ranked.iterrows(), 1):
        factors = row.get("key_factors", "")
        probability = f"{float(row['hr_probability']):.1%}"
        lines.append(
            f"| {rank} | {row['batter']} | {row['opposing_pitcher']} | **{probability}** | {factors} |"
        )
    lines.extend(
        [
            "",
            "### Model inputs",
            "Barrel%, hard-hit%, fly-ball%, launch angle, exit velocity, recent form, pitcher HR/contact profile, pitch mix, batter-vs-pitch-type performance, park, weather, and expected opportunity.",
        ]
    )
    return "\n".join(lines)
