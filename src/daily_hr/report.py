"""Human-friendly formatting for daily HR model output."""

from __future__ import annotations

import pandas as pd

from .market import add_market_comparison


def _odds(value: object) -> str:
    if pd.isna(value):
        return "—"
    odds = int(float(value))
    return f"+{odds}" if odds > 0 else str(odds)


def format_daily_report(predictions: pd.DataFrame, date_label: str) -> str:
    """Render a concise Markdown report from ranked predictions.

    Sportsbook prices are display/comparison data only. They are never model
    features and therefore cannot determine the model's HR probability.
    """
    required = {"batter", "opposing_pitcher", "hr_probability"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing report columns: {sorted(missing)}")

    ranked = add_market_comparison(predictions).sort_values(
        "hr_probability", ascending=False
    ).head(25)
    has_market = any(
        column in ranked.columns for column in ("fanduel_odds", "draftkings_odds")
    )

    header = "| Rank | Batter | Opposing Pitcher | HR Probability |"
    separator = "|---:|---|---|---:|"
    if has_market:
        header += " FanDuel | DraftKings | Best Market Implied | Model Edge |"
        separator += "---:|---:|---:|---:|"
    header += " Key Factors |"
    separator += "---|"

    lines = [
        f"# 🔥 Daily Home Run Board — {date_label}",
        "",
        "> Model ranking of the strongest HR candidates. Sportsbook odds are shown for market comparison only; they do not drive the model probability.",
        "",
        header,
        separator,
    ]
    for rank, (_, row) in enumerate(ranked.iterrows(), 1):
        factors = row.get("key_factors", "")
        probability = f"{float(row['hr_probability']):.1%}"
        values = [
            str(rank),
            str(row["batter"]),
            str(row["opposing_pitcher"]),
            f"**{probability}**",
        ]
        if has_market:
            best_implied = row.get("best_market_implied", pd.NA)
            edge = row.get("model_edge", pd.NA)
            values.extend(
                [
                    _odds(row.get("fanduel_odds", pd.NA)),
                    _odds(row.get("draftkings_odds", pd.NA)),
                    "—" if pd.isna(best_implied) else f"{float(best_implied):.1%}",
                    "—" if pd.isna(edge) else f"{float(edge):+.1%}",
                ]
            )
        values.append(factors)
        lines.append("| " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "### Model inputs",
            "Barrel%, hard-hit%, fly-ball%, launch angle, exit velocity, recent form, pitcher HR/contact profile, pitch mix, batter-vs-pitch-type performance, park, weather, and expected opportunity.",
        ]
    )
    return "\n".join(lines)
