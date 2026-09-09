"""Sportsbook market data kept separate from the HR probability model."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MarketQuote:
    """A sportsbook's current 1+ HR price for one hitter."""

    sportsbook: str
    american_odds: int

    @property
    def implied_probability(self) -> float:
        return american_to_probability(self.american_odds)


def american_to_probability(american_odds: float) -> float:
    """Convert American odds to raw implied probability."""
    odds = float(american_odds)
    if odds == 0:
        raise ValueError("American odds cannot be zero")
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return -odds / (-odds + 100.0)


def add_market_comparison(predictions: pd.DataFrame) -> pd.DataFrame:
    """Add display/value fields without feeding sportsbook prices into the model.

    Expected optional columns are ``fanduel_odds`` and ``draftkings_odds``.
    The returned ``model_edge`` is the model probability minus the lower raw
    implied probability available across the two books.
    """
    out = predictions.copy()
    for column in ("fanduel_odds", "draftkings_odds"):
        if column in out:
            out[f"{column}_implied"] = out[column].apply(
                lambda value: american_to_probability(value) if pd.notna(value) else pd.NA
            )

    implied_columns = [
        column
        for column in ("fanduel_odds_implied", "draftkings_odds_implied")
        if column in out
    ]
    if implied_columns:
        out["best_market_implied"] = out[implied_columns].min(axis=1, skipna=True)
        out["model_edge"] = out["hr_probability"] - out["best_market_implied"]
    return out
