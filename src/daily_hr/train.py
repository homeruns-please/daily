"""Command-line training helper for chronological HR models."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .model import evaluate, fit_calibrated_model


def train_file(path: Path, date_column: str = "game_date", target_column: str = "hr_target"):
    """Train with chronological train/calibration/test blocks."""
    df = pd.read_parquet(path).sort_values(date_column).reset_index(drop=True)
    if target_column not in df:
        raise ValueError(f"Missing target column: {target_column}")
    dates = pd.to_datetime(df[date_column])
    train_cut = dates.quantile(0.70)
    cal_cut = dates.quantile(0.85)
    excluded = {date_column, target_column, "game_date", "batter", "game_pk", "home_runs"}
    features = [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
    X = df[features].replace([float("inf"), float("-inf")], pd.NA).fillna(0)
    y = df[target_column].astype(int)
    train = dates < train_cut
    cal = (dates >= train_cut) & (dates < cal_cut)
    test = dates >= cal_cut
    fitted = fit_calibrated_model(X.loc[train], y.loc[train], X.loc[cal], y.loc[cal])
    probabilities = fitted.predict_proba(X.loc[test])
    metrics = evaluate(y.loc[test], probabilities)
    return fitted, metrics, features


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the daily HR model with time-based validation")
    parser.add_argument("parquet", type=Path)
    args = parser.parse_args()
    _, metrics, features = train_file(args.parquet)
    print(f"features={len(features)}")
    print(f"test_roc_auc={metrics.roc_auc:.4f}")
    print(f"test_log_loss={metrics.log_loss:.4f}")
    print(f"test_brier={metrics.brier:.4f}")


if __name__ == "__main__":
    main()
