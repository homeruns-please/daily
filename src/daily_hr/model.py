"""Model training and probability calibration."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


@dataclass
class ModelMetrics:
    roc_auc: float
    log_loss: float
    brier: float


def train_baseline(X: pd.DataFrame, y: pd.Series) -> HistGradientBoostingClassifier:
    """Train a strong, interpretable first-pass gradient-boosted classifier."""
    model = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=300,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(X, y)
    return model


def evaluate(y_true: pd.Series, probabilities: pd.Series) -> ModelMetrics:
    """Evaluate probabilistic predictions."""
    return ModelMetrics(
        roc_auc=roc_auc_score(y_true, probabilities),
        log_loss=log_loss(y_true, probabilities, labels=[0, 1]),
        brier=brier_score_loss(y_true, probabilities),
    )
