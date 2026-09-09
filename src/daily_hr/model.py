"""Model training, evaluation, and probability calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


@dataclass
class ModelMetrics:
    roc_auc: float
    log_loss: float
    brier: float


@dataclass
class CalibratedModel:
    model: HistGradientBoostingClassifier
    calibrator: IsotonicRegression

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.model.predict_proba(X)[:, 1]
        calibrated = self.calibrator.predict(raw)
        return np.clip(calibrated, 1e-6, 1 - 1e-6)


def train_baseline(X: pd.DataFrame, y: pd.Series) -> HistGradientBoostingClassifier:
    """Train a strong first-pass gradient-boosted classifier."""
    model = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=300,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(X, y)
    return model


def fit_calibrated_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
) -> CalibratedModel:
    """Fit on an earlier time block and calibrate on a later block."""
    model = train_baseline(X_train, y_train)
    raw = model.predict_proba(X_cal)[:, 1]
    calibrator = IsotonicRegression(
        y_min=1e-6,
        y_max=1 - 1e-6,
        out_of_bounds="clip",
    )
    calibrator.fit(raw, y_cal)
    return CalibratedModel(model=model, calibrator=calibrator)


def evaluate(y_true: pd.Series, probabilities: pd.Series | np.ndarray) -> ModelMetrics:
    """Evaluate probabilistic predictions."""
    p = np.asarray(probabilities)
    return ModelMetrics(
        roc_auc=roc_auc_score(y_true, p),
        log_loss=log_loss(y_true, p, labels=[0, 1]),
        brier=brier_score_loss(y_true, p),
    )
