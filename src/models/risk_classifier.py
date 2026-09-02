"""
Multi-Class Failure Risk Classification for Industrial Predictive Maintenance.

Classifies asset operational degradation into:
- NORMAL: Healthy machinery baseline state.
- WARNING: Incipient degradation detected; inspection recommended.
- CRITICAL: Severe wear / imminent failure; intervention required.

Outputs class probabilities, risk level, and prediction confidence.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV


@dataclass
class RiskClassificationResult:
    """Encapsulates failure-risk predictions and probability distributions."""
    predicted_states: List[str]                  # List of 'NORMAL', 'WARNING', 'CRITICAL'
    class_probabilities: List[Dict[str, float]] # Per-sample probability distribution
    confidence_scores: np.ndarray               # Max probability per sample (0.0 - 1.0)
    critical_risk_ratio: float                  # Fraction of cycles in CRITICAL state
    warning_risk_ratio: float                   # Fraction of cycles in WARNING state


class FailureRiskClassifier:
    """
    Classifies industrial asset health states and assigns failure risk probabilities.
    """

    STATES = ["NORMAL", "WARNING", "CRITICAL"]

    def __init__(
        self,
        model_type: str = "random_forest",  # 'random_forest', 'gradient_boosting', 'logistic'
        warning_rul_threshold: Optional[float] = None,
        critical_rul_threshold: Optional[float] = None,
    ):
        self.model_type = model_type
        self.warning_rul_threshold = warning_rul_threshold
        self.critical_rul_threshold = critical_rul_threshold
        
        self.scaler = StandardScaler()
        if model_type == "random_forest":
            base_clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        elif model_type == "gradient_boosting":
            base_clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
        elif model_type == "logistic":
            base_clf = LogisticRegression(max_iter=1000, random_state=42)
        else:
            base_clf = RandomForestClassifier(n_estimators=100, random_state=42)

        self.model = CalibratedClassifierCV(estimator=base_clf, cv=3)
        self.feature_names: List[str] = []

    def assign_risk_labels(self, y_rul: np.ndarray) -> np.ndarray:
        """Converts continuous Remaining Useful Life (RUL) cycles into discrete 3-stage risk states."""
        y_rul = np.asarray(y_rul, dtype=float)
        max_rul = max(10.0, float(np.max(y_rul)))
        
        # Adaptive thresholds if not specified
        warn_th = self.warning_rul_threshold if self.warning_rul_threshold is not None else 0.45 * max_rul
        crit_th = self.critical_rul_threshold if self.critical_rul_threshold is not None else 0.15 * max_rul
        
        labels = []
        for rul in y_rul:
            if rul > warn_th:
                labels.append("NORMAL")
            elif rul > crit_th:
                labels.append("WARNING")
            else:
                labels.append("CRITICAL")
        return np.array(labels)

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y_rul_or_labels: np.ndarray) -> "FailureRiskClassifier":
        """
        Trains the calibrated failure risk classifier.
        Accepts either continuous RUL values or explicit string state labels.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = [c for c in X.columns if c not in ["cycle", "asset_id", "asset_type", "RUL", "true_degradation_index", "health_state"]]
            X_mat = X[self.feature_names].values
        else:
            X_mat = np.asarray(X, dtype=float)
            self.feature_names = [f"feat_{i}" for i in range(X_mat.shape[1])]

        # If numeric RUL is passed, discretize into 3 classes
        if np.issubdtype(np.array(y_rul_or_labels).dtype, np.number):
            y_labels = self.assign_risk_labels(np.asarray(y_rul_or_labels, dtype=float))
        else:
            y_labels = np.asarray(y_rul_or_labels, dtype=str)

        X_scaled = self.scaler.fit_transform(X_mat)
        self.model.fit(X_scaled, y_labels)
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> RiskClassificationResult:
        """
        Predicts failure-risk state, class probabilities, and confidence scores.
        """
        if isinstance(X, pd.DataFrame):
            cols = [c for c in self.feature_names if c in X.columns]
            X_mat = X[cols].values
        else:
            X_mat = np.asarray(X, dtype=float)

        X_scaled = self.scaler.transform(X_mat)
        preds = self.model.predict(X_scaled)
        probs_mat = self.model.predict_proba(X_scaled)
        classes = list(self.model.classes_)

        prob_dicts = []
        confidences = []
        for i in range(len(preds)):
            row_prob = {cls: round(float(probs_mat[i, idx]), 4) for idx, cls in enumerate(classes)}
            # Ensure all 3 states are present in dictionary
            for st in self.STATES:
                if st not in row_prob:
                    row_prob[st] = 0.0
            prob_dicts.append(row_prob)
            confidences.append(float(np.max(probs_mat[i])))

        conf_arr = np.array(confidences)
        crit_ratio = float(np.mean(preds == "CRITICAL"))
        warn_ratio = float(np.mean(preds == "WARNING"))

        return RiskClassificationResult(
            predicted_states=list(preds),
            class_probabilities=prob_dicts,
            confidence_scores=conf_arr,
            critical_risk_ratio=crit_ratio,
            warning_risk_ratio=warn_ratio,
        )
