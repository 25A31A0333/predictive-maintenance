"""
Industrial Multi-Sensor Anomaly Detection Module for Predictive Maintenance.

Provides:
- Unsupervised anomaly scoring (PCA Reconstruction Residuals & Covariance Mahalanobis Distance).
- Dynamic, baseline-calibrated anomaly thresholds.
- Continuous anomaly timeline & severity state assignment.
- Sensor-level attribution identifying which telemetry channels caused the anomaly.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


@dataclass
class AnomalyDetectionResult:
    """Encapsulates full anomaly detection diagnostics."""
    anomaly_scores: np.ndarray          # Continuous anomaly score (0 - 100)
    anomaly_threshold: float           # Calibrated dynamic threshold
    is_anomaly: np.ndarray             # Boolean array of anomaly triggers
    severity_levels: List[str]         # 'NORMAL', 'WARNING', 'CRITICAL' per cycle
    affected_sensors: List[Dict[str, float]] # Top contributing sensors per cycle
    onset_cycle: Optional[int]         # First sustained anomaly cycle
    total_anomaly_cycles: int          # Count of anomalous cycles


class IndustrialAnomalyDetector:
    """
    Detects incipient mechanical degradation & sensor anomalies prior to functional equipment failure.
    """

    def __init__(
        self,
        method: str = "pca_reconstruction",  # 'pca_reconstruction', 'mahalanobis', or 'isolation_forest'
        n_components: int = 3,
        contamination: float = 0.05,
        threshold_percentile: float = 98.5,
    ):
        self.method = method
        self.n_components = n_components
        self.contamination = contamination
        self.threshold_percentile = threshold_percentile
        
        self.scaler = StandardScaler()
        self.pca_model: Optional[PCA] = None
        self.cov_model: Optional[EmpiricalCovariance] = None
        self.iforest_model: Optional[IsolationForest] = None
        self.threshold: float = 50.0
        self.sensor_cols: List[str] = []

    def fit(self, X_healthy: Union[pd.DataFrame, np.ndarray], sensor_cols: Optional[List[str]] = None) -> "IndustrialAnomalyDetector":
        """
        Calibrates anomaly models and baseline thresholds on known healthy operational data.
        """
        if isinstance(X_healthy, pd.DataFrame):
            self.sensor_cols = sensor_cols or [c for c in X_healthy.columns if c not in ["cycle", "asset_id", "asset_type", "RUL", "true_degradation_index", "health_state"]]
            X_mat = X_healthy[self.sensor_cols].values
        else:
            X_mat = np.asarray(X_healthy, dtype=float)
            self.sensor_cols = sensor_cols or [f"sensor_{i}" for i in range(X_mat.shape[1])]

        X_scaled = self.scaler.fit_transform(X_mat)

        if self.method == "pca_reconstruction":
            k = min(self.n_components, X_scaled.shape[1] - 1) if X_scaled.shape[1] > 1 else 1
            self.pca_model = PCA(n_components=k, random_state=42)
            self.pca_model.fit(X_scaled)
            scores = self._compute_pca_residuals(X_scaled)
        elif self.method == "mahalanobis":
            self.cov_model = EmpiricalCovariance().fit(X_scaled)
            scores = self.cov_model.mahalanobis(X_scaled)
        elif self.method == "isolation_forest":
            self.iforest_model = IsolationForest(contamination=self.contamination, random_state=42)
            self.iforest_model.fit(X_scaled)
            raw_scores = -self.iforest_model.score_samples(X_scaled)
            scores = raw_scores
        else:
            raise ValueError(f"Unknown anomaly detection method: {self.method}")

        # Compute dynamic baseline threshold from healthy data percentile
        raw_thresh = np.percentile(scores, self.threshold_percentile)
        self.threshold = float(raw_thresh)
        return self

    def _compute_pca_residuals(self, X_scaled: np.ndarray) -> np.ndarray:
        """Computes per-sample squared reconstruction error."""
        X_reconstructed = self.pca_model.inverse_transform(self.pca_model.transform(X_scaled))
        residuals = np.sum((X_scaled - X_reconstructed) ** 2, axis=1)
        return residuals

    def _compute_sensor_contributions(self, X_scaled: np.ndarray) -> List[Dict[str, float]]:
        """
        Determines the relative percentage contribution of each sensor channel to the anomaly score.
        """
        n_samples, n_features = X_scaled.shape
        contributions_list = []

        if self.pca_model is not None:
            X_reconstructed = self.pca_model.inverse_transform(self.pca_model.transform(X_scaled))
            per_sensor_sq_error = (X_scaled - X_reconstructed) ** 2
        else:
            # Fallback: squared standard deviation from mean
            per_sensor_sq_error = X_scaled ** 2

        for i in range(n_samples):
            row_err = per_sensor_sq_error[i]
            total_err = np.sum(row_err)
            if total_err < 1e-9:
                contrib = {self.sensor_cols[j]: round(100.0 / n_features, 1) for j in range(n_features)}
            else:
                raw_pct = (row_err / total_err) * 100.0
                contrib = {self.sensor_cols[j]: round(float(raw_pct[j]), 1) for j in range(n_features)}
            # Sort descending
            sorted_contrib = dict(sorted(contrib.items(), key=lambda x: x[1], reverse=True))
            contributions_list.append(sorted_contrib)

        return contributions_list

    def detect(self, X_eval: Union[pd.DataFrame, np.ndarray]) -> AnomalyDetectionResult:
        """
        Evaluates incoming telemetry and outputs continuous anomaly scores, severity, and triggers.
        """
        if isinstance(X_eval, pd.DataFrame):
            cols = [c for c in self.sensor_cols if c in X_eval.columns]
            X_mat = X_eval[cols].values
        else:
            X_mat = np.asarray(X_eval, dtype=float)

        X_scaled = self.scaler.transform(X_mat)

        # Compute raw scores
        if self.method == "pca_reconstruction":
            raw_scores = self._compute_pca_residuals(X_scaled)
        elif self.method == "mahalanobis":
            raw_scores = self.cov_model.mahalanobis(X_scaled)
        elif self.method == "isolation_forest":
            raw_scores = -self.iforest_model.score_samples(X_scaled)

        # Normalize score to a continuous index between 0 and 100
        scale_factor = max(1e-5, self.threshold)
        norm_scores = (raw_scores / scale_factor) * 50.0  # Threshold maps to 50.0
        norm_scores = np.clip(norm_scores, 0.0, 100.0)

        # Triggers and Severity Levels
        is_anomaly = norm_scores >= 50.0
        severity_levels = []
        for s in norm_scores:
            if s < 45.0:
                severity_levels.append("NORMAL")
            elif s < 75.0:
                severity_levels.append("WARNING")
            else:
                severity_levels.append("CRITICAL")

        # Sensor-level attribution
        affected_sensors = self._compute_sensor_contributions(X_scaled)

        # Detect sustained onset cycle (at least 3 consecutive anomalous triggers)
        onset_cycle = None
        for i in range(len(is_anomaly) - 2):
            if is_anomaly[i] and is_anomaly[i+1] and is_anomaly[i+2]:
                onset_cycle = i + 1  # 1-indexed cycle
                break

        return AnomalyDetectionResult(
            anomaly_scores=norm_scores,
            anomaly_threshold=50.0,  # Normalized threshold is 50.0
            is_anomaly=is_anomaly,
            severity_levels=severity_levels,
            affected_sensors=affected_sensors,
            onset_cycle=onset_cycle,
            total_anomaly_cycles=int(np.sum(is_anomaly)),
        )
