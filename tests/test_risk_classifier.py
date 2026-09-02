"""
Unit tests for Failure Risk Classifier.
"""

import numpy as np
import pandas as pd
import pytest
from src.data.telemetry_generator import IndustrialAssetTelemetryGenerator, AssetConfig
from src.data.feature_engineering import IndustrialFeatureEngineer
from src.models.risk_classifier import FailureRiskClassifier, RiskClassificationResult


def test_failure_risk_classifier():
    cfg = AssetConfig(asset_id="TEST-RISK-01", asset_type="utility_turbine", total_cycles=100, fault_onset_cycle=50, random_seed=42)
    df = IndustrialAssetTelemetryGenerator(cfg).generate_single_run()
    engineer = IndustrialFeatureEngineer()
    feat_df = engineer.extract_time_series_features(df)

    y_rul = df["RUL"].values

    classifier = FailureRiskClassifier(model_type="random_forest")
    classifier.fit(feat_df, y_rul)

    res = classifier.predict(feat_df)
    assert isinstance(res, RiskClassificationResult)
    assert len(res.predicted_states) == 100
    assert len(res.class_probabilities) == 100
    assert len(res.confidence_scores) == 100

    # Initial state should be classified as NORMAL
    assert res.predicted_states[0] == "NORMAL"
    # End state should be classified as CRITICAL
    assert res.predicted_states[-1] == "CRITICAL"

    # Probabilities should sum to ~1.0
    first_prob = res.class_probabilities[0]
    assert pytest.approx(sum(first_prob.values()), 0.01) == 1.0
    assert first_prob["NORMAL"] > 0.5
