"""
Unit tests for Explainable AI and Maintenance Recommendations.
"""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from src.models.explainability import ModelExplainabilityAnalyzer, ExplanationResult
from src.models.maintenance_advisor import MaintenanceAdvisor, MaintenanceRecommendation


def test_explainability_analyzer():
    np.random.seed(42)
    # Feature 0 and 2 strongly affect target
    X = np.random.uniform(0, 10, size=(60, 4))
    y = 3.0 * X[:, 0] + 5.0 * X[:, 2] + np.random.normal(0, 0.1, 60)

    model = RandomForestRegressor(n_estimators=30, random_state=42)
    model.fit(X, y)

    sensor_names = ["vibration_rms", "vibration_kurtosis", "bearing_temperature", "lubrication_pressure"]
    analyzer = ModelExplainabilityAnalyzer(sensor_names=sensor_names)
    exp_res = analyzer.compute_permutation_importance(model, X, y)

    assert isinstance(exp_res, ExplanationResult)
    assert len(exp_res.feature_importance) == 4
    assert len(exp_res.top_contributing_sensors) == 4
    assert pytest.approx(sum(exp_res.feature_importance.values()), 0.05) == 100.0

    # Bearing temperature (index 2) and Vibration RMS (index 0) should have high importance
    top_feature_names = [s[0] for s in exp_res.top_contributing_sensors[:2]]
    assert "bearing_temperature" in top_feature_names or "vibration_rms" in top_feature_names
    assert "Top contributing sensors:" in exp_res.summary_text


def test_maintenance_advisor():
    advisor = MaintenanceAdvisor()

    # Case 1: High Risk Critical Case
    rec_crit = advisor.generate_recommendation(
        asset_id="PUMP-01",
        asset_type="chemical_pump",
        predicted_rul=35.0,
        risk_state="CRITICAL",
        anomaly_score=85.0,
        top_sensors=[("bearing_temperature", 65.0), ("vibration_rms", 25.0)],
    )
    assert isinstance(rec_crit, MaintenanceRecommendation)
    assert rec_crit.risk_level == "HIGH RISK"
    assert rec_crit.urgency == "IMMEDIATE_ACTION"
    assert "bearing" in rec_crit.likely_issue.lower() or "overheating" in rec_crit.likely_issue.lower()
    assert len(rec_crit.inspection_checklist) > 0
    assert "DECISION-SUPPORT" in rec_crit.disclaimer

    # Case 2: Healthy Normal Case
    rec_norm = advisor.generate_recommendation(
        asset_id="TURBINE-02",
        asset_type="utility_turbine",
        predicted_rul=240.0,
        risk_state="NORMAL",
        anomaly_score=15.0,
    )
    assert rec_norm.risk_level == "LOW RISK / NORMAL"
    assert rec_norm.urgency == "ROUTINE_MONITORING"
