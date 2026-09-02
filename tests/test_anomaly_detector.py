"""
Unit tests for Industrial Anomaly Detector.
"""

import numpy as np
import pandas as pd
import pytest
from src.data.telemetry_generator import IndustrialAssetTelemetryGenerator, AssetConfig
from src.data.feature_engineering import IndustrialFeatureEngineer
from src.models.anomaly_detector import IndustrialAnomalyDetector, AnomalyDetectionResult


def test_anomaly_detector_training_and_detection():
    cfg = AssetConfig(asset_id="TEST-ANOM-01", asset_type="refinery_compressor", total_cycles=120, fault_onset_cycle=60, random_seed=42)
    df = IndustrialAssetTelemetryGenerator(cfg).generate_single_run()
    engineer = IndustrialFeatureEngineer()
    feat_df = engineer.extract_time_series_features(df)

    # Train on healthy data (cycles before fault onset)
    df_healthy = feat_df.iloc[:50]
    
    for method in ["pca_reconstruction", "mahalanobis", "isolation_forest"]:
        detector = IndustrialAnomalyDetector(method=method)
        detector.fit(df_healthy)
        
        result = detector.detect(feat_df)
        assert isinstance(result, AnomalyDetectionResult)
        assert len(result.anomaly_scores) == 120
        assert len(result.severity_levels) == 120
        assert len(result.affected_sensors) == 120
        
        # Early healthy cycles should have low anomaly scores (< 50)
        assert np.mean(result.anomaly_scores[:30]) < np.mean(result.anomaly_scores[90:])
        
        # End of lifecycle should show CRITICAL or WARNING severity
        assert result.severity_levels[-1] in ["WARNING", "CRITICAL"]
        
        # Check that top affected sensors dictionary is non-empty
        top_sensors = result.affected_sensors[-1]
        assert len(top_sensors) > 0
        assert sum(top_sensors.values()) > 0
