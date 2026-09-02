"""
Unit tests for Advanced Sensor Feature Engineering.
"""

import numpy as np
import pandas as pd
import pytest
from src.data.telemetry_generator import IndustrialAssetTelemetryGenerator, AssetConfig
from src.data.feature_engineering import IndustrialFeatureEngineer


def test_feature_engineering_pipeline():
    cfg = AssetConfig(asset_id="TEST-ENG-01", asset_type="refinery_compressor", total_cycles=80, fault_onset_cycle=40, random_seed=42)
    df = IndustrialAssetTelemetryGenerator(cfg).generate_single_run()
    
    # Introduce some artificial NaN values to test imputation
    df.loc[10, "vibration_rms"] = np.nan
    df.loc[25, "bearing_temperature"] = np.nan

    engineer = IndustrialFeatureEngineer(window_size=5, ewma_span=7)
    feat_df = engineer.extract_time_series_features(df)

    assert len(feat_df) == 80
    assert not feat_df.isnull().any().any(), "No NaNs should remain after cleaning and feature extraction"
    
    # Check that rolling, dynamic, and coupling features are present
    assert "vibration_rms_roll_mean" in feat_df.columns
    assert "vibration_rms_roll_std" in feat_df.columns
    assert "bearing_temperature_velocity" in feat_df.columns
    assert "bearing_temperature_acceleration" in feat_df.columns
    assert "vib_temp_coupling" in feat_df.columns
    assert "composite_health_indicator" in feat_df.columns

    # Health indicator should be high initially and decrease after fault onset
    hi = feat_df["composite_health_indicator"].values
    assert hi[0] > hi[-1]
    assert np.all(hi >= 0.0) and np.all(hi <= 1.0)


def test_quantum_feature_transformation():
    cfg = AssetConfig(asset_id="TEST-ENG-02", asset_type="port_gantry_crane", total_cycles=60, fault_onset_cycle=30, random_seed=42)
    df = IndustrialAssetTelemetryGenerator(cfg).generate_single_run()

    engineer = IndustrialFeatureEngineer()
    X_quant = engineer.prepare_quantum_features(df, num_qubits=4, scale_range=(0.0, np.pi), fit=True)

    assert X_quant.shape == (60, 4)
    assert np.min(X_quant) >= -1e-5
    assert np.max(X_quant) <= np.pi + 1e-5
