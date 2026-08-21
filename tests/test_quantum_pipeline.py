"""
Automated Unit Tests for Quantum AI Predictive Maintenance Pipeline.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.telemetry_generator import (
    IndustrialAssetTelemetryGenerator,
    AssetConfig,
    generate_industrial_dataset,
    prepare_quantum_timeseries_dataset,
)
from src.quantum.feature_maps import AngleFeatureMap, ZZFeatureMap, build_quantum_feature_circuit
from src.quantum.quantum_kernel import QuantumKernel, compute_quantum_kernel_matrix
from src.quantum.quantum_regressor import (
    QuantumKernelRidgeRegressor,
    QuantumSVR,
)
from src.models.classical_baselines import get_all_classical_baselines
from src.models.evaluator import PredictiveMaintenanceEvaluator, calculate_economic_savings


def test_telemetry_generator():
    """Validates lifecycle telemetry generation across sensors."""
    cfg = AssetConfig(
        asset_id="TEST-ASSET-01",
        asset_type="refinery_compressor",
        total_cycles=100,
        fault_onset_cycle=50,
        noise_std=0.01,
        random_seed=42,
    )
    gen = IndustrialAssetTelemetryGenerator(cfg)
    df = gen.generate_single_run()

    assert len(df) == 100
    assert "vibration_rms" in df.columns
    assert "bearing_temperature" in df.columns
    assert "lubrication_pressure" in df.columns
    assert "acoustic_emission" in df.columns
    assert "RUL" in df.columns
    assert "true_degradation_index" in df.columns
    assert "health_state" in df.columns

    # Initial state should be healthy
    assert df.iloc[0]["health_state"] == "HEALTHY"
    # End state should be critical
    assert df.iloc[-1]["health_state"] == "CRITICAL"
    assert (df["vibration_rms"] > 0).all()


def test_quantum_dataset_preparation():
    """Validates quantum time-series windowing and normalization to [0, pi]."""
    df = generate_industrial_dataset(num_assets=2, cycles_range=(80, 100), random_seed=42)
    X, y_rul, y_state, meta = prepare_quantum_timeseries_dataset(
        df,
        window_size=4,
        stride=2,
        num_qubits=4,
        scale_range=(0.0, np.pi),
    )

    assert X.ndim == 2
    assert X.shape[1] == 4
    assert len(y_rul) == len(X)
    assert len(y_state) == len(X)
    # Check bounded rotation range for quantum gates
    assert np.min(X) >= -1e-5
    assert np.max(X) <= np.pi + 1e-5


def test_quantum_feature_maps():
    """Validates circuit constructions and connectivity."""
    zz_map = ZZFeatureMap(num_qubits=4, reps=2, entanglement="linear")
    pairs = zz_map.get_entanglement_pairs()
    assert pairs == [(0, 1), (1, 2), (2, 3)]

    angle_map = AngleFeatureMap(num_qubits=4, rotation="Y")
    assert angle_map.rotation == "Y"

    fmap = build_quantum_feature_circuit("zz", num_qubits=4)
    assert isinstance(fmap, ZZFeatureMap)


def test_quantum_kernel_properties():
    """Validates Gram matrix symmetry, unit diagonal, and [0, 1] bounds."""
    np.random.seed(42)
    X_sample = np.random.uniform(0.0, np.pi, size=(6, 4))

    qk = QuantumKernel(num_qubits=4, feature_map="zz", reps=1)
    K = qk.compute_matrix(X_sample)

    assert K.shape == (6, 6)
    # Unit diagonal
    np.testing.assert_allclose(np.diag(K), np.ones(6), atol=1e-5)
    # Symmetry
    np.testing.assert_allclose(K, K.T, atol=1e-5)
    # Bounded in [0, 1]
    assert np.all(K >= 0.0)
    assert np.all(K <= 1.0 + 1e-6)


def test_quantum_regressors_and_baselines():
    """Validates end-to-end fitting and inference for QKRR and QSVR."""
    np.random.seed(42)
    X_train = np.random.uniform(0.0, np.pi, size=(12, 4))
    y_train = np.linspace(100, 10, 12) + np.random.normal(0, 1, 12)

    X_test = np.random.uniform(0.0, np.pi, size=(5, 4))
    y_test = np.linspace(90, 20, 5)

    # 1. QKRR
    qkrr = QuantumKernelRidgeRegressor(alpha_reg=1e-2, num_qubits=4, reps=1)
    qkrr.fit(X_train, y_train)
    preds_qkrr = qkrr.predict(X_test)
    assert len(preds_qkrr) == 5

    # 2. QSVR
    qsvr = QuantumSVR(C=10.0, epsilon=0.1, num_qubits=4, reps=1)
    qsvr.fit(X_train, y_train)
    preds_qsvr = qsvr.predict(X_test)
    assert len(preds_qsvr) == 5

    # 3. Classical baselines
    baselines = get_all_classical_baselines()
    for name, model in baselines.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        assert len(pred) == 5

    # 4. Evaluator
    evaluator = PredictiveMaintenanceEvaluator()
    metrics = evaluator.evaluate_model("QKRR", y_test, preds_qkrr)
    assert metrics.rmse >= 0.0
    assert metrics.mae >= 0.0
    assert isinstance(metrics.estimated_cost_savings_usd, float)
