"""
Unit tests for RUL Estimator and Quantum-Classical Ensemble.
"""

import numpy as np
import pandas as pd
import pytest
from src.models.rul_estimator import RULEstimator, RULPredictionResult
from src.models.ensemble import QuantumClassicalEnsemble


def test_rul_estimator():
    y_true = np.linspace(100, 0, 50)
    y_pred = y_true + np.random.normal(0, 5, 50)
    
    estimator = RULEstimator()
    estimator.fit_residuals(y_true, y_pred)
    
    res = estimator.estimate_rul(y_pred, current_cycles=np.arange(1, 51))
    assert isinstance(res, RULPredictionResult)
    assert len(res.predicted_rul) == 50
    assert len(res.lower_bound) == 50
    assert len(res.upper_bound) == 50
    assert len(res.estimated_failure_cycles) == 50
    assert len(res.degradation_trend) == 50

    # Lower bound should always be <= predicted_rul <= upper_bound
    assert np.all(res.lower_bound <= res.predicted_rul + 1e-5)
    assert np.all(res.predicted_rul <= res.upper_bound + 1e-5)
    
    # Degradation trend should increase towards 1.0
    assert res.degradation_trend[-1] >= res.degradation_trend[0]


def test_quantum_classical_ensemble():
    n = 30
    y_true = np.linspace(150, 10, n)
    
    # Simulate predictions from 5 distinct models with varying noise levels
    preds = {
        "QKRR": y_true + np.random.normal(0, 3, n),
        "QSVR": y_true + np.random.normal(0, 4, n),
        "SVR": y_true + np.random.normal(0, 6, n),
        "Random Forest": y_true + np.random.normal(0, 7, n),
        "Ridge": y_true + np.random.normal(0, 10, n),
    }

    # 1. Weighted blend
    ens_blend = QuantumClassicalEnsemble(strategy="weighted_blend")
    ens_blend.fit_weights(preds, y_true)
    assert len(ens_blend.weights) == 5
    assert pytest.approx(sum(ens_blend.weights.values()), 0.01) == 1.0
    
    p_blend, var_blend = ens_blend.predict(preds)
    assert len(p_blend) == n
    assert len(var_blend) == n
    assert np.all(var_blend >= 0.0)

    # 2. Stacking meta-model
    ens_stack = QuantumClassicalEnsemble(strategy="stacking")
    ens_stack.fit_weights(preds, y_true)
    p_stack, var_stack = ens_stack.predict(preds)
    assert len(p_stack) == n
    assert len(var_stack) == n
