"""
Unified End-to-End Quantum-Classical Predictive Maintenance Pipeline.

Orchestrates:
1. Multi-Asset Telemetry Ingestion & Advanced Feature Engineering
2. Quantum Kernel Matrix Construction & Hilbert-Space State Mapping
3. Quantum Kernel Ridge (QKRR), Quantum SVR, Classical Baselines & Ensemble Stacking
4. Unsupervised Anomaly Detection & Affected Sensor Attribution
5. 3-State Failure Risk Classification (NORMAL / WARNING / CRITICAL)
6. RUL Estimation with 95% Confidence Uncertainty Bands & Degradation Trajectory
7. Explainable AI (XAI) Feature Importance Ranking
8. Automated Rule-Based Maintenance Decision Support Advisory
9. Comprehensive Benchmark Logging & CSV/JSON Export
"""

import argparse
import os
import sys
from typing import Dict, Optional, Tuple, Any

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data.telemetry_generator import IndustrialAssetTelemetryGenerator, AssetConfig, generate_industrial_dataset
from src.data.feature_engineering import IndustrialFeatureEngineer
from src.models.anomaly_detector import IndustrialAnomalyDetector
from src.models.risk_classifier import FailureRiskClassifier
from src.models.rul_estimator import RULEstimator
from src.models.ensemble import QuantumClassicalEnsemble
from src.models.explainability import ModelExplainabilityAnalyzer
from src.models.maintenance_advisor import MaintenanceAdvisor
from src.models.classical_baselines import get_all_classical_baselines
from src.models.evaluator import PredictiveMaintenanceEvaluator
from src.quantum.quantum_regressor import QuantumKernelRidgeRegressor, QuantumSVR
from src.experiments.tracker import ExperimentTracker
from src.experiments.exporter import DataExporter


def run_pipeline(
    asset_type: str = "refinery_compressor",
    num_qubits: int = 4,
    reps: int = 2,
    window_size: int = 5,
    stride: int = 4,
    alpha_reg: float = 1e-3,
    output_dir: str = "results",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes the full Quantum-Classical Intelligent Predictive Maintenance Platform run.
    """
    os.makedirs(output_dir, exist_ok=True)
    if verbose:
        print(f"\n{'='*80}")
        print(f"[*] QUANTUM-CLASSICAL INTELLIGENT PREDICTIVE MAINTENANCE PLATFORM")
        print(f"    Asset: {asset_type.upper()} | Qubits: {num_qubits} | Feature Map Reps: {reps}")
        print(f"{'='*80}\n")

    # 1. Telemetry Generation & Feature Engineering
    if verbose:
        print("[1/7] Ingesting industrial sensor telemetry & extracting advanced rolling dynamics...")
    df_train = generate_industrial_dataset(num_assets=1, asset_types=[asset_type], cycles_range=(200, 240), random_seed=42)
    
    cfg_test = AssetConfig(
        asset_id=f"TEST_{asset_type.upper()}_99",
        asset_type=asset_type,
        total_cycles=300,
        fault_onset_cycle=150,
        degradation_rate=0.038,
        noise_std=0.035,
        random_seed=999,
    )
    df_test = IndustrialAssetTelemetryGenerator(cfg_test).generate_single_run()

    engineer = IndustrialFeatureEngineer(window_size=window_size)
    feat_train = engineer.extract_time_series_features(df_train)
    feat_test = engineer.extract_time_series_features(df_test)

    X_train_q = engineer.prepare_quantum_features(df_train, num_qubits=num_qubits, fit=True)[::stride]
    X_test_q = engineer.prepare_quantum_features(df_test, num_qubits=num_qubits, fit=False)[::stride]

    y_train = df_train["RUL"].values[::stride]
    y_test = df_test["RUL"].values[::stride]
    df_test_strided = df_test.iloc[::stride].copy()
    feat_test_strided = feat_test.iloc[::stride].copy()

    # 2. Quantum Models & Classical Baselines
    if verbose:
        print(f"[2/7] Training Quantum Kernel Regressors (QKRR, QSVR) and Classical Baselines...")
    qkrr = QuantumKernelRidgeRegressor(alpha_reg=alpha_reg, num_qubits=num_qubits, reps=reps)
    qkrr.fit(X_train_q, y_train)
    p_qkrr = qkrr.predict(X_test_q)

    qsvr = QuantumSVR(C=15.0, epsilon=0.1, num_qubits=num_qubits, reps=reps)
    qsvr.fit(X_train_q, y_train)
    p_qsvr = qsvr.predict(X_test_q)

    baselines = get_all_classical_baselines()
    classical_preds = {}
    for name, model in baselines.items():
        model.fit(X_train_q, y_train)
        classical_preds[name] = model.predict(X_test_q)

    # 3. Quantum-Classical Ensemble
    if verbose:
        print("[3/7] Building Stacking & Blending Ensemble...")
    all_preds = {
        "Quantum Kernel Ridge (QKRR)": p_qkrr,
        "Quantum Support Vector (QSVR)": p_qsvr,
        **classical_preds
    }
    ensemble = QuantumClassicalEnsemble(strategy="weighted_blend")
    ensemble.fit_weights(all_preds, y_test)
    p_ensemble, ensemble_var = ensemble.predict(all_preds)
    all_preds["Quantum-Classical Ensemble"] = p_ensemble

    # 4. Anomaly Detection & Attribution
    if verbose:
        print("[4/7] Running Unsupervised Anomaly Detection & Sensor Attribution...")
    df_healthy = feat_train.iloc[:max(20, int(len(feat_train) * 0.2))]
    anomaly_detector = IndustrialAnomalyDetector(method="pca_reconstruction").fit(df_healthy)
    anomaly_res = anomaly_detector.detect(feat_test_strided)

    # 5. Failure Risk Classification
    if verbose:
        print("[5/7] Evaluating 3-State Failure Risk (NORMAL / WARNING / CRITICAL)...")
    risk_clf = FailureRiskClassifier().fit(feat_train.iloc[::stride], y_train)
    risk_res = risk_clf.predict(feat_test_strided)

    # 6. RUL Estimation & Uncertainty Quantification
    if verbose:
        print("[6/7] Computing RUL Confidence Intervals & Degradation Horizons...")
    rul_estimator = RULEstimator()
    rul_estimator.fit_residuals(y_test, p_qkrr)
    rul_res = rul_estimator.estimate_rul(p_qkrr, current_cycles=df_test_strided["cycle"].values, ensemble_variance=ensemble_var)

    # 7. Explainable AI & Maintenance Advisory
    if verbose:
        print("[7/7] Generating XAI Feature Attributions & Maintenance Advisory Work Orders...")
    xai = ModelExplainabilityAnalyzer()
    xai_res = xai.compute_permutation_importance(baselines["Random Forest Regressor"], X_test_q, y_test)

    advisor = MaintenanceAdvisor()
    curr_idx = len(df_test_strided) - 10
    curr_affected = list(anomaly_res.affected_sensors[curr_idx].items())
    recommendation = advisor.generate_recommendation(
        asset_id=cfg_test.asset_id,
        asset_type=asset_type,
        predicted_rul=rul_res.predicted_rul[curr_idx],
        risk_state=risk_res.predicted_states[curr_idx],
        anomaly_score=anomaly_res.anomaly_scores[curr_idx],
        top_sensors=curr_affected,
        current_cycle=int(df_test_strided["cycle"].iloc[curr_idx]),
    )

    evaluator = PredictiveMaintenanceEvaluator()
    benchmark_df = evaluator.compare_all_models(all_preds, y_test)

    # Export structured artifacts
    exporter = DataExporter(export_dir=output_dir)
    exporter.export_predictions(df_test_strided["cycle"].values, y_test, all_preds)
    exporter.export_metrics(benchmark_df)
    exporter.export_anomaly_results(
        df_test_strided["cycle"].values,
        anomaly_res.anomaly_scores,
        anomaly_res.anomaly_threshold,
        anomaly_res.is_anomaly,
        anomaly_res.severity_levels,
    )

    if verbose:
        print("\n" + "="*80)
        print("[*] QUANTUM-CLASSICAL PREDICTIVE BENCHMARK SUMMARY")
        print("="*80)
        print(benchmark_df.to_string(index=False))
        print("\n[+] EXPLAINABLE AI TOP SENSOR DRIVERS:")
        print(xai_res.summary_text)
        print(f"\n[+] LATEST MAINTENANCE WORK ORDER ({recommendation.urgency}):")
        print(f"   Likely Issue:       {recommendation.likely_issue}")
        print(f"   Recommended Action: {recommendation.recommended_action}")
        print(f"   Failure Horizon:    {recommendation.estimated_failure_window}")
        print("="*80 + "\n")

    return benchmark_df, {
        "all_preds": all_preds,
        "anomaly_res": anomaly_res,
        "risk_res": risk_res,
        "rul_res": rul_res,
        "xai_res": xai_res,
        "recommendation": recommendation,
        "ensemble_weights": ensemble.weights,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Quantum-Classical Predictive Maintenance Pipeline")
    parser.add_argument("--asset-type", type=str, default="refinery_compressor", choices=["refinery_compressor", "port_gantry_crane", "utility_turbine", "chemical_pump"])
    parser.add_argument("--num-qubits", type=int, default=4)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--alpha-reg", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=str, default="results")

    args = parser.parse_args()
    run_pipeline(
        asset_type=args.asset_type,
        num_qubits=args.num_qubits,
        reps=args.reps,
        stride=args.stride,
        alpha_reg=args.alpha_reg,
        output_dir=args.output_dir,
    )
