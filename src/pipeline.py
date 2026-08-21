"""
End-to-End Quantum AI/ML Predictive Maintenance Pipeline.

Executes data generation, quantum feature encoding, kernel matrix computation,
classical vs quantum benchmark training, evaluation metrics, and degradation trajectory analysis.
"""

import argparse
import os
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data.telemetry_generator import (
    IndustrialAssetTelemetryGenerator,
    AssetConfig,
    generate_industrial_dataset,
    prepare_quantum_timeseries_dataset,
)
from src.quantum.quantum_kernel import QuantumKernel
from src.quantum.quantum_regressor import (
    QuantumKernelRidgeRegressor,
    QuantumSVR,
    VariationalQuantumRegressor,
)
from src.models.classical_baselines import get_all_classical_baselines
from src.models.evaluator import PredictiveMaintenanceEvaluator


def run_pipeline(
    asset_type: str = "refinery_compressor",
    num_qubits: int = 4,
    reps: int = 2,
    window_size: int = 5,
    stride: int = 2,
    alpha_reg: float = 1e-3,
    output_dir: str = "results",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Executes the full Quantum Predictive Maintenance training and benchmarking run.
    """
    os.makedirs(output_dir, exist_ok=True)
    if verbose:
        print(f"\n{'='*75}")
        print(f"[*] STARTING QUANTUM AI PREDICTIVE MAINTENANCE PIPELINE")
        print(f"    Asset: {asset_type.upper()} | Qubits: {num_qubits} | Circuit Reps: {reps}")
        print(f"{'='*75}\n")

    # 1. Generate Fleet Telemetry
    if verbose:
        print("[1/5] Generating multi-asset industrial sensor telemetry...")
    df_train = generate_industrial_dataset(
        num_assets=4,
        asset_types=[asset_type],
        cycles_range=(280, 340),
        random_seed=42,
    )
    
    # Generate a dedicated unseen test asset
    cfg_test = AssetConfig(
        asset_id=f"TEST_{asset_type.upper()}_99",
        asset_type=asset_type,
        total_cycles=320,
        fault_onset_cycle=170,
        degradation_rate=0.038,
        noise_std=0.035,
        random_seed=999,
    )
    df_test = IndustrialAssetTelemetryGenerator(cfg_test).generate_single_run()
    df_test["asset_id"] = cfg_test.asset_id
    df_test["asset_type"] = cfg_test.asset_type

    # 2. Extract Quantum Time-Series Windows
    if verbose:
        print(f"[2/5] Preparing sliding windows and quantum feature encoding (n_qubits={num_qubits})...")
    X_train, y_train_rul, y_train_state, meta = prepare_quantum_timeseries_dataset(
        df_train,
        window_size=window_size,
        stride=stride,
        num_qubits=num_qubits,
        scale_range=(0.0, np.pi),
    )
    
    # Prepare test windows using training scaler and PCA
    X_test_raw = []
    y_test_rul = []
    sensor_matrix = df_test[meta["sensor_cols"]].values
    rul_test = df_test["RUL"].values

    for s_idx in range(0, len(df_test) - window_size + 1, stride):
        e_idx = s_idx + window_size
        win = sensor_matrix[s_idx:e_idx]
        feat = np.concatenate([
            np.mean(win, axis=0),
            np.std(win, axis=0),
            np.ptp(win, axis=0),
            (win[-1] - win[0]) / float(window_size),
        ])
        X_test_raw.append(feat)
        y_test_rul.append(rul_test[e_idx - 1])

    X_test_raw = np.array(X_test_raw)
    y_test_rul = np.array(y_test_rul, dtype=float)

    if meta["pca"] is not None:
        X_test_red = meta["pca"].transform(X_test_raw)
    else:
        X_test_red = X_test_raw[:, :num_qubits]
    X_test = meta["scaler"].transform(X_test_red)

    # Subsample training data if large for fast exact quantum statevector simulation
    if len(X_train) > 120:
        sub_indices = np.linspace(0, len(X_train) - 1, 120, dtype=int)
        X_train_sub = X_train[sub_indices]
        y_train_sub = y_train_rul[sub_indices]
    else:
        X_train_sub = X_train
        y_train_sub = y_train_rul

    # 3. Compute Quantum Kernel Gram Matrices
    if verbose:
        print(f"[3/5] Computing Quantum Kernel Gram Matrices (ZZ-Feature Map, Reps={reps})...")
    qk = QuantumKernel(num_qubits=num_qubits, feature_map="zz", reps=reps)
    K_train = qk.compute_matrix(X_train_sub, show_progress=verbose)
    K_test = qk.compute_matrix(X_test, X_train_sub, show_progress=verbose)

    # 4. Train Models
    if verbose:
        print("[4/5] Training Quantum Regressors & Classical Baselines...")
    
    predictions: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    # A. Quantum Kernel Ridge Regression (QKRR)
    qkrr = QuantumKernelRidgeRegressor(alpha_reg=alpha_reg, num_qubits=num_qubits, reps=reps)
    qkrr.fit(X_train_sub, y_train_sub, K_train=K_train)
    y_pred_qkrr = qkrr.predict(X_test, K_test=K_test)
    predictions["Quantum Kernel Ridge (QKRR)"] = (y_test_rul, y_pred_qkrr)

    # B. Quantum SVR
    qsvr = QuantumSVR(C=10.0, epsilon=0.1, num_qubits=num_qubits, reps=reps)
    qsvr.fit(X_train_sub, y_train_sub, K_train=K_train)
    y_pred_qsvr = qsvr.predict(X_test, K_test=K_test)
    predictions["Quantum Support Vector Regressor (QSVR)"] = (y_test_rul, y_pred_qsvr)

    # C. Classical Baselines
    classical_models = get_all_classical_baselines()
    for c_name, c_model in classical_models.items():
        c_model.fit(X_train_sub, y_train_sub)
        y_pred_c = c_model.predict(X_test)
        predictions[c_name] = (y_test_rul, y_pred_c)

    # 5. Evaluate and Benchmark
    if verbose:
        print("[5/5] Calculating Earliness of Detection, RMSE, MAE, and Downtime Cost Savings ($)...")
    evaluator = PredictiveMaintenanceEvaluator()
    benchmark_df = evaluator.benchmark_fleet(predictions)

    if verbose:
        print("\n" + "="*85)
        print("[BENCHMARK] RESULTS TABLE (QUANTUM VS CLASSICAL)")
        print("="*85)
        print(benchmark_df.to_string(index=False))
        print("="*85 + "\n")

    # Save benchmark table
    table_path = os.path.join(output_dir, "benchmark_metrics.csv")
    benchmark_df.to_csv(table_path, index=False)
    if verbose:
        print(f"[SUCCESS] Saved metrics table to: {table_path}")

    # Generate and save comparison plot
    plot_path = os.path.join(output_dir, "rul_forecast_comparison.png")
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_rul, label="True Actual RUL", color="black", linewidth=2.5, linestyle="--")
    plt.plot(y_pred_qkrr, label="Quantum Kernel Ridge (QKRR)", color="#00e5ff", linewidth=2.0)
    plt.plot(predictions["Classical SVR (RBF Kernel)"][1], label="Classical SVR (RBF)", color="#ff5252", linewidth=1.5, alpha=0.8)
    plt.plot(predictions["Random Forest Regressor"][1], label="Random Forest", color="#ffd600", linewidth=1.5, alpha=0.8)
    plt.axhline(y=50, color="red", linestyle=":", label="Critical Threshold (50 cycles)")
    plt.title(f"Predictive Maintenance: Quantum vs Classical RUL Forecast ({asset_type.replace('_', ' ').title()})", fontsize=14)
    plt.xlabel("Operational Time-Series Steps (Windows)", fontsize=12)
    plt.ylabel("Remaining Useful Life (Cycles)", fontsize=12)
    plt.legend(loc="upper right", frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()

    if verbose:
        print(f"[SUCCESS] Saved comparison plot to: {plot_path}")

    return benchmark_df, {
        "df_train": df_train,
        "df_test": df_test,
        "X_train": X_train_sub,
        "X_test": X_test,
        "y_test_rul": y_test_rul,
        "predictions": predictions,
        "K_train": K_train,
        "K_test": K_test,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum AI Predictive Maintenance Pipeline")
    parser.add_argument("--asset-type", type=str, default="refinery_compressor", choices=["refinery_compressor", "port_gantry_crane", "utility_turbine", "chemical_pump"])
    parser.add_argument("--num-qubits", type=int, default=4)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--alpha-reg", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    run_pipeline(
        asset_type=args.asset_type,
        num_qubits=args.num_qubits,
        reps=args.reps,
        alpha_reg=args.alpha_reg,
        output_dir=args.output_dir,
    )
