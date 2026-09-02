"""
Unit tests for Quantum Experimentation Lab, Tracker, and Exporter.
"""

import os
import shutil
import numpy as np
import pandas as pd
import pytest
from src.experiments.tracker import ExperimentTracker, ExperimentRun
from src.experiments.exporter import DataExporter
from src.quantum.experimenter import QuantumExperimenter


def test_experiment_tracker_and_exporter(tmp_path):
    exp_dir = str(tmp_path / "test_experiments")
    tracker = ExperimentTracker(experiment_dir=exp_dir)

    run = tracker.log_run(
        model_name="QKRR_Test",
        asset_type="refinery_compressor",
        num_qubits=4,
        circuit_reps=2,
        feature_map="zz",
        rmse=12.4,
        mae=9.8,
        r2_score=0.96,
        earliness_cycles=35,
        execution_time_seconds=1.25,
    )
    assert isinstance(run, ExperimentRun)
    assert os.path.exists(os.path.join(exp_dir, "runs", f"{run.run_id}.json"))

    df_runs = tracker.list_runs()
    assert len(df_runs) == 1
    assert df_runs.iloc[0]["model_name"] == "QKRR_Test"

    # Test Exporter
    export_dir = str(tmp_path / "test_results")
    exporter = DataExporter(export_dir=export_dir)
    
    p_path = exporter.export_predictions(
        cycles=np.arange(1, 11),
        y_true_rul=np.linspace(100, 10, 10),
        predictions={"QKRR": np.linspace(98, 12, 10)},
    )
    assert os.path.exists(p_path)


def test_quantum_experimenter(tmp_path):
    exp_dir = str(tmp_path / "test_sweeps")
    tracker = ExperimentTracker(experiment_dir=exp_dir)
    exp = QuantumExperimenter(tracker=tracker)

    np.random.seed(42)
    X_train = np.random.uniform(0.0, np.pi, size=(8, 4))
    y_train = np.linspace(100, 20, 8)
    X_test = np.random.uniform(0.0, np.pi, size=(4, 4))
    y_test = np.linspace(80, 10, 4)

    df_sweep = exp.run_sweep(
        X_train, y_train, X_test, y_test,
        asset_type="port_gantry_crane",
        qubit_options=[2],
        reps_options=[1],
        feature_maps=["angle"],
        alpha_options=[1e-2],
    )
    assert len(df_sweep) == 1
    assert "RMSE" in df_sweep.columns
    assert "Time (s)" in df_sweep.columns
