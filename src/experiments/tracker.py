"""
Experiment Tracking & Reproducibility Module for Quantum AI Predictive Maintenance.

Records:
- Configuration (qubits, feature map, circuit depth, alpha, asset type)
- Model architecture (QKRR, QSVR, Classical baselines, Ensemble)
- Benchmark metrics (RMSE, MAE, R2, Earliness lead time, False alarm rate)
- Execution time (seconds) & random seed
- Timestamps and reproducibility parameters
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import os
import json
import time
from datetime import datetime
import pandas as pd


@dataclass
class ExperimentRun:
    """Represents a single benchmarked experiment run."""
    run_id: str
    timestamp: str
    model_name: str
    asset_type: str
    num_qubits: int
    circuit_reps: int
    feature_map: str
    backend: str
    rmse: float
    mae: float
    r2_score: float
    earliness_cycles: int
    false_alarm_rate_pct: float
    execution_time_seconds: float
    random_seed: int
    notes: Optional[str] = None


class ExperimentTracker:
    """
    Manages persistent experiment logging and comparison under experiments/runs/.
    """

    def __init__(self, experiment_dir: str = "experiments"):
        self.experiment_dir = experiment_dir
        self.runs_dir = os.path.join(experiment_dir, "runs")
        self.summary_csv = os.path.join(experiment_dir, "experiment_history.csv")
        os.makedirs(self.runs_dir, exist_ok=True)

    def log_run(
        self,
        model_name: str,
        asset_type: str,
        num_qubits: int,
        circuit_reps: int,
        feature_map: str,
        rmse: float,
        mae: float,
        r2_score: float,
        earliness_cycles: int = 0,
        false_alarm_rate_pct: float = 0.0,
        execution_time_seconds: float = 0.0,
        backend: str = "default.qubit",
        random_seed: int = 42,
        notes: Optional[str] = None,
    ) -> ExperimentRun:
        """Logs an experiment run to JSON and updates the global summary CSV."""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        run_id = f"EXP_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{model_name.replace(' ', '_')[:10]}"

        run = ExperimentRun(
            run_id=run_id,
            timestamp=timestamp,
            model_name=model_name,
            asset_type=asset_type,
            num_qubits=num_qubits,
            circuit_reps=circuit_reps,
            feature_map=feature_map,
            backend=backend,
            rmse=round(float(rmse), 3),
            mae=round(float(mae), 3),
            r2_score=round(float(r2_score), 4),
            earliness_cycles=int(earliness_cycles),
            false_alarm_rate_pct=round(float(false_alarm_rate_pct), 2),
            execution_time_seconds=round(float(execution_time_seconds), 3),
            random_seed=random_seed,
            notes=notes,
        )

        # 1. Save single JSON artifact
        json_path = os.path.join(self.runs_dir, f"{run_id}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(run), f, indent=2)

        # 2. Append to summary CSV
        run_dict = asdict(run)
        if os.path.exists(self.summary_csv):
            df_hist = pd.read_csv(self.summary_csv)
            df_hist = pd.concat([df_hist, pd.DataFrame([run_dict])], ignore_index=True)
        else:
            df_hist = pd.DataFrame([run_dict])
            
        df_hist.to_csv(self.summary_csv, index=False)
        return run

    def list_runs(self) -> pd.DataFrame:
        """Retrieves all tracked experiment runs as a DataFrame."""
        if os.path.exists(self.summary_csv):
            return pd.read_csv(self.summary_csv)
        return pd.DataFrame()
