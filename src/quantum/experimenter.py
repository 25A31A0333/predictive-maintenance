"""
Quantum Hyperparameter Sweeper and Experimentation Lab.

Runs systematic quantum parameter sweeps over:
- Number of qubits (2, 3, 4, 6)
- Feature map architectures (Angle, ZZ-entangling, Pauli-Z)
- Circuit repetitions / depth (1, 2, 3)
- Regularization parameters (lambda / alpha)
- Simulator backends vs optional IBM Quantum QPUs

Logs all runs automatically via ExperimentTracker.
"""

import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.quantum.quantum_regressor import QuantumKernelRidgeRegressor
from src.experiments.tracker import ExperimentTracker, ExperimentRun


class QuantumExperimenter:
    """
    Executes and records quantum hyperparameter experiments for predictive maintenance.
    """

    def __init__(self, tracker: Optional[ExperimentTracker] = None):
        self.tracker = tracker or ExperimentTracker()

    def run_sweep(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        asset_type: str = "refinery_compressor",
        qubit_options: Optional[List[int]] = None,
        reps_options: Optional[List[int]] = None,
        feature_maps: Optional[List[str]] = None,
        alpha_options: Optional[List[float]] = None,
        backend: str = "default.qubit",
        random_seed: int = 42,
    ) -> pd.DataFrame:
        """
        Sweeps hyperparameter grid and logs every experiment run.
        """
        qubit_options = qubit_options or [3, 4]
        reps_options = reps_options or [1, 2]
        feature_maps = feature_maps or ["zz", "angle"]
        alpha_options = alpha_options or [1e-3]

        runs_data = []

        for q in qubit_options:
            # Adjust dimension to match current qubit count if needed
            X_tr_q = X_train[:, :q] if X_train.shape[1] >= q else np.pad(X_train, ((0,0), (0, q - X_train.shape[1])))
            X_te_q = X_test[:, :q] if X_test.shape[1] >= q else np.pad(X_test, ((0,0), (0, q - X_test.shape[1])))

            for fmap in feature_maps:
                for rep in reps_options:
                    for alpha in alpha_options:
                        t0 = time.time()
                        
                        qkrr = QuantumKernelRidgeRegressor(
                            alpha_reg=alpha,
                            num_qubits=q,
                            feature_map=fmap,
                            reps=rep,
                            backend=backend,
                        )
                        qkrr.fit(X_tr_q, y_train)
                        preds = qkrr.predict(X_te_q)
                        
                        elapsed = time.time() - t0

                        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
                        mae = float(mean_absolute_error(y_test, preds))
                        r2 = float(r2_score(y_test, preds))

                        # Log experiment run
                        exp_run = self.tracker.log_run(
                            model_name=f"QKRR_{fmap.upper()}_q{q}_r{rep}",
                            asset_type=asset_type,
                            num_qubits=q,
                            circuit_reps=rep,
                            feature_map=fmap,
                            rmse=rmse,
                            mae=mae,
                            r2_score=r2,
                            execution_time_seconds=elapsed,
                            backend=backend,
                            random_seed=random_seed,
                            notes=f"alpha={alpha}",
                        )
                        runs_data.append({
                            "Run ID": exp_run.run_id,
                            "Qubits": q,
                            "Feature Map": fmap,
                            "Reps": rep,
                            "Alpha": alpha,
                            "RMSE": round(rmse, 2),
                            "MAE": round(mae, 2),
                            "R2": round(r2, 3),
                            "Time (s)": round(elapsed, 2),
                        })

        return pd.DataFrame(runs_data)
