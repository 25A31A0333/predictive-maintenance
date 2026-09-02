"""
Quantum-Classical Ensemble and Stacking Module for Predictive Maintenance.

Combines predictions from:
1. Quantum Kernel Ridge Regression (QKRR)
2. Quantum Support Vector Regression (QSVR)
3. Classical Support Vector Regression (SVR RBF)
4. Random Forest Regressor
5. Linear Ridge Baseline

Supports:
- Weighted Blending (optimal inverse-RMSE or quadratic programming weights)
- Meta-Learner Stacking (Ridge/Linear meta-model on out-of-fold predictions)
- Ensemble variance computation for uncertainty estimation.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error


class QuantumClassicalEnsemble:
    """
    Ensemble regressor combining Quantum Kernel Methods and Classical ML models.
    """

    def __init__(
        self,
        strategy: str = "weighted_blend",  # 'weighted_blend' or 'stacking'
        model_names: Optional[List[str]] = None,
    ):
        self.strategy = strategy
        self.model_names = model_names or [
            "Quantum Kernel Ridge (QKRR)",
            "Quantum Support Vector (QSVR)",
            "Classical SVR (Gaussian RBF)",
            "Random Forest Regressor",
            "Linear Ridge Baseline",
        ]
        self.weights: Dict[str, float] = {}
        self.meta_model: Optional[Ridge] = None

    def fit_weights(self, model_predictions: Dict[str, np.ndarray], y_true: np.ndarray) -> "QuantumClassicalEnsemble":
        """
        Calculates optimal model blending weights or trains a meta-learner using validation predictions.
        """
        y_true = np.asarray(y_true, dtype=float)
        active_models = [m for m in self.model_names if m in model_predictions]
        
        if not active_models:
            active_models = list(model_predictions.keys())

        if self.strategy == "weighted_blend":
            # Compute inverse RMSE weights: w_i = (1 / RMSE_i^2) / sum(1 / RMSE_j^2)
            inv_mse_list = []
            for name in active_models:
                p = np.asarray(model_predictions[name], dtype=float)
                mse = max(1e-4, mean_squared_error(y_true, p))
                inv_mse_list.append(1.0 / mse)

            total_inv = sum(inv_mse_list)
            for idx, name in enumerate(active_models):
                self.weights[name] = round(inv_mse_list[idx] / total_inv, 4)

        elif self.strategy == "stacking":
            # Stack predictions as feature matrix for meta-learner
            P_mat = np.column_stack([model_predictions[m] for m in active_models])
            self.meta_model = Ridge(alpha=1.0, positive=True, fit_intercept=True)
            self.meta_model.fit(P_mat, y_true)
            
            # Extract normalized weights from meta-model coefficients
            coefs = self.meta_model.coef_
            total_c = sum(coefs) if sum(coefs) > 0 else 1.0
            for idx, name in enumerate(active_models):
                self.weights[name] = round(float(coefs[idx] / total_c), 4)

        return self

    def predict(self, model_predictions: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combines model predictions into an ensemble forecast and calculates ensemble variance.
        
        Returns:
            Tuple of (ensemble_predictions, ensemble_variance).
        """
        active_models = [m for m in self.weights if m in model_predictions]
        if not active_models:
            active_models = list(model_predictions.keys())
            # Equal weighting fallback
            w_eq = 1.0 / len(active_models)
            self.weights = {m: w_eq for m in active_models}

        P_mat = np.column_stack([model_predictions[m] for m in active_models])
        w_vec = np.array([self.weights[m] for m in active_models])
        w_vec = w_vec / np.sum(w_vec)  # Ensure exact sum to 1.0

        if self.strategy == "stacking" and self.meta_model is not None:
            ensemble_pred = self.meta_model.predict(P_mat)
        else:
            ensemble_pred = np.dot(P_mat, w_vec)

        # Ensemble Variance: Weighted variance of individual model predictions around the ensemble mean
        # Var = sum(w_i * (p_i - p_ens)^2)
        diff_sq = (P_mat - ensemble_pred[:, np.newaxis]) ** 2
        ensemble_var = np.dot(diff_sq, w_vec)
        ensemble_var = np.maximum(1.0, ensemble_var)

        return np.maximum(0.0, ensemble_pred), ensemble_var
