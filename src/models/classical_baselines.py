"""
Classical Machine Learning Baselines for Industrial Predictive Maintenance.

Provides standardized implementations of classical SVR (RBF Kernel),
Kernel Ridge (RBF Kernel), Random Forest, and Linear Ridge Regressors for
benchmarking against Quantum Machine Learning models.
"""

from typing import Dict, Optional
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.svm import SVR


class ClassicalRBFRegressor:
    """Support Vector Regression using Classical Gaussian RBF Kernel."""

    def __init__(self, C: float = 10.0, gamma: str = "scale", epsilon: float = 0.1):
        self.model = SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class ClassicalKernelRidgeRBF:
    """Kernel Ridge Regression using Classical Gaussian RBF Kernel."""

    def __init__(self, alpha: float = 1e-3, gamma: float = 0.5):
        self.model = KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class ClassicalRandomForestRegressor:
    """Random Forest Regressor ensemble baseline."""

    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = 8, random_state: int = 42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class ClassicalRidgeRegressor:
    """Linear Ridge Regressor baseline."""

    def __init__(self, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


def get_all_classical_baselines(random_state: int = 42) -> Dict[str, object]:
    """Returns a dictionary of all classical benchmark models."""
    return {
        "Classical SVR (RBF Kernel)": ClassicalRBFRegressor(),
        "Classical Kernel Ridge (RBF)": ClassicalKernelRidgeRBF(),
        "Random Forest Regressor": ClassicalRandomForestRegressor(random_state=random_state),
        "Linear Ridge Regressor": ClassicalRidgeRegressor(),
    }
