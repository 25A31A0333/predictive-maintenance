"""
Quantum Regressors for Industrial Machinery Remaining Useful Life (RUL) Forecasting.

Implements Quantum Kernel Ridge Regression (QKRR), Quantum Support Vector Regression (QSVR),
and Variational Quantum Regressors (VQR) for sensitive early-stage equipment degradation tracking.
"""

from typing import Optional, Union
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.svm import SVR

from src.quantum.quantum_kernel import QuantumKernel
from src.quantum.feature_maps import HAS_PENNYLANE

if HAS_PENNYLANE:
    import pennylane as qml


class QuantumKernelRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Quantum Kernel Ridge Regressor (QKRR).
    Fits a regularized linear model in the Quantum Hilbert Reproducing Kernel space:
    min_alpha || K alpha - y ||^2 + alpha_reg * alpha^T K alpha
    Dual solution: alpha = (K + alpha_reg * I)^(-1) y
    """

    def __init__(
        self,
        alpha_reg: float = 1e-3,
        num_qubits: int = 4,
        feature_map_type: str = "zz",
        reps: int = 2,
        entanglement: str = "linear",
        feature_map: Optional[str] = None,
        backend: str = "default.qubit",
    ):
        self.alpha_reg = alpha_reg
        self.num_qubits = num_qubits
        self.feature_map_type = feature_map or feature_map_type
        self.reps = reps
        self.entanglement = entanglement
        self.backend = backend

        self.qk = QuantumKernel(
            num_qubits=num_qubits,
            feature_map=self.feature_map_type,
            reps=reps,
            entanglement=entanglement,
            backend=backend,
        )
        self.X_train_: Optional[np.ndarray] = None
        self.dual_coef_: Optional[np.ndarray] = None
        self.K_train_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray, K_train: Optional[np.ndarray] = None):
        """Fits the QKRR model on training data or a precomputed Gram matrix."""
        self.X_train_ = np.asarray(X)
        y = np.asarray(y, dtype=float)
        n = len(X)

        if K_train is not None:
            self.K_train_ = K_train
        else:
            self.K_train_ = self.qk.compute_matrix(self.X_train_)

        # Regularized Inversion: (K + alpha_reg * I)^(-1) y
        A = self.K_train_ + self.alpha_reg * np.eye(n)
        self.dual_coef_ = np.linalg.solve(A, y)
        return self

    def predict(self, X: np.ndarray, K_test: Optional[np.ndarray] = None) -> np.ndarray:
        """Predicts target values (e.g. RUL or degradation index)."""
        if self.dual_coef_ is None or self.X_train_ is None:
            raise ValueError("The QKRR model has not been fitted yet.")

        if K_test is not None:
            K = K_test
        else:
            K = self.qk.compute_matrix(X, self.X_train_)

        return np.dot(K, self.dual_coef_)


class QuantumSVR(BaseEstimator, RegressorMixin):
    """
    Quantum Support Vector Regressor (QSVR).
    Equips classical Support Vector Regression with Quantum Kernel Gram matrices.
    """

    def __init__(
        self,
        C: float = 10.0,
        epsilon: float = 0.1,
        num_qubits: int = 4,
        feature_map_type: str = "zz",
        reps: int = 2,
    ):
        self.C = C
        self.epsilon = epsilon
        self.num_qubits = num_qubits
        self.feature_map_type = feature_map_type
        self.reps = reps

        self.qk = QuantumKernel(
            num_qubits=num_qubits,
            feature_map=feature_map_type,
            reps=reps,
        )
        self.svr = SVR(kernel="precomputed", C=self.C, epsilon=self.epsilon)
        self.X_train_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray, K_train: Optional[np.ndarray] = None):
        self.X_train_ = np.asarray(X)
        if K_train is None:
            K_train = self.qk.compute_matrix(self.X_train_)
        self.svr.fit(K_train, y)
        return self

    def predict(self, X: np.ndarray, K_test: Optional[np.ndarray] = None) -> np.ndarray:
        if self.X_train_ is None:
            raise ValueError("Model is not fitted.")
        if K_test is None:
            K_test = self.qk.compute_matrix(X, self.X_train_)
        return self.svr.predict(K_test)


class VariationalQuantumRegressor(BaseEstimator, RegressorMixin):
    """
    Variational Quantum Circuit Regressor (VQR).
    Parametrized Quantum Neural Network (QNN) trained via analytic quantum gradient descent.
    """

    def __init__(
        self,
        num_qubits: int = 4,
        num_layers: int = 2,
        learning_rate: float = 0.05,
        epochs: int = 30,
        random_seed: int = 42,
    ):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_seed = random_seed
        self.weights_: Optional[np.ndarray] = None
        self.bias_: float = 0.0
        self.target_mean_: float = 0.0
        self.target_std_: float = 1.0

        if HAS_PENNYLANE:
            self.dev = qml.device("default.qubit", wires=self.num_qubits)
            self._init_qnode()

    def _init_qnode(self):
        @qml.qnode(self.dev, diff_method="parameter-shift")
        def circuit(weights, x):
            # 1. Feature embedding
            for i in range(self.num_qubits):
                qml.Hadamard(wires=i)
                qml.RY(x[i % len(x)], wires=i)

            # 2. Variational layers
            for l in range(self.num_layers):
                for i in range(self.num_qubits):
                    qml.Rot(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=i)
                for i in range(self.num_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                if self.num_qubits > 2:
                    qml.CNOT(wires=[self.num_qubits - 1, 0])

            # 3. Measurement
            return qml.expval(qml.PauliZ(0))

        self._circuit = circuit

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        self.target_mean_ = float(np.mean(y))
        self.target_std_ = float(np.std(y)) if np.std(y) > 1e-6 else 1.0
        y_norm = (y - self.target_mean_) / self.target_std_

        np.random.seed(self.random_seed)
        shape = (self.num_layers, self.num_qubits, 3)
        self.weights_ = 0.1 * np.random.randn(*shape)
        self.bias_ = 0.0

        if not HAS_PENNYLANE:
            return self

        opt = qml.AdamOptimizer(stepsize=self.learning_rate)

        def cost(w, b, x_batch, y_batch):
            preds = [self._circuit(w, xi) + b for xi in x_batch]
            return np.mean((np.array(preds) - y_batch) ** 2)

        # Mini-batch training loop
        n_samples = len(X)
        batch_size = min(16, n_samples)

        for _ in range(self.epochs):
            indices = np.random.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                batch_idx = indices[start:start + batch_size]
                xb = X[batch_idx]
                yb = y_norm[batch_idx]

                # Update weights and bias
                (self.weights_, self.bias_), _ = opt.step_and_cost(
                    lambda params: cost(params[0], params[1], xb, yb),
                    (self.weights_, self.bias_),
                )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not HAS_PENNYLANE or self.weights_ is None:
            # Fallback mean prediction
            return np.full(len(X), self.target_mean_)

        raw_preds = np.array([self._circuit(self.weights_, xi) + self.bias_ for xi in X])
        # Rescale back to original target units
        return raw_preds * self.target_std_ + self.target_mean_
