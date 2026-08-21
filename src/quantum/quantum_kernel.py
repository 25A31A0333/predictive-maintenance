"""
Quantum Kernel Matrix Computer for Industrial Time-Series Telemetry.

Computes fidelity-based Quantum Kernels K(x, x') = |<phi(x)|phi(x')>|^2 using
quantum state-vector overlap measurement. Supports symmetric matrix caching,
train-test evaluations, and scikit-learn compatible kernel interfaces.
"""

from typing import Callable, Optional, Tuple, Union
import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else range(kwargs.get('total', 0))

from src.quantum.feature_maps import (
    AngleFeatureMap,
    ZZFeatureMap,
    build_quantum_feature_circuit,
    HAS_PENNYLANE,
)

if HAS_PENNYLANE:
    import pennylane as qml


class QuantumKernel:
    """
    Computes Quantum Kernel Gram matrices using PennyLane quantum state fidelity.
    """

    def __init__(
        self,
        num_qubits: int = 4,
        feature_map: Optional[Union[str, ZZFeatureMap, AngleFeatureMap]] = "zz",
        reps: int = 2,
        entanglement: str = "linear",
        backend: str = "default.qubit",
    ):
        self.num_qubits = num_qubits
        self.reps = reps
        self.entanglement = entanglement
        self.backend_name = backend

        if isinstance(feature_map, str):
            self.feature_map = build_quantum_feature_circuit(
                feature_map_type=feature_map,
                num_qubits=num_qubits,
                reps=reps,
                entanglement=entanglement,
            )
        else:
            self.feature_map = feature_map or build_quantum_feature_circuit("zz", num_qubits, reps, entanglement)

        self._init_quantum_node()

    def _init_quantum_node(self):
        """Initializes the PennyLane QNode for kernel fidelity evaluation."""
        if not HAS_PENNYLANE:
            return

        self.dev = qml.device(self.backend_name, wires=self.num_qubits)

        @qml.qnode(self.dev)
        def kernel_circuit(x1: np.ndarray, x2: np.ndarray):
            # Apply U(x1)
            self.feature_map.pennylane_circuit(x1, wires=list(range(self.num_qubits)))
            # Apply adjoint U^\dagger(x2)
            qml.adjoint(self.feature_map.pennylane_circuit)(x2, wires=list(range(self.num_qubits)))
            # Measure probability of projecting onto |00...0>
            return qml.probs(wires=list(range(self.num_qubits)))

        self._kernel_circuit = kernel_circuit

    def evaluate_pair(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Evaluates the quantum kernel value between two feature vectors:
        K(x1, x2) = |<phi(x1)|phi(x2)>|^2
        """
        if HAS_PENNYLANE:
            probs = self._kernel_circuit(x1, x2)
            # Probability of all-zero bitstring |00...0> is index 0
            return float(probs[0])
        else:
            # Vectorized NumPy fallback (classical RBF simulation)
            gamma = 1.0 / (2.0 * self.num_qubits)
            return float(np.exp(-gamma * np.sum((x1 - x2) ** 2)))

    def compute_matrix(
        self,
        X1: np.ndarray,
        X2: Optional[np.ndarray] = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Computes the Gram kernel matrix between dataset X1 and dataset X2.
        If X2 is None, computes the symmetric square Gram matrix K(X1, X1).
        """
        n1 = len(X1)
        is_symmetric = X2 is None

        if is_symmetric:
            X2 = X1
            n2 = n1
            K = np.eye(n1, dtype=float)  # Diagonal elements are identically 1.0
            
            total_pairs = (n1 * (n1 - 1)) // 2
            pbar = tqdm(total=total_pairs, desc="Quantum Kernel (Symmetric)", disable=not show_progress)
            
            for i in range(n1):
                for j in range(i + 1, n2):
                    val = self.evaluate_pair(X1[i], X2[j])
                    K[i, j] = val
                    K[j, i] = val
                    pbar.update(1)
            pbar.close()
            
        else:
            n2 = len(X2)
            K = np.zeros((n1, n2), dtype=float)
            total_pairs = n1 * n2
            pbar = tqdm(total=total_pairs, desc="Quantum Kernel (Rectangular)", disable=not show_progress)
            
            for i in range(n1):
                for j in range(n2):
                    K[i, j] = self.evaluate_pair(X1[i], X2[j])
                    pbar.update(1)
            pbar.close()

        # Numerical cleanup: ensure kernel matrix values lie in [0, 1]
        K = np.clip(K, 0.0, 1.0)
        return K

    def __call__(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        """Scikit-learn compatible callable interface."""
        return self.compute_matrix(X1, X2)


def compute_quantum_kernel_matrix(
    X_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    num_qubits: int = 4,
    feature_map_type: str = "zz",
    reps: int = 2,
    show_progress: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Convenience function for quick kernel Gram matrix computation."""
    qk = QuantumKernel(
        num_qubits=num_qubits,
        feature_map=feature_map_type,
        reps=reps,
    )
    K_train = qk.compute_matrix(X_train, show_progress=show_progress)
    if X_test is not None:
        K_test = qk.compute_matrix(X_test, X_train, show_progress=show_progress)
        return K_train, K_test
    return K_train
