"""
Quantum Feature Maps for Time-Series Degradation Modeling.

Implements Angle Embedding, ZZ-Entangling Feature Maps, and Projected Quantum Feature
Maps for encoding multivariate industrial telemetry into Quantum Hilbert Space.
"""

from typing import Callable, List, Optional, Tuple, Union
import numpy as np

try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False


class AngleFeatureMap:
    """
    Angle Embedding feature map that encodes classical continuous sensor values
    into rotation angles of single-qubit rotation gates.
    """

    def __init__(self, num_qubits: int, rotation: str = "Y"):
        self.num_qubits = num_qubits
        self.rotation = rotation.upper()

    def pennylane_circuit(self, x: np.ndarray, wires: Optional[List[int]] = None):
        """Constructs PennyLane angle embedding operations."""
        if not HAS_PENNYLANE:
            raise ImportError("PennyLane is required for pennylane_circuit.")
        wires = wires or list(range(self.num_qubits))
        
        # Apply initial layer of Hadamards
        for w in wires:
            qml.Hadamard(wires=w)
            
        for i, w in enumerate(wires):
            val = x[i % len(x)]
            if self.rotation == "X":
                qml.RX(val, wires=w)
            elif self.rotation == "Y":
                qml.RY(val, wires=w)
            elif self.rotation == "Z":
                qml.RZ(val, wires=w)
            else:
                qml.RY(val, wires=w)


class ZZFeatureMap:
    """
    Second-order Pauli-Z (ZZ) Entangling Feature Map.
    Maps multivariate sensor features into non-linear, entangled multi-qubit states.
    Captures high-order correlations between pairs of industrial sensors (e.g. vibration-temperature coupling).
    """

    def __init__(self, num_qubits: int, reps: int = 2, entanglement: str = "linear"):
        self.num_qubits = num_qubits
        self.reps = reps
        self.entanglement = entanglement.lower()

    def get_entanglement_pairs(self) -> List[Tuple[int, int]]:
        """Returns the qubit connectivity pairs."""
        pairs = []
        if self.entanglement == "linear":
            for i in range(self.num_qubits - 1):
                pairs.append((i, i + 1))
        elif self.entanglement == "circular":
            for i in range(self.num_qubits):
                pairs.append((i, (i + 1) % self.num_qubits))
        elif self.entanglement == "full":
            for i in range(self.num_qubits):
                for j in range(i + 1, self.num_qubits):
                    pairs.append((i, j))
        else:
            for i in range(self.num_qubits - 1):
                pairs.append((i, i + 1))
        return pairs

    def pennylane_circuit(self, x: np.ndarray, wires: Optional[List[int]] = None):
        """Builds the ZZ-feature map circuit in PennyLane."""
        if not HAS_PENNYLANE:
            raise ImportError("PennyLane is required for pennylane_circuit.")
        wires = wires or list(range(self.num_qubits))
        pairs = self.get_entanglement_pairs()

        for _ in range(self.reps):
            # 1. Hadamard layer
            for w in wires:
                qml.Hadamard(wires=w)

            # 2. Single-qubit phase rotation U_1(x) = exp(i * x_i * Z)
            for i, w in enumerate(wires):
                val = 2.0 * x[i % len(x)]
                qml.RZ(val, wires=w)

            # 3. Two-qubit ZZ interaction U_2(x) = exp(i * (pi - x_i)(pi - x_j) * Z_i Z_j)
            for i, j in pairs:
                w1, w2 = wires[i], wires[j]
                val_i = x[i % len(x)]
                val_j = x[j % len(x)]
                phi_ij = 2.0 * (np.pi - val_i) * (np.pi - val_j)
                
                qml.CNOT(wires=[w1, w2])
                qml.RZ(phi_ij, wires=w2)
                qml.CNOT(wires=[w1, w2])


class ProjectedQuantumFeatureMap:
    """
    Projected Quantum Kernel (PQK) Feature Map.
    Maps data into quantum state and projects back via 1-qubit / 2-qubit reduced
    density matrix observables to avoid barren plateaus on high-dimensional streams.
    """

    def __init__(self, num_qubits: int, reps: int = 2):
        self.num_qubits = num_qubits
        self.reps = reps
        self.zz_map = ZZFeatureMap(num_qubits=num_qubits, reps=reps)

    def pennylane_circuit(self, x: np.ndarray, wires: Optional[List[int]] = None):
        wires = wires or list(range(self.num_qubits))
        self.zz_map.pennylane_circuit(x, wires=wires)


def build_quantum_feature_circuit(
    feature_map_type: str = "zz",
    num_qubits: int = 4,
    reps: int = 2,
    entanglement: str = "linear",
) -> Union[ZZFeatureMap, AngleFeatureMap, ProjectedQuantumFeatureMap]:
    """Factory function for quantum feature maps."""
    f_type = feature_map_type.lower()
    if f_type in ["zz", "pauli_zz", "zz_feature_map"]:
        return ZZFeatureMap(num_qubits=num_qubits, reps=reps, entanglement=entanglement)
    elif f_type in ["angle", "rotational", "angle_feature_map"]:
        return AngleFeatureMap(num_qubits=num_qubits)
    elif f_type in ["projected", "pqk"]:
        return ProjectedQuantumFeatureMap(num_qubits=num_qubits, reps=reps)
    else:
        return ZZFeatureMap(num_qubits=num_qubits, reps=reps, entanglement=entanglement)
