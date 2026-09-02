"""
Run_In_The_Quantum_Computer - Multi-Sensor Predictive Maintenance Qiskit Workflow.

Compatible with Qiskit 1.x & 2.x (Qiskit Primitives V2, StatevectorSampler, DataBin, BitArray, and Qiskit Runtime).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple

import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.primitives import StatevectorSampler
from qiskit.visualization import plot_histogram

try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as IBMRuntimeSampler
    HAS_IBM_RUNTIME = True
except ImportError:
    HAS_IBM_RUNTIME = False


def extract_counts_from_pub_result(pub_result: Any, circuit: Optional[QuantumCircuit] = None) -> Dict[str, int]:
    """
    Robustly extracts bitstring measurement counts from a Qiskit SamplerPubResult / DataBin.
    
    Compatible with:
    - Named ClassicalRegister (e.g., cr = ClassicalRegister(4, 'sensor_meas'))
    - measure_all() (which creates a register named 'meas')
    - QuantumCircuit(n, n) (which creates a default register named 'c')
    - Arbitrary user-defined register names
    """
    data = pub_result.data
    
    # Strategy 1: Check known ClassicalRegister names from circuit.cregs
    if circuit is not None and hasattr(circuit, 'cregs') and len(circuit.cregs) > 0:
        for creg in circuit.cregs:
            if hasattr(data, creg.name):
                bit_array = getattr(data, creg.name)
                if hasattr(bit_array, 'get_counts'):
                    return bit_array.get_counts()
                    
    # Strategy 2: Check keys if DataBin implements dict-like keys()
    if hasattr(data, 'keys') and callable(getattr(data, 'keys')):
        for k in data.keys():
            item = data[k]
            if hasattr(item, 'get_counts'):
                return item.get_counts()
                
    # Strategy 3: Dynamic attribute inspection for any BitArray with get_counts
    for attr_name in dir(data):
        if not attr_name.startswith('_'):
            val = getattr(data, attr_name)
            if hasattr(val, 'get_counts'):
                return val.get_counts()
                
    raise AttributeError(
        f"Unable to extract measurement counts from DataBin. Available fields: "
        f"{list(data.keys()) if hasattr(data, 'keys') else [a for a in dir(data) if not a.startswith('_')]}"
    )


def create_sensor_quantum_circuit(
    sensor_features: np.ndarray,
    num_qubits: int = 4,
    reps: int = 2,
    register_name: str = "sensor_meas"
) -> QuantumCircuit:
    """
    Constructs a Parameterized ZZ-Entangling Quantum Circuit for multi-sensor state encoding.
    
    Args:
        sensor_features: Array of normalized sensor angles in [0, pi], length >= num_qubits.
        num_qubits: Number of qubits (sensor channels).
        reps: Number of entangling layers.
        register_name: Name for the ClassicalRegister.
        
    Returns:
        Measured QuantumCircuit ready for Qiskit Sampler execution.
    """
    qr = QuantumRegister(num_qubits, name="sensor_q")
    cr = ClassicalRegister(num_qubits, name=register_name)
    qc = QuantumCircuit(qr, cr, name="MultiSensor_Quantum_PredictiveMaintenance")
    
    # Ensure sensor features match qubit count
    x = np.asarray(sensor_features, dtype=float)
    if len(x) < num_qubits:
        x = np.pad(x, (0, num_qubits - len(x)), mode="constant")
    else:
        x = x[:num_qubits]
        
    # Build ZZ-Entangling Feature Map
    for _ in range(reps):
        # 1. Hadamard Superposition
        for i in range(num_qubits):
            qc.h(qr[i])
            
        # 2. Single-Qubit Sensor Phase Rotations RZ(2 * x_i)
        for i in range(num_qubits):
            qc.rz(2.0 * float(x[i]), qr[i])
            
        # 3. Two-Qubit Entangling Interaction ZZ: CNOT -> RZ(2*(pi - x_i)*(pi - x_j)) -> CNOT
        for i in range(num_qubits - 1):
            q1, q2 = qr[i], qr[i + 1]
            phi_ij = 2.0 * (np.pi - float(x[i])) * (np.pi - float(x[i + 1]))
            qc.cx(q1, q2)
            qc.rz(phi_ij, q2)
            qc.cx(q1, q2)
            
    # Measurement onto classical register
    qc.barrier()
    qc.measure(qr, cr)
    
    return qc


def run_local_simulation(qc_hw: QuantumCircuit, shots: int = 2048) -> Dict[str, int]:
    """
    Executes the multi-sensor quantum circuit on the local StatevectorSampler (Qiskit 1.x/2.x).
    """
    print(f"[*] Running Local Quantum Simulation (StatevectorSampler, shots={shots})...")
    
    # 1. Verify measurements exist in qc_hw
    ops = qc_hw.count_ops()
    if "measure" not in ops or ops["measure"] == 0:
        raise ValueError("qc_hw does not contain any measurement gates. Cannot sample bitstrings.")
        
    # 2. Run with modern Qiskit Primitives StatevectorSampler
    sampler = StatevectorSampler()
    job = sampler.run([qc_hw], shots=shots)
    result = job.result()
    pub_result = result[0]
    
    # 3. Robustly extract counts
    counts = extract_counts_from_pub_result(pub_result, circuit=qc_hw)
    
    total_shots = sum(counts.values())
    print(f"[SUCCESS] Execution completed. Total measured shots: {total_shots} / {shots}")
    return counts


def run_on_ibm_hardware(
    qc_hw: QuantumCircuit,
    shots: int = 2048,
    backend_name: Optional[str] = None,
    channel: str = "ibm_quantum_platform"
) -> Optional[Dict[str, int]]:
    """
    Executes the quantum circuit on an IBM Quantum Hardware QPU (or Cloud simulator) using Qiskit Runtime V2.
    """
    if not HAS_IBM_RUNTIME:
        print("[WARN] 'qiskit-ibm-runtime' is not installed. Please install it using:")
        print("       pip install qiskit-ibm-runtime")
        return None
        
    token = os.environ.get("IBM_QUANTUM_TOKEN", None)
    
    try:
        if token:
            service = QiskitRuntimeService(channel=channel, token=token)
        else:
            service = QiskitRuntimeService(channel=channel)
    except Exception as e:
        print("\n=======================================================")
        print(" [IBM QUANTUM HARDWARE SETUP REQUIRED]")
        print("=======================================================")
        print(f"Could not connect to IBM Quantum: {e}")
        print("To run on a real IBM Quantum computer:")
        print("1. Get your free token from https://quantum.ibm.com/")
        print("2. Save it by running: python src/quantum/ibm_quantum_setup.py --token <YOUR_TOKEN>")
        print("3. Or set the environment variable: export IBM_QUANTUM_TOKEN='<YOUR_TOKEN>'\n")
        return None
        
    try:
        # Select target backend
        if backend_name:
            backend = service.backend(backend_name)
        else:
            backend = service.least_busy(operational=True, simulator=False)
            
        print(f"[*] Target IBM Quantum Hardware Backend: {backend.name} ({backend.num_qubits} qubits)")
        
        # Transpile circuit for backend target basis gates and coupling map
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
        transpiled_circuit = pm.run(qc_hw)
        
        # Submit job to IBM Quantum SamplerV2
        print(f"[*] Submitting job to IBM Quantum Hardware (shots={shots})...")
        sampler = IBMRuntimeSampler(mode=backend)
        job = sampler.run([transpiled_circuit], shots=shots)
        print(f"[*] Job ID: {job.job_id()}. Waiting for execution on QPU queue...")
        result = job.result()
        pub_result = result[0]
        
        counts = extract_counts_from_pub_result(pub_result, circuit=transpiled_circuit)
        print(f"[SUCCESS] Real Hardware Job Complete! Shots: {sum(counts.values())}")
        return counts
        
    except Exception as e:
        print(f"[ERROR] IBM Quantum Hardware execution failed: {e}")
        return None


if __name__ == "__main__":
    print(f"=== Qiskit Predictive Maintenance: Multi-Sensor Quantum State Sampling ===")
    print(f"Qiskit Version: {qiskit.__version__}")
    
    # 1. Sample multi-sensor values: [Vibration, Kurtosis, Temperature, Pressure]
    sensor_telemetry_angles = np.array([0.45 * np.pi, 0.72 * np.pi, 0.28 * np.pi, 0.88 * np.pi])
    print(f"Sensor normalized input angles: {sensor_telemetry_angles}")
    
    # 2. Build Quantum Circuit
    qc_hw = create_sensor_quantum_circuit(
        sensor_features=sensor_telemetry_angles,
        num_qubits=4,
        reps=2,
        register_name="sensor_meas"
    )
    
    print("\n--- Quantum Circuit Summary ---")
    print(f"Qubits: {qc_hw.num_qubits}, Classical Bits: {qc_hw.num_clbits}")
    print(f"Classical Registers: {[r.name for r in qc_hw.cregs]}")
    print(f"Operations: {dict(qc_hw.count_ops())}")
    
    # 3. Local Simulation (StatevectorSampler)
    counts = run_local_simulation(qc_hw, shots=2048)
    print("\nQuantum State Measurement Bitstring Frequencies (2048 shots):")
    for bs, freq in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:8]:
        print(f"  |{bs}> : {freq} shots ({freq/2048*100:.2f}%)")
        
    # 4. Attempt IBM Hardware Execution (if configured)
    print("\n--- IBM Quantum Hardware Execution Check ---")
    hw_counts = run_on_ibm_hardware(qc_hw, shots=2048)
    if hw_counts is not None:
        print("Hardware Counts Sample:", list(hw_counts.items())[:4])
    else:
        print("[INFO] Local simulation completed successfully. IBM hardware available upon token registration.")
