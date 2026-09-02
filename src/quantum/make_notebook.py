"""
Script to generate Run_In_The_Quantum_Computer.ipynb and sync notebooks.
"""

import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# ⚡ Run_In_The_Quantum_Computer: Industrial Predictive Maintenance\n",
                "### Multi-Sensor State Telemetry Sampling on IBM Quantum & Qiskit Primitives V2\n",
                "\n",
                "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/25A31A0333/predictive-maintenance/blob/main/Run_In_The_Quantum_Computer.ipynb)\n",
                "\n",
                "This notebook implements an end-to-end quantum computing pipeline for **Industrial Predictive Maintenance**:\n",
                "1. **Multi-Sensor Telemetry Ingestion**: Simulates turbomachinery vibration, temperature, pressure, and acoustic signals.\n",
                "2. **Quantum Hilbert Space Encoding**: Parameterized $ZZ$-Entangling feature map on 4 qubits.\n",
                "3. **Qiskit Circuit Construction (`qc_hw`)**: Named classical registers and measurement gates.\n",
                "4. **Local Statevector Simulation**: Qiskit 1.x / 2.x `StatevectorSampler` with robust `DataBin` result parsing.\n",
                "5. **Measurement Bitstring Histogram**: Frequency distribution across 2048 shots.\n",
                "6. **Real IBM Quantum Hardware Execution**: Qiskit Runtime V2 on IBM Quantum QPUs.\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🛠️ Step 1: Environment & Qiskit Setup\n",
                "Verify installed Qiskit and Qiskit Runtime versions."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import sys\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "import qiskit\n",
                "from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister\n",
                "from qiskit.primitives import StatevectorSampler\n",
                "from qiskit.visualization import plot_histogram\n",
                "\n",
                "try:\n",
                "    import qiskit_ibm_runtime\n",
                "    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as IBMRuntimeSampler\n",
                "    HAS_IBM_RUNTIME = True\n",
                "    ibm_ver = qiskit_ibm_runtime.__version__\n",
                "except ImportError:\n",
                "    HAS_IBM_RUNTIME = False\n",
                "    ibm_ver = 'Not Installed'\n",
                "\n",
                "print(f'[OK] Qiskit Core Version:         {qiskit.__version__}')\n",
                "print(f'[OK] Qiskit IBM Runtime Version: {ibm_ver}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🏭 Step 2: Multi-Sensor Telemetry & Quantum Feature Normalization\n",
                "Simulate 4 continuous sensor streams representing industrial equipment health."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Multi-sensor telemetry snapshot: [Vibration RMS, Kurtosis, Bearing Temp, Pressure]\n",
                "np.random.seed(42)\n",
                "sensor_raw = np.array([3.85, 4.20, 78.5, 3.20]) # Incipient wear state\n",
                "\n",
                "# Normalize sensor channels into quantum rotation angles in [0, pi]\n",
                "sensor_mins = np.array([1.0, 2.5, 40.0, 2.0])\n",
                "sensor_maxs = np.array([6.0, 8.0, 100.0, 5.0])\n",
                "sensor_angles = np.pi * (sensor_raw - sensor_mins) / (sensor_maxs - sensor_mins)\n",
                "sensor_angles = np.clip(sensor_angles, 0.0, np.pi)\n",
                "\n",
                "print('--- Industrial Sensor Snapshot ---')\n",
                "print(f'Raw Telemetry:        {sensor_raw}')\n",
                "print(f'Normalized Angles:    {[round(float(a), 3) for a in sensor_angles]} rad')\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## ⚛️ Step 3: Constructing Quantum Circuit (`qc_hw`)\n",
                "Create parameterized $ZZ$-Entangling feature map with explicit `ClassicalRegister`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "num_qubits = 4\n",
                "qr = QuantumRegister(num_qubits, name='sensor_q')\n",
                "cr = ClassicalRegister(num_qubits, name='sensor_meas')\n",
                "qc_hw = QuantumCircuit(qr, cr, name='MultiSensor_Quantum_Circuit')\n",
                "\n",
                "reps = 2\n",
                "for _ in range(reps):\n",
                "    # Layer 1: Hadamard Superposition\n",
                "    for i in range(num_qubits):\n",
                "        qc_hw.h(qr[i])\n",
                "        \n",
                "    # Layer 2: Single-Qubit Sensor Rotations RZ(2 * x_i)\n",
                "    for i in range(num_qubits):\n",
                "        qc_hw.rz(2.0 * float(sensor_angles[i]), qr[i])\n",
                "        \n",
                "    # Layer 3: Two-Qubit Entangling Interaction (Cross-Sensor Correlation)\n",
                "    for i in range(num_qubits - 1):\n",
                "        phi_ij = 2.0 * (np.pi - float(sensor_angles[i])) * (np.pi - float(sensor_angles[i+1]))\n",
                "        qc_hw.cx(qr[i], qr[i+1])\n",
                "        qc_hw.rz(phi_ij, qr[i+1])\n",
                "        qc_hw.cx(qr[i], qr[i+1])\n",
                "\n",
                "qc_hw.barrier()\n",
                "qc_hw.measure(qr, cr)\n",
                "\n",
                "print('--- Quantum Circuit Specifications ---')\n",
                "print(f'Qubits:              {qc_hw.num_qubits}')\n",
                "print(f'Classical Bits:      {qc_hw.num_clbits}')\n",
                "print(f'Classical Registers: {[r.name for r in qc_hw.cregs]}')\n",
                "print(f'Operation Count:     {dict(qc_hw.count_ops())}')\n",
                "\n",
                "# Draw Circuit representation\n",
                "try:\n",
                "    print(qc_hw.draw(output='text'))\n",
                "except UnicodeEncodeError:\n",
                "    print(str(qc_hw.draw(output='text')).encode('ascii', errors='replace').decode('ascii'))\n",
                "except Exception as e:\n",
                "    print(f'Circuit: {qc_hw.name} ({qc_hw.num_qubits} qubits)')\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🚀 Step 4: Robust Qiskit Primitives V2 Local Simulation\n",
                "Run circuit using `StatevectorSampler` and robustly extract measurement counts from `DataBin`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def extract_counts_robust(pub_result, circuit=None):\n",
                "    \"\"\"\n",
                "    Dynamically extracts bitstring counts from Qiskit SamplerPubResult / DataBin,\n",
                "    inspecting actual classical registers rather than assuming hardcoded names.\n",
                "    \"\"\"\n",
                "    data = pub_result.data\n",
                "    \n",
                "    # 1. Check known ClassicalRegister names from circuit\n",
                "    if circuit is not None and hasattr(circuit, 'cregs') and len(circuit.cregs) > 0:\n",
                "        for creg in circuit.cregs:\n",
                "            if hasattr(data, creg.name):\n",
                "                bit_array = getattr(data, creg.name)\n",
                "                if hasattr(bit_array, 'get_counts'):\n",
                "                    return bit_array.get_counts()\n",
                "                    \n",
                "    # 2. Check dictionary keys if supported by DataBin\n",
                "    if hasattr(data, 'keys') and callable(getattr(data, 'keys')):\n",
                "        for k in data.keys():\n",
                "            item = data[k]\n",
                "            if hasattr(item, 'get_counts'):\n",
                "                return item.get_counts()\n",
                "                \n",
                "    # 3. Dynamic attribute inspection for BitArray\n",
                "    for attr_name in dir(data):\n",
                "        if not attr_name.startswith('_'):\n",
                "            val = getattr(data, attr_name)\n",
                "            if hasattr(val, 'get_counts'):\n",
                "                return val.get_counts()\n",
                "                \n",
                "    raise AttributeError(f'Could not find BitArray in DataBin: {dir(data)}')\n",
                "\n",
                "# Execute 2048 shots via StatevectorSampler\n",
                "sampler = StatevectorSampler()\n",
                "job = sampler.run([qc_hw], shots=2048)\n",
                "result = job.result()\n",
                "\n",
                "# Robustly obtain counts from the PubResult DataBin\n",
                "counts = extract_counts_robust(result[0], circuit=qc_hw)\n",
                "\n",
                "print('Quantum State Measurement Bitstring Frequencies (2048 shots):')\n",
                "print(f'Total Counted Shots: {sum(counts.values())}')\n",
                "for bs, freq in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:8]:\n",
                "    print(f'  |{bs}> : {freq:4d} shots ({freq/2048*100:5.2f}%)')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 📊 Step 5: Multi-Sensor Quantum State Histogram Visualization"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(10, 4))\n",
                "plot_histogram(counts, title='IBM Quantum Bitstring Histogram – Multi-Sensor State')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🌐 Step 6: IBM Quantum Real Hardware Execution\n",
                "Connect to IBM Quantum QPUs using `QiskitRuntimeService` and `SamplerV2`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "\n",
                "def run_on_ibm_qpu(circuit, shots=2048):\n",
                "    if not HAS_IBM_RUNTIME:\n",
                "        print('[WARN] qiskit-ibm-runtime is not installed. Run: pip install qiskit-ibm-runtime')\n",
                "        return None\n",
                "        \n",
                "    token = os.environ.get('IBM_QUANTUM_TOKEN', None)\n",
                "    try:\n",
                "        if token:\n",
                "            service = QiskitRuntimeService(channel='ibm_quantum_platform', token=token)\n",
                "        else:\n",
                "            service = QiskitRuntimeService(channel='ibm_quantum_platform')\n",
                "            \n",
                "        backend = service.least_busy(operational=True, simulator=False)\n",
                "        print(f'[*] Connected to IBM Quantum Hardware Backend: {backend.name} ({backend.num_qubits} qubits)')\n",
                "        \n",
                "        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager\n",
                "        pm = generate_preset_pass_manager(optimization_level=1, backend=backend)\n",
                "        transpiled = pm.run(circuit)\n",
                "        \n",
                "        sampler = IBMRuntimeSampler(mode=backend)\n",
                "        job = sampler.run([transpiled], shots=shots)\n",
                "        print(f'[*] Submitted job ID: {job.job_id()}')\n",
                "        hw_result = job.result()\n",
                "        hw_counts = extract_counts_robust(hw_result[0], circuit=transpiled)\n",
                "        return hw_counts\n",
                "        \n",
                "    except Exception as e:\n",
                "        print('\\n=======================================================')\n",
                "        print(' [IBM QUANTUM HARDWARE CREDENTIALS REQUIRED]')\n",
                "        print('=======================================================')\n",
                "        print(f'Connection status: {e}')\n",
                "        print('To execute on real IBM QPUs:')\n",
                "        print('1. Obtain API token at https://quantum.ibm.com/')\n",
                "        print('2. Save token: python src/quantum/ibm_quantum_setup.py --token <YOUR_TOKEN>')\n",
                "        print('\\n[INFO] Local simulation completed successfully.')\n",
                "        return None\n",
                "\n",
                "# Attempt real hardware execution\n",
                "hw_counts = run_on_ibm_qpu(qc_hw, shots=2048)"
            ]
        }
    ],
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.11"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

if __name__ == "__main__":
    targets = [
        "Run_In_The_Quantum_Computer.ipynb",
        "notebooks/Run_In_The_Quantum_Computer.ipynb"
    ]
    for target in targets:
        with open(target, "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=2)
        print(f"Generated: {target}")
