"""
Industrial Asset Telemetry Generator for Quantum Predictive Maintenance.

Simulates realistic, multi-sensor degradation dynamics across heavy industry,
refineries, ports, and power utilities. Models non-linear degradation trajectories,
operational load variations, noise, and incipient fault signatures.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA


@dataclass
class AssetConfig:
    """Configuration parameters for industrial asset degradation simulation."""
    asset_id: str
    asset_type: str  # e.g., 'refinery_compressor', 'port_gantry_crane', 'utility_turbine'
    total_cycles: int = 350
    fault_onset_cycle: int = 180
    degradation_rate: float = 0.035
    noise_std: float = 0.04
    regime_shift_frequency: int = 50
    random_seed: Optional[int] = 42


class IndustrialAssetTelemetryGenerator:
    """
    Generates multi-sensor time-series streams simulating industrial machinery
    degradation leading to functional failure.
    """

    ASSET_PRESETS = {
        "refinery_compressor": {
            "name": "Refinery Centrifugal Gas Compressor (C-401)",
            "sensors": ["vibration_rms", "vibration_kurtosis", "bearing_temperature", "lubrication_pressure", "acoustic_emission"],
            "base_values": {"vibration_rms": 1.2, "vibration_kurtosis": 3.0, "bearing_temperature": 55.0, "lubrication_pressure": 4.5, "acoustic_emission": 35.0},
            "failure_multipliers": {"vibration_rms": 4.5, "vibration_kurtosis": 3.2, "bearing_temperature": 1.8, "lubrication_pressure": 0.55, "acoustic_emission": 2.6},
            "critical_limits": {"vibration_rms": 5.0, "bearing_temperature": 95.0, "lubrication_pressure": 2.2},
        },
        "port_gantry_crane": {
            "name": "Port Container Ship-to-Shore Crane Hoist Gearbox (STS-08)",
            "sensors": ["vibration_rms", "vibration_kurtosis", "bearing_temperature", "lubrication_pressure", "acoustic_emission"],
            "base_values": {"vibration_rms": 2.0, "vibration_kurtosis": 2.8, "bearing_temperature": 48.0, "lubrication_pressure": 6.0, "acoustic_emission": 40.0},
            "failure_multipliers": {"vibration_rms": 3.8, "vibration_kurtosis": 3.5, "bearing_temperature": 1.9, "lubrication_pressure": 0.60, "acoustic_emission": 2.4},
            "critical_limits": {"vibration_rms": 7.5, "bearing_temperature": 90.0, "lubrication_pressure": 3.5},
        },
        "utility_turbine": {
            "name": "Power Utility Combined-Cycle Gas Turbine (GT-02)",
            "sensors": ["vibration_rms", "vibration_kurtosis", "bearing_temperature", "lubrication_pressure", "acoustic_emission"],
            "base_values": {"vibration_rms": 0.8, "vibration_kurtosis": 2.9, "bearing_temperature": 70.0, "lubrication_pressure": 5.2, "acoustic_emission": 30.0},
            "failure_multipliers": {"vibration_rms": 5.2, "vibration_kurtosis": 2.8, "bearing_temperature": 1.7, "lubrication_pressure": 0.65, "acoustic_emission": 3.0},
            "critical_limits": {"vibration_rms": 4.2, "bearing_temperature": 115.0, "lubrication_pressure": 3.2},
        },
        "chemical_pump": {
            "name": "Chemical Slurry Booster Pump (P-105)",
            "sensors": ["vibration_rms", "vibration_kurtosis", "bearing_temperature", "lubrication_pressure", "acoustic_emission"],
            "base_values": {"vibration_rms": 1.5, "vibration_kurtosis": 3.1, "bearing_temperature": 42.0, "lubrication_pressure": 3.8, "acoustic_emission": 38.0},
            "failure_multipliers": {"vibration_rms": 4.0, "vibration_kurtosis": 3.0, "bearing_temperature": 2.0, "lubrication_pressure": 0.50, "acoustic_emission": 2.8},
            "critical_limits": {"vibration_rms": 6.0, "bearing_temperature": 85.0, "lubrication_pressure": 1.8},
        },
    }

    def __init__(self, config: Optional[AssetConfig] = None):
        self.config = config or AssetConfig(
            asset_id="ASSET-REF-01",
            asset_type="refinery_compressor",
            total_cycles=320,
            fault_onset_cycle=160,
            degradation_rate=0.04,
            noise_std=0.035,
            random_seed=42,
        )
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

    def generate_single_run(self) -> pd.DataFrame:
        """
        Generates full lifecycle time-series telemetry for one machine run
        from healthy baseline to end-of-life (EOL).
        """
        cfg = self.config
        preset = self.ASSET_PRESETS.get(cfg.asset_type, self.ASSET_PRESETS["refinery_compressor"])
        cycles = np.arange(1, cfg.total_cycles + 1)
        n = len(cycles)

        # 1. Operational load regime shifts (simulates fluctuating production demands)
        regimes = np.sin(2 * np.pi * cycles / cfg.regime_shift_frequency) * 0.15 + 1.0

        # 2. Non-linear degradation index (Weibull/Exponential wear curve)
        deg_index = np.zeros(n)
        for i, c in enumerate(cycles):
            if c > cfg.fault_onset_cycle:
                tau = c - cfg.fault_onset_cycle
                # Exponentially accelerated degradation with Weibull shape parameter
                deg_index[i] = (1.0 - np.exp(-cfg.degradation_rate * (tau ** 1.35)))

        # Clip degradation index to [0, 1]
        deg_index = np.clip(deg_index, 0.0, 1.0)

        # 3. Remaining Useful Life (RUL)
        rul = np.maximum(0, cfg.total_cycles - cycles)

        # 4. Generate sensor streams
        data: Dict[str, np.ndarray] = {
            "cycle": cycles,
            "regime_load": regimes,
            "true_degradation_index": deg_index,
            "RUL": rul,
        }

        sensors = preset["sensors"]
        base = preset["base_values"]
        multipliers = preset["failure_multipliers"]

        for s in sensors:
            b_val = base[s]
            m_val = multipliers[s]
            
            # Baseline + regime influence + non-linear degradation progression
            if m_val >= 1.0:
                # Degrading upwards (e.g., vibration, temperature, acoustic emissions)
                stream = b_val * regimes + (b_val * (m_val - 1.0) * deg_index)
            else:
                # Degrading downwards (e.g., lubrication pressure drop)
                stream = b_val * regimes - (b_val * (1.0 - m_val) * deg_index)

            # Add stochastic sensor noise + occasional micro-spikes as degradation worsens
            noise = np.random.normal(0, cfg.noise_std * b_val, size=n)
            spikes = (np.random.rand(n) > 0.95).astype(float) * (deg_index ** 2) * (0.3 * b_val)
            
            data[s] = np.maximum(0.01, stream + noise + spikes)

        df = pd.DataFrame(data)

        # Assign discrete health status
        # 0: Healthy, 1: Degradation Warning (Incipient), 2: Critical / Impending Failure
        conditions = [
            df["true_degradation_index"] < 0.25,
            (df["true_degradation_index"] >= 0.25) & (df["true_degradation_index"] < 0.70),
            df["true_degradation_index"] >= 0.70,
        ]
        choices = ["HEALTHY", "WARNING", "CRITICAL"]
        df["health_state"] = np.select(conditions, choices, default="HEALTHY")
        df["health_label"] = np.select(conditions, [0, 1, 2], default=0)

        return df


def generate_industrial_dataset(
    num_assets: int = 6,
    asset_types: Optional[List[str]] = None,
    cycles_range: Tuple[int, int] = (250, 350),
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generates a multi-asset fleet dataset containing multiple machinery run-to-failure runs.
    """
    if asset_types is None:
        asset_types = ["refinery_compressor", "port_gantry_crane", "utility_turbine", "chemical_pump"]

    all_dfs = []
    np.random.seed(random_seed)

    for i in range(num_assets):
        atype = asset_types[i % len(asset_types)]
        total_cyc = np.random.randint(cycles_range[0], cycles_range[1])
        fault_onset = int(total_cyc * np.random.uniform(0.45, 0.65))
        deg_rate = float(np.random.uniform(0.03, 0.05))

        cfg = AssetConfig(
            asset_id=f"ASSET_{atype.upper()}_{i+1:02d}",
            asset_type=atype,
            total_cycles=total_cyc,
            fault_onset_cycle=fault_onset,
            degradation_rate=deg_rate,
            noise_std=0.035,
            random_seed=random_seed + i,
        )
        gen = IndustrialAssetTelemetryGenerator(cfg)
        df_run = gen.generate_single_run()
        df_run["asset_id"] = cfg.asset_id
        df_run["asset_type"] = cfg.asset_type
        all_dfs.append(df_run)

    return pd.concat(all_dfs, ignore_index=True)


def prepare_quantum_timeseries_dataset(
    df: pd.DataFrame,
    sensor_cols: Optional[List[str]] = None,
    window_size: int = 5,
    stride: int = 2,
    num_qubits: int = 4,
    scale_range: Tuple[float, float] = (0.0, np.pi),
    pca_reduce: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Prepares sliding time-series windows and normalizes them into Hilbert-space
    angle feature ranges for Quantum Machine Learning.

    Parameters:
    -----------
    df: Telemetry DataFrame.
    sensor_cols: List of sensor columns to extract.
    window_size: Number of past time cycles per window.
    stride: Step size between consecutive windows.
    num_qubits: Dimensionality matching the target quantum circuit qubits.
    scale_range: (min, max) range for quantum angle encoding, usually [0, pi] or [-pi, pi].
    pca_reduce: Whether to apply PCA dimensionality reduction to match num_qubits.

    Returns:
    --------
    X: Feature matrix of shape (num_samples, num_qubits) scaled for quantum rotation.
    y_rul: Target Remaining Useful Life values.
    y_state: Target degradation state labels (0, 1, 2).
    metadata: Dict containing scalers, PCA object, and column lists.
    """
    if sensor_cols is None:
        sensor_cols = ["vibration_rms", "vibration_kurtosis", "bearing_temperature", "lubrication_pressure", "acoustic_emission"]

    available_cols = [c for c in sensor_cols if c in df.columns]
    
    samples_X = []
    targets_rul = []
    targets_state = []

    # Process per asset if multi-asset dataset
    asset_groups = df.groupby("asset_id") if "asset_id" in df.columns else [(None, df)]

    for _, group in asset_groups:
        sensor_matrix = group[available_cols].values
        rul_vals = group["RUL"].values
        state_vals = group["health_label"].values
        n_points = len(group)

        for start_idx in range(0, n_points - window_size + 1, stride):
            end_idx = start_idx + window_size
            window = sensor_matrix[start_idx:end_idx]  # Shape: (window_size, num_sensors)
            
            # Extract summary statistics per sensor across window (mean, std, min, max, trend)
            means = np.mean(window, axis=0)
            stds = np.std(window, axis=0)
            p2p = np.ptp(window, axis=0)
            slopes = (window[-1] - window[0]) / float(window_size)
            
            feat = np.concatenate([means, stds, p2p, slopes])
            samples_X.append(feat)
            targets_rul.append(rul_vals[end_idx - 1])
            targets_state.append(state_vals[end_idx - 1])

    X_raw = np.array(samples_X)
    y_rul = np.array(targets_rul, dtype=float)
    y_state = np.array(targets_state, dtype=int)

    # Dimensionality reduction to match Quantum Circuit Qubits
    pca = None
    if pca_reduce and X_raw.shape[1] > num_qubits:
        pca = PCA(n_components=num_qubits, random_state=42)
        X_reduced = pca.fit_transform(X_raw)
    else:
        X_reduced = X_raw[:, :num_qubits]

    # MinMax Scale strictly to quantum rotation parameter range (e.g. [0, pi])
    scaler = MinMaxScaler(feature_range=scale_range)
    X_quantum = scaler.fit_transform(X_reduced)

    metadata = {
        "scaler": scaler,
        "pca": pca,
        "sensor_cols": available_cols,
        "window_size": window_size,
        "num_qubits": num_qubits,
        "scale_range": scale_range,
    }

    return X_quantum, y_rul, y_state, metadata
