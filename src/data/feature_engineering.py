"""
Advanced Sensor Telemetry Feature Engineering for Industrial Predictive Maintenance.

Extracts:
1. Rolling temporal window statistics (mean, std, skew, kurtosis, min, max, peak-to-peak, EWMA).
2. Rate-of-change dynamics (velocities, accelerations, relative gradient).
3. Cross-sensor covariance & correlation indicators.
4. Composite Equipment Health Indicators (HI).
5. Robust sensor scaling & missing-value handling.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.decomposition import PCA


class IndustrialFeatureEngineer:
    """
    Computes time-series statistical, dynamic, and cross-sensor features from raw industrial telemetry.
    """

    def __init__(
        self,
        sensor_cols: Optional[List[str]] = None,
        window_size: int = 5,
        ewma_span: int = 7,
        handle_missing: bool = True,
    ):
        self.sensor_cols = sensor_cols or [
            "vibration_rms",
            "vibration_kurtosis",
            "bearing_temperature",
            "lubrication_pressure",
            "acoustic_emission",
        ]
        self.window_size = window_size
        self.ewma_span = ewma_span
        self.handle_missing = handle_missing
        self.baseline_stats: Dict[str, Dict[str, float]] = {}
        self.scaler: Optional[MinMaxScaler] = None
        self.pca: Optional[PCA] = None

    def clean_telemetry(self, df: pd.DataFrame) -> pd.DataFrame:
        """Imputes missing values and removes non-physical negative values for sensor readings."""
        df_clean = df.copy()
        for col in self.sensor_cols:
            if col in df_clean.columns:
                if self.handle_missing and df_clean[col].isnull().any():
                    # Forward-fill then backward-fill, fallback to column median
                    df_clean[col] = df_clean[col].ffill().bfill().fillna(df_clean[col].median() if not df_clean[col].isnull().all() else 0.0)
                
                # Prevent non-physical negative pressures/vibrations
                if "pressure" in col or "vibration" in col or "temperature" in col:
                    df_clean[col] = np.maximum(0.001, df_clean[col])
        return df_clean

    def extract_time_series_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes rolling statistical and dynamic trend features for each sensor stream.
        """
        df_clean = self.clean_telemetry(df)
        feat_df = pd.DataFrame(index=df_clean.index)
        
        # Copy metadata if present
        for meta_col in ["cycle", "asset_id", "asset_type", "RUL", "true_degradation_index", "health_state"]:
            if meta_col in df_clean.columns:
                feat_df[meta_col] = df_clean[meta_col]

        # 1. Base values & Rolling statistics
        for col in self.sensor_cols:
            if col not in df_clean.columns:
                continue
            series = df_clean[col]
            feat_df[f"{col}_raw"] = series
            
            # Rolling window statistics
            feat_df[f"{col}_roll_mean"] = series.rolling(window=self.window_size, min_periods=1).mean()
            feat_df[f"{col}_roll_std"] = series.rolling(window=self.window_size, min_periods=1).std().fillna(0.0)
            feat_df[f"{col}_roll_min"] = series.rolling(window=self.window_size, min_periods=1).min()
            feat_df[f"{col}_roll_max"] = series.rolling(window=self.window_size, min_periods=1).max()
            feat_df[f"{col}_roll_ptp"] = feat_df[f"{col}_roll_max"] - feat_df[f"{col}_roll_min"]
            
            # Exponentially Weighted Moving Average (EWMA)
            feat_df[f"{col}_ewma"] = series.ewm(span=self.ewma_span, adjust=False).mean()
            
            # Rate of change: Velocity (1st diff) and Acceleration (2nd diff)
            feat_df[f"{col}_velocity"] = series.diff().fillna(0.0)
            feat_df[f"{col}_acceleration"] = feat_df[f"{col}_velocity"].diff().fillna(0.0)
            
            # Relative change to rolling mean
            roll_m = np.maximum(1e-5, feat_df[f"{col}_roll_mean"])
            feat_df[f"{col}_rel_change"] = (series - roll_m) / roll_m

        # 2. Cross-Sensor Degradation Indicators
        if "vibration_rms" in df_clean.columns and "bearing_temperature" in df_clean.columns:
            feat_df["vib_temp_coupling"] = df_clean["vibration_rms"] * df_clean["bearing_temperature"]
            
        if "lubrication_pressure" in df_clean.columns and "bearing_temperature" in df_clean.columns:
            press_safe = np.maximum(0.1, df_clean["lubrication_pressure"])
            feat_df["temp_pressure_ratio"] = df_clean["bearing_temperature"] / press_safe

        if "vibration_rms" in df_clean.columns and "vibration_kurtosis" in df_clean.columns:
            feat_df["vibration_impulsiveness_product"] = df_clean["vibration_rms"] * df_clean["vibration_kurtosis"]

        # 3. Composite Health Indicator (HI) [1.0 = healthy, 0.0 = critical]
        feat_df["composite_health_indicator"] = self.compute_composite_health_indicator(df_clean)

        return feat_df

    def compute_composite_health_indicator(self, df: pd.DataFrame) -> pd.Series:
        """
        Computes a normalized composite machinery health index between 1.0 (healthy) and 0.0 (failed)
        based on multi-sensor deviations from healthy operating baselines.
        """
        df_clean = self.clean_telemetry(df)
        available_sensors = [c for c in self.sensor_cols if c in df_clean.columns]
        if not available_sensors:
            return pd.Series(1.0, index=df_clean.index)

        # Establish healthy baseline from first 15% of cycles if not already calibrated
        norm_deviations = []
        for col in available_sensors:
            series = df_clean[col].values
            base_slice = series[:max(5, int(len(series) * 0.15))]
            base_mean = np.mean(base_slice)
            base_std = np.maximum(1e-4, np.std(base_slice))
            
            # Z-score deviation from healthy baseline
            z_dev = (series - base_mean) / base_std
            
            # For lubrication pressure, a drop is bad; for vibration/temp, a rise is bad
            if "pressure" in col:
                penalty = np.maximum(0.0, -z_dev)
            else:
                penalty = np.maximum(0.0, z_dev)
            norm_deviations.append(penalty)

        avg_deviation = np.mean(norm_deviations, axis=0)
        # Map deviation to [0, 1] Health Indicator via exponential decay
        health_index = np.exp(-0.25 * avg_deviation)
        return pd.Series(np.clip(health_index, 0.0, 1.0), index=df_clean.index)

    def prepare_quantum_features(
        self,
        df: pd.DataFrame,
        num_qubits: int = 4,
        scale_range: Tuple[float, float] = (0.0, np.pi),
        fit: bool = True,
    ) -> np.ndarray:
        """
        Extracts multi-sensor features and maps them into an n-dimensional quantum rotation angle space.
        """
        feat_df = self.extract_time_series_features(df)
        num_cols = [c for c in feat_df.columns if c not in ["cycle", "asset_id", "asset_type", "RUL", "true_degradation_index", "health_state"]]
        X_mat = feat_df[num_cols].values
        
        # PCA Dimensionality Reduction to target qubit space
        if fit or self.pca is None:
            self.pca = PCA(n_components=min(num_qubits, X_mat.shape[1]), random_state=42)
            X_red = self.pca.fit_transform(X_mat)
        else:
            X_red = self.pca.transform(X_mat)

        # Scale into Quantum Rotation Space [0, pi]
        if fit or self.scaler is None:
            self.scaler = MinMaxScaler(feature_range=scale_range)
            X_quant = self.scaler.fit_transform(X_red)
        else:
            X_quant = self.scaler.transform(X_red)

        # Pad with zeros if X_quant has fewer dimensions than num_qubits
        if X_quant.shape[1] < num_qubits:
            pad_width = ((0, 0), (0, num_qubits - X_quant.shape[1]))
            X_quant = np.pad(X_quant, pad_width, mode="constant")

        return np.clip(X_quant, scale_range[0], scale_range[1])
