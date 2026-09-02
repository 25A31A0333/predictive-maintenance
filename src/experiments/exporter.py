"""
Data and Diagnostic Report Exporter for Predictive Maintenance.

Exports:
1. predictions.csv
2. metrics.csv
3. anomaly_results.csv
4. maintenance_report.csv
"""

import os
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd


class DataExporter:
    """Exports structured predictive maintenance datasets, forecasts, and reports."""

    def __init__(self, export_dir: str = "results"):
        self.export_dir = export_dir
        os.makedirs(self.export_dir, exist_ok=True)

    def export_predictions(
        self,
        cycles: np.ndarray,
        y_true_rul: np.ndarray,
        predictions: Dict[str, np.ndarray],
        filename: str = "predictions.csv",
    ) -> str:
        """Exports cycle-by-cycle model predictions to CSV."""
        df = pd.DataFrame({"cycle": cycles, "actual_RUL": y_true_rul})
        for model_name, preds in predictions.items():
            df[f"pred_{model_name}"] = preds
            
        out_path = os.path.join(self.export_dir, filename)
        df.to_csv(out_path, index=False)
        return out_path

    def export_metrics(
        self,
        benchmark_df: pd.DataFrame,
        filename: str = "metrics.csv",
    ) -> str:
        """Exports model performance benchmark metrics to CSV."""
        out_path = os.path.join(self.export_dir, filename)
        benchmark_df.to_csv(out_path, index=False)
        return out_path

    def export_anomaly_results(
        self,
        cycles: np.ndarray,
        anomaly_scores: np.ndarray,
        threshold: float,
        is_anomaly: np.ndarray,
        severity_levels: List[str],
        filename: str = "anomaly_results.csv",
    ) -> str:
        """Exports cycle-by-cycle anomaly scores, triggers, and severity states to CSV."""
        df = pd.DataFrame({
            "cycle": cycles,
            "anomaly_score": anomaly_scores,
            "threshold": threshold,
            "is_anomaly": is_anomaly,
            "severity_level": severity_levels,
        })
        out_path = os.path.join(self.export_dir, filename)
        df.to_csv(out_path, index=False)
        return out_path

    def export_maintenance_report(
        self,
        report_records: List[Dict[str, Any]],
        filename: str = "maintenance_report.csv",
    ) -> str:
        """Exports structured maintenance recommendations and work orders to CSV."""
        df = pd.DataFrame(report_records)
        out_path = os.path.join(self.export_dir, filename)
        df.to_csv(out_path, index=False)
        return out_path
