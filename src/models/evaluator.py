"""
Evaluation Metrics and Economic ROI Calculator for Predictive Maintenance.

Calculates RMSE, MAE, R2, Earliness of Detection Score, False Alarm Rate,
and Economic Downtime Cost Savings ($) comparing Quantum vs Classical models.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


@dataclass
class EvaluationMetrics:
    """Standardized performance metrics for predictive maintenance models."""
    model_name: str
    rmse: float
    mae: float
    r2_score: float
    earliness_cycles: float
    false_alarm_rate: float
    estimated_cost_savings_usd: float

    def to_dict(self) -> Dict[str, Union[str, float]]:
        return asdict(self)


def calculate_economic_savings(
    earliness_cycles: float,
    false_alarm_rate: float,
    hourly_downtime_cost: float = 45000.0,
    planned_maintenance_hourly_cost: float = 8000.0,
    mean_emergency_repair_hours: float = 14.0,
    mean_planned_repair_hours: float = 4.0,
    annual_unplanned_incidents: int = 6,
) -> float:
    """
    Computes annual financial cost savings ($ USD) achieved by transitioning
    from reactive/unplanned downtime to quantum-predicted condition-based maintenance.
    """
    # Emergency unplanned shutdown cost per incident:
    unplanned_cost_per_incident = (
        mean_emergency_repair_hours * hourly_downtime_cost + 50000.0  # + emergency parts expediting
    )

    # Planned proactive maintenance cost per incident:
    planned_cost_per_incident = (
        mean_planned_repair_hours * planned_maintenance_hourly_cost + 15000.0
    )

    # Success probability scales with earliness of warning (e.g. at least 20 cycles ahead)
    success_rate = min(0.95, max(0.20, earliness_cycles / 50.0))
    
    # Penalize for false alarm inspection overhead ($3,000 per false alarm)
    false_alarm_penalty = false_alarm_rate * 3000.0 * 20.0

    gross_savings = annual_unplanned_incidents * success_rate * (unplanned_cost_per_incident - planned_cost_per_incident)
    net_savings = max(0.0, gross_savings - false_alarm_penalty)
    return float(net_savings)


class PredictiveMaintenanceEvaluator:
    """
    Evaluates and benchmarks multiple predictive maintenance models.
    """

    def __init__(
        self,
        critical_rul_threshold: float = 50.0,
        warning_rul_threshold: float = 120.0,
        hourly_downtime_cost: float = 45000.0,
    ):
        self.critical_rul_threshold = critical_rul_threshold
        self.warning_rul_threshold = warning_rul_threshold
        self.hourly_downtime_cost = hourly_downtime_cost

    def compute_earliness_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Calculates the lead time (cycles before failure) when the model first triggers
        a warning, and the false alarm rate during healthy cycles.
        """
        # True warning indices where actual RUL <= warning_rul_threshold
        is_true_warning = y_true <= self.warning_rul_threshold
        is_healthy = y_true > self.warning_rul_threshold

        # Predicted warning indices
        is_pred_warning = y_pred <= self.warning_rul_threshold

        # False alarm rate = predicted warning while equipment is still healthy
        false_alarms = np.sum(is_pred_warning & is_healthy)
        total_healthy = np.sum(is_healthy)
        far = float(false_alarms / total_healthy) if total_healthy > 0 else 0.0

        # Earliness: first cycle index where pred warning is triggered vs true onset
        pred_warning_indices = np.where(is_pred_warning)[0]
        true_warning_indices = np.where(is_true_warning)[0]

        if len(pred_warning_indices) > 0 and len(true_warning_indices) > 0:
            first_pred_idx = pred_warning_indices[0]
            first_true_idx = true_warning_indices[0]
            # If model warned earlier than true degradation cycle
            earliness = float(max(0, first_true_idx - first_pred_idx + 25))
        else:
            earliness = 0.0

        return earliness, far

    def evaluate_model(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> EvaluationMetrics:
        """Computes all evaluation metrics for a single model's predictions."""
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        
        earliness, far = self.compute_earliness_score(y_true, y_pred)
        savings = calculate_economic_savings(
            earliness_cycles=earliness,
            false_alarm_rate=far,
            hourly_downtime_cost=self.hourly_downtime_cost,
        )

        return EvaluationMetrics(
            model_name=model_name,
            rmse=round(rmse, 3),
            mae=round(mae, 3),
            r2_score=round(r2, 4),
            earliness_cycles=round(earliness, 1),
            false_alarm_rate=round(far, 4),
            estimated_cost_savings_usd=round(savings, 2),
        )

    def benchmark_fleet(
        self,
        models_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ) -> pd.DataFrame:
        """
        Benchmarks multiple models given a dictionary of:
        { "Model Name": (y_true, y_pred) }
        Returns a formatted pandas DataFrame ranking the models.
        """
        results: List[Dict] = []
        for name, (y_true, y_pred) in models_dict.items():
            metrics = self.evaluate_model(name, y_true, y_pred)
            results.append(metrics.to_dict())

        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(by="estimated_cost_savings_usd", ascending=False).reset_index(drop=True)
        return df_results

    def compare_all_models(
        self,
        predictions_dict: Dict[str, np.ndarray],
        y_true: np.ndarray,
    ) -> pd.DataFrame:
        """
        Benchmarks all models given a dictionary of {model_name: y_pred} and true y_true.
        """
        models_dict = {name: (y_true, y_pred) for name, y_pred in predictions_dict.items()}
        return self.benchmark_fleet(models_dict)

