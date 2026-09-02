"""
Rule-Based Maintenance Advisory and Decision Support Module.

Translates machine health state, anomaly scores, RUL predictions, and top-contributing
sensor channels into actionable industrial maintenance recommendations and work orders.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Union, Any


@dataclass
class MaintenanceRecommendation:
    """Represents a structured decision-support maintenance work order."""
    asset_id: str
    asset_type: str
    risk_level: str                         # 'HIGH RISK', 'MODERATE RISK', 'LOW RISK / NORMAL'
    urgency: str                            # 'IMMEDIATE_ACTION', 'PLANNED_MAINTENANCE', 'ROUTINE_MONITORING'
    predicted_rul_cycles: float
    estimated_failure_window: str
    likely_issue: str
    recommended_action: str
    inspection_checklist: List[str]
    top_affected_sensors: List[Tuple[str, float]]
    disclaimer: str


class MaintenanceAdvisor:
    """
    Generates deterministic, domain-expert maintenance recommendations for industrial assets.
    """

    FAILURE_MODES = {
        "bearing_temperature": {
            "issue": "Bearing overheating / thermal fatigue / lubrication breakdown",
            "action": "Inspect bearing housing, verify lubrication flow rate and oil viscosity, check for mechanical friction.",
            "checklist": ["Measure lubricant temperature and contamination level", "Check bearing clearance with feeler gauge", "Inspect for thermal discoloration or spalling"],
        },
        "vibration_rms": {
            "issue": "Mechanical unbalance / shaft misalignment / structural looseness",
            "action": "Perform dynamic laser shaft alignment, check foundation anchor bolts, inspect impeller/rotor balance.",
            "checklist": ["Execute phase vibration analysis", "Verify torque on foundation hold-down bolts", "Inspect flexible coupling elements"],
        },
        "vibration_kurtosis": {
            "issue": "Incipient bearing raceway spalling / gear tooth impact pitting",
            "action": "Perform high-frequency demodulation (envelope analysis), inspect outer/inner bearing raceways.",
            "checklist": ["Collect high-frequency acoustic shock pulse readings", "Inspect gear teeth contact pattern for micro-pitting", "Sample oil for metallic wear debris"],
        },
        "lubrication_pressure": {
            "issue": "Lube oil filter clogging / pump cavitation / oil seal leakage",
            "action": "Clean or replace lube oil filter cartridge, verify booster pump discharge pressure, check for external leaks.",
            "checklist": ["Inspect differential pressure across lube filter", "Check suction strainer for debris", "Verify oil reservoir level and seal integrity"],
        },
        "acoustic_emission": {
            "issue": "Micro-crack propagation / cavitation / high-frequency friction",
            "action": "Conduct ultrasonic non-destructive testing (NDT), inspect for pump cavitation or seal rubbing.",
            "checklist": ["Perform ultrasonic leak and friction detection", "Verify suction net positive suction head (NPSH)", "Inspect mechanical seal face for dry running"],
        },
    }

    DISCLAIMER_TEXT = (
        "DECISION-SUPPORT NOTICE: This recommendation is generated automatically from quantum-classical predictive "
        "algorithms and telemetry analytics. It is designed to assist maintenance engineers and should be validated "
        "with physical inspection prior to critical shutdown decisions."
    )

    def generate_recommendation(
        self,
        asset_id: str,
        asset_type: str,
        predicted_rul: float,
        risk_state: str = "NORMAL",
        anomaly_score: float = 20.0,
        top_sensors: Optional[List[Tuple[str, float]]] = None,
        current_cycle: int = 100,
    ) -> MaintenanceRecommendation:
        """
        Generates a comprehensive maintenance recommendation.
        """
        top_sensors = top_sensors or [("vibration_rms", 35.0), ("bearing_temperature", 30.0)]
        primary_sensor = top_sensors[0][0] if top_sensors else "vibration_rms"

        # Determine Risk Level and Urgency
        if risk_state == "CRITICAL" or predicted_rul <= 50.0 or anomaly_score >= 75.0:
            risk_level = "HIGH RISK"
            urgency = "IMMEDIATE_ACTION"
            window_str = f"Within {max(1, int(predicted_rul))} cycles (~{max(1, int(predicted_rul * 1.5))} operating hours)"
        elif risk_state == "WARNING" or predicted_rul <= 120.0 or anomaly_score >= 45.0:
            risk_level = "MODERATE RISK"
            urgency = "PLANNED_MAINTENANCE"
            window_str = f"Between {int(predicted_rul * 0.7)} and {int(predicted_rul * 1.3)} cycles"
        else:
            risk_level = "LOW RISK / NORMAL"
            urgency = "ROUTINE_MONITORING"
            window_str = f"Estimated > {int(predicted_rul)} cycles remaining"

        # Lookup domain diagnostics from primary sensor failure mode
        mode_info = self.FAILURE_MODES.get(primary_sensor, self.FAILURE_MODES["vibration_rms"])
        likely_issue = mode_info["issue"]
        recommended_action = mode_info["action"]
        checklist = mode_info["checklist"]

        if risk_level == "LOW RISK / NORMAL":
            likely_issue = "Normal operation; no active degradation faults detected."
            recommended_action = "Continue standard operating surveillance and periodic vibration logging."
            checklist = ["Log baseline operating telemetry", "Verify standard lubricant change schedule"]

        return MaintenanceRecommendation(
            asset_id=asset_id,
            asset_type=asset_type,
            risk_level=risk_level,
            urgency=urgency,
            predicted_rul_cycles=round(float(predicted_rul), 1),
            estimated_failure_window=window_str,
            likely_issue=likely_issue,
            recommended_action=recommended_action,
            inspection_checklist=checklist,
            top_affected_sensors=top_sensors,
            disclaimer=self.DISCLAIMER_TEXT,
        )
