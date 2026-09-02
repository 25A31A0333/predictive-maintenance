"""
Real-Time Industrial Sensor Telemetry Streaming Simulator.

Simulates live streaming telemetry ticks with:
- Multi-asset operational dynamics
- Continuous fluctuating production load regimes
- Injected progressive degradation trajectories
- Real-time buffer management for live dashboard monitoring
"""

from typing import Dict, List, Optional, Iterator, Tuple
import numpy as np
import pandas as pd
from src.data.telemetry_generator import IndustrialAssetTelemetryGenerator, AssetConfig


class IndustrialTelemetryStreamer:
    """
    Simulates a live streaming sensor feed for industrial equipment monitoring.
    """

    def __init__(
        self,
        asset_type: str = "refinery_compressor",
        total_cycles: int = 320,
        fault_onset_cycle: int = 160,
        noise_std: float = 0.035,
        random_seed: int = 42,
    ):
        self.asset_type = asset_type
        self.config = AssetConfig(
            asset_id=f"STREAM_{asset_type.upper()}",
            asset_type=asset_type,
            total_cycles=total_cycles,
            fault_onset_cycle=fault_onset_cycle,
            noise_std=noise_std,
            random_seed=random_seed,
        )
        self.generator = IndustrialAssetTelemetryGenerator(self.config)
        self.full_telemetry = self.generator.generate_single_run()
        self.current_step = 0

    def get_snapshot(self, up_to_cycle: int) -> pd.DataFrame:
        """Returns telemetry buffer from cycle 1 up to the specified cycle."""
        k = max(1, min(up_to_cycle, len(self.full_telemetry)))
        return self.full_telemetry.iloc[:k].copy()

    def get_latest_tick(self, cycle: int) -> Dict[str, float]:
        """Returns the single sensor reading dictionary for the given cycle."""
        idx = max(0, min(cycle - 1, len(self.full_telemetry) - 1))
        row = self.full_telemetry.iloc[idx]
        return row.to_dict()

    def stream_ticks(self) -> Iterator[Tuple[int, Dict[str, float]]]:
        """Generator yielding (cycle_number, sensor_reading_dict) sequentially."""
        for i in range(len(self.full_telemetry)):
            row = self.full_telemetry.iloc[i]
            yield int(row["cycle"]), row.to_dict()
