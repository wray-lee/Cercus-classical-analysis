"""Data models and IPC message types for Cercus Analysis."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np
import pandas as pd


class CommandAction(str, Enum):
    """Valid commands for DataProcessor."""
    PROCESS_BATCH = "process_batch"
    PROCESS_GROUP_BATCH = "process_group_batch"
    COMPUTE_PSTH = "compute_psth"
    EXPORT_RESULTS = "export_results"
    SHUTDOWN = "shutdown"


class TelemetryStatus(str, Enum):
    """Valid telemetry statuses from DataProcessor."""
    PROCESSING = "processing"
    COMPLETE = "complete"
    ERROR = "error"
    PROGRESS = "progress"


@dataclass(frozen=True)
class Command:
    """Message from GUI to DataProcessor via cmd_queue."""
    action: CommandAction
    params: Dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass
class Telemetry:
    """Message from DataProcessor to GUI via telemetry_queue."""
    status: TelemetryStatus
    request_id: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: float = 0.0


@dataclass
class ProcessingParams:
    """Parameters for signal processing pipeline."""
    input_dir: str
    output_dir: str
    filter_cutoff_hz: float = 10.0
    filter_order: int = 4
    resample_freq_hz: float = 100.0
    freezing_threshold_mm_s: float = 0.5
    freezing_min_duration_ms: float = 1000.0
    psth_window_pre_s: float = -2.0
    psth_window_post_s: float = 5.0
    escape_acceleration_threshold: float = 2.0
    stimulus_offset_s: float = 0.0


@dataclass
class KinematicData:
    """Processed kinematic data for a single recording."""
    session_id: str
    time: np.ndarray
    speed: np.ndarray
    angular_velocity: np.ndarray
    trajectory_x: np.ndarray
    trajectory_y: np.ndarray
    raw_kinematics: pd.DataFrame
    raw_events: Optional[pd.DataFrame] = None
    heading_angle: Optional[np.ndarray] = None       # Integrated Z-axis: θ = cumsum(dz)
    steering_sine: Optional[np.ndarray] = None        # sin(heading_angle)
    decoupled_dx: Optional[np.ndarray] = None         # Filtered lateral displacement
    decoupled_dy: Optional[np.ndarray] = None         # Filtered forward/back displacement
    decoupled_dz: Optional[np.ndarray] = None         # Filtered rotational displacement


@dataclass
class PSTHResult:
    """Peri-Stimulus Time Histogram result for one event type."""
    event_type: str
    time_axis: np.ndarray
    mean_speed: np.ndarray
    sem_speed: np.ndarray
    mean_angular_velocity: np.ndarray
    sem_angular_velocity: np.ndarray
    n_trials: int
    speed_trials: Optional[np.ndarray] = None
    trial_x: Optional[np.ndarray] = None
    trial_y: Optional[np.ndarray] = None


@dataclass
class FreezingEpisode:
    """Single freezing episode detected."""
    start_time: float
    end_time: float
    duration_ms: float


@dataclass
class EscapeBout:
    """Single escape bout detected."""
    peak_time: float
    latency_from_stimulus: float
    peak_velocity: float
    total_distance: float


@dataclass
class SessionResults:
    """Aggregated results for a single recording session."""
    session_id: str
    kinematic_data: KinematicData
    psth_results: Dict[str, PSTHResult] = field(default_factory=dict)
    freezing_episodes: List[FreezingEpisode] = field(default_factory=list)
    escape_bouts: List[EscapeBout] = field(default_factory=list)
    mean_speed: float = 0.0
    mean_angular_velocity: float = 0.0
    total_distance: float = 0.0
    total_freezing_time_s: float = 0.0
    group_label: str = "treatment"
