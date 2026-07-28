"""
Cercus Core — Domain Models
============================
Typed data classes for trials, escape intervals, and classification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ResponseType(str, Enum):
    """Ternary behavioral response classification."""
    NO_RESPONSE = "NoResponse"
    PRE_WALK = "PreWalk"
    ESCAPE = "Escape"


@dataclass(frozen=True)
class EscapeInterval:
    """Time interval of the escape burst, relative to TTC (ms)."""
    onset_ms: float
    offset_ms: float
    duration_ms: float

    @classmethod
    def empty(cls) -> "EscapeInterval":
        return cls(onset_ms=float("nan"), offset_ms=float("nan"), duration_ms=float("nan"))

    @property
    def is_valid(self) -> bool:
        import math
        return not (math.isnan(self.onset_ms) or math.isnan(self.offset_ms))


@dataclass
class TrialFeatures:
    """Physical features extracted from one trial's kinematics."""
    v_max: float
    latency_ms: float
    trial_type: str | None = None
    stim_onset_t_rel: float | None = None


@dataclass
class Trial:
    """Typed representation of a single trial's data."""
    global_trial_id: int
    subject_id: str = ""
    trial_type: str = ""
    response_type: str = "NoResponse"
    v_max: float = float("nan")
    latency_ms: float = float("nan")
    interval_onset_ms: float = float("nan")
    interval_offset_ms: float = float("nan")
    screen_side: str = ""
    wind_dir: str = ""
    direction: str = ""
    side: str = ""
    target_ttc_ms: float = float("nan")
