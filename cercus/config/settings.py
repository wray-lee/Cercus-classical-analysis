"""
Cercus Framework — Trajectory Configuration
===========================================
Pydantic-based trajectory configuration with cross-field validation.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Literal

from pydantic import BaseModel, field_validator, model_validator

log = logging.getLogger(__name__)


class HeatmapConfig(BaseModel):
    """Heatmap rendering configuration."""
    t_window_ttc: tuple[float, float] = (-1.0, 2.0)
    t_window_onset: tuple[float, float] = (-1.0, 1.5)
    t_bin_s: float = 0.01
    vmax: float = 50.0
    gamma: float = 0.4

    @field_validator("t_window_onset")
    @classmethod
    def _validate_onset_window(cls, v: tuple[float, float]) -> tuple[float, float]:
        if v[0] > -1.0:
            log.warning(
                "t_window_onset[0]=%.2f > -1.0; pre-window [-1.0, 0] not fully covered. "
                "Clamping to -1.0.", v[0]
            )
            return (-1.0, v[1])
        if v[0] < -1.0:
            log.warning(
                "t_window_onset[0]=%.2f < -1.0; shows data outside classification window, "
                "will look like contamination.", v[0]
            )
        return v


class BarLabelStyle(str, Enum):
    """Bar chart percentage label placement style."""
    INLINE = "inline"       # label inside bar (intuitive)
    AXIS = "axis"           # label via dashed line to y-axis (publication style)


class TrajectoryConfig(BaseModel, frozen=True):
    """Validated trajectory rendering configuration.

    Cross-field invariants:
    - If *use_rigid_rotation* is False, *dz_integration_range* is ignored
      (warns at construction time).
    - If *use_escape_onset_heading* is True and *use_z_degree_to_draw* is False,
      raises ValueError — escape-onset heading requires z-degree trajectory.
    """

    use_z_degree_to_draw: bool = True
    use_rigid_rotation: bool = False
    use_escape_onset_heading: bool = True
    use_escape_onset_only_xy: bool = True
    use_angular_velocity_offset: bool = False

    dz_integration_range: Literal[
        "full_trial", "escape_interval", "trial_to_onset",
        "escape_angular_peak", "escape_onset_heading",
    ] = "escape_interval"

    bar_label_style: BarLabelStyle = BarLabelStyle.INLINE

    @field_validator("dz_integration_range")
    @classmethod
    def _validate_dz_range(cls, v: str) -> str:
        valid = {
            "full_trial", "escape_interval", "trial_to_onset",
            "escape_angular_peak", "escape_onset_heading",
        }
        if v not in valid:
            log.warning(
                "Invalid dz_integration_range=%r, falling back to 'escape_interval'", v
            )
            return "escape_interval"
        return v

    @model_validator(mode="after")
    def _validate_rigid_rotation(self) -> "TrajectoryConfig":
        if not self.use_rigid_rotation and self.dz_integration_range != "escape_interval":
            log.warning(
                "use_rigid_rotation=False — dz_integration_range=%r is ignored. "
                "Set use_rigid_rotation=True to apply non-default dz integration.",
                self.dz_integration_range,
            )
        if self.use_escape_onset_heading and not self.use_z_degree_to_draw:
            raise ValueError(
                "use_escape_onset_heading=True requires use_z_degree_to_draw=True. "
                "Escape-onset heading correction needs z-degree trajectory integration."
            )
        return self

    @classmethod
    def from_dict(cls, cfg: dict) -> "TrajectoryConfig":
        """Build from a raw config dict (e.g. from YAML)."""
        traj = cfg.get("trajectory", {})
        viz = cfg.get("visualization", {})

        return cls(
            use_z_degree_to_draw=bool(traj.get("use_z_degree_to_draw", True)),
            use_rigid_rotation=bool(traj.get("use_rigid_rotation", False)),
            use_escape_onset_heading=bool(traj.get("use_escape_onset_heading", True)),
            use_escape_onset_only_xy=bool(traj.get("use_escape_onset_only_xy", True)),
            use_angular_velocity_offset=bool(traj.get("use_angular_velocity_offset", False)),
            dz_integration_range=traj.get("dz_integration_range", "escape_interval"),
            bar_label_style=BarLabelStyle(viz.get("bar_label_style", "inline")),
        )
