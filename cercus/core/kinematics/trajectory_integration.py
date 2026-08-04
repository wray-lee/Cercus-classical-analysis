"""
Cercus Core Kinematics — Trajectory Integration
===============================================
Pure physics functions for trajectory reconstruction from body-frame
micro-displacements. No matplotlib dependency.

This module contains the core dual-stage trajectory integration algorithm:

Stage 1 — Build an *inner trajectory* via dynamic dz integration so every
          frame's local curvature and S-turns are preserved.
Stage 2 — Apply curvature-thresholded rigid macro rotation to the curved
          trajectory, producing the correct left/right fan-shaped dispersion.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pipeline.constants import RADIUS_MM
from pipeline.kinematics import _refine_offset_by_angular_velocity

log = logging.getLogger(__name__)

_DEG2RAD: float = np.pi / 180.0


def find_peak_bracket_interval(
    dz: np.ndarray,
    angular_velocity: np.ndarray,
    onset_idx: int,
    offset_idx: int,
    eps: float = 2.0,
) -> tuple[int, int]:
    """Find the bracket around the peak angular velocity within [onset, offset).

    The bracket spans from the first sign change (or below-epsilon) before the
    peak to the first such change after the peak, isolating the dominant
    rotational impulse while excluding the rebound phase. Falls back to the
    ``escape_angular_peak`` behaviour when the bracket is too short (<5 frames)
    or the peak prominence is below *eps*.
    """
    eps_rad = eps * _DEG2RAD
    search = angular_velocity[onset_idx:offset_idx]
    if len(search) == 0:
        return onset_idx, offset_idx

    # Locate the peak: frame with maximum |angular_velocity| in the window
    peak_local = int(np.nanargmax(np.abs(search)))
    peak_global = onset_idx + peak_local
    peak_val = search[peak_local]

    # Near-zero peak -> fallback to escape_angular_peak behaviour
    if abs(peak_val) < eps_rad:
        refined = _refine_offset_by_angular_velocity(
            np.arange(len(dz), dtype=float), angular_velocity,
            onset_idx, offset_idx,
        )
        end = refined if refined is not None else offset_idx
        log.warning(
            "peak_bracket peak too weak (|omega|max=%.3f rad/s < eps=%.3f), "
            "falling back to escape_angular_peak (end=%d)",
            abs(peak_val), eps_rad, end,
        )
        return onset_idx, end

    # Left bracket: scan backwards from peak for sign change or |omega| < eps
    left = peak_global
    for i in range(peak_global, onset_idx - 1, -1):
        if abs(angular_velocity[i]) < eps_rad or (
            i > onset_idx and (
                angular_velocity[i] * angular_velocity[i - 1] < 0
            )
        ):
            left = i
            break
        left = i
    # Clamp into the window [onset_idx, offset_idx)
    left = max(left, onset_idx)

    # Right bracket: scan forwards from peak for sign change or |omega| < eps
    right = peak_global
    for i in range(peak_global, min(offset_idx, len(angular_velocity))):
        if abs(angular_velocity[i]) < eps_rad or (
            i + 1 < len(angular_velocity)
            and angular_velocity[i] * angular_velocity[i + 1] < 0
        ):
            right = i
            break
        right = i
    # Clamp to the half-open window [onset_idx, offset_idx)
    right = min(right, offset_idx - 1)

    # Validate bracket size: < 5 frames -> fallback to escape_angular_peak
    bracket_len = right - left + 1
    if bracket_len < 5:
        refined = _refine_offset_by_angular_velocity(
            np.arange(len(dz), dtype=float), angular_velocity,
            onset_idx, offset_idx,
        )
        end = refined if refined is not None else offset_idx
        log.warning(
            "peak_bracket interval too short (%d frames at %d..%d), "
            "falling back to escape_angular_peak (end=%d)",
            bracket_len, left, right, end,
        )
        return onset_idx, end

    log.debug(
        "peak_bracket: peak=%d (|omega|max=%.3f rad/s), bracket=%d..%d, len=%d",
        peak_global, abs(peak_val), left, right, bracket_len,
    )
    return left, right


def integrate_body_trajectory(
    burst_dx: np.ndarray,
    burst_dy: np.ndarray,
    burst_dz: np.ndarray,
    *,
    use_heading: bool = True,
    use_rigid_rotation: bool = True,
    macro_yaw_override: float | None = None,
    heading_dz: np.ndarray | None = None,
    heading_offset: float = 0.0,
    src_idx: np.ndarray | None = None,
    dst_idx: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Dual-stage trajectory integration core.

    Stage 1 — Build an *inner trajectory* via dynamic dz integration so every
              frame's local curvature and S-turns are preserved.
              Controlled by *use_heading*.
    Stage 2 — Apply curvature-thresholded rigid macro rotation to the curved
              trajectory, producing the correct left/right fan-shaped
              dispersion without flattening the natural bends.
              Controlled by *use_rigid_rotation* (implies Stage 1).

    Parameters
    ----------
    burst_dx, burst_dy, burst_dz : np.ndarray
        Pre-extracted body-frame micro-displacements (already masked/sliced).
    use_heading : bool
        Whether to apply Stage 1 per-frame heading integration from dz.
    use_rigid_rotation : bool
        Whether to apply Stage 2 macro rotation (implies Stage 1).
    macro_yaw_override : float | None
        If provided, overrides the total yaw computed from ``sum(burst_dz)``
        for Stage 2 rigid rotation.
    heading_dz : np.ndarray | None
        Optional separate dz array for Stage 1 per-frame heading integration.
    heading_offset : float
        Initial heading offset in radians.
    src_idx, dst_idx : np.ndarray | None
        Frame indices for heading interpolation when arrays have different lengths.
    """
    if len(burst_dx) == 0:
        return None, None

    _heading_dz = heading_dz if heading_dz is not None else burst_dz

    if use_heading:
        # Stage 1: build curved trajectory via per-frame heading integration
        local_heading = np.cumsum(_heading_dz) / RADIUS_MM
        local_heading -= local_heading[0]
        local_heading += heading_offset

        # Interpolate heading onto the xy grid if lengths differ
        if len(local_heading) != len(burst_dx):
            if src_idx is None:
                src_idx = np.arange(len(_heading_dz))
            if dst_idx is None:
                dst_idx = np.linspace(0, len(_heading_dz) - 1, len(burst_dx))
            if len(src_idx) >= 2:
                local_heading = np.interp(dst_idx, src_idx, local_heading)
            else:
                local_heading = np.full(len(burst_dx), local_heading[0])

        dx_inner = burst_dx * np.cos(local_heading) - burst_dy * np.sin(local_heading)
        dy_inner = burst_dx * np.sin(local_heading) + burst_dy * np.cos(local_heading)

        x_inner = np.cumsum(dx_inner)
        y_inner = np.cumsum(dy_inner)
    else:
        # Straight accumulation without per-frame heading rotation
        x_inner = np.cumsum(burst_dx)
        y_inner = np.cumsum(burst_dy)

    # Stage 2: rigid macro rotation (with curvature thresholding)
    if use_rigid_rotation:
        total_yaw_rad = np.sum(burst_dz) / RADIUS_MM
        macro_yaw = macro_yaw_override if macro_yaw_override is not None else total_yaw_rad

        # Curvature Thresholding
        MACRO_YAW_THRESHOLD = np.pi / 2   # 90° — high intrinsic curvature
        COMP_ANGLE_MAX = np.pi / 2        # 90° — max allowed compensation

        comp_angle = macro_yaw

        yaw_abs = abs(macro_yaw)
        if yaw_abs > MACRO_YAW_THRESHOLD:
            decay_factor = max(
                0.0, 1.0 - (yaw_abs - MACRO_YAW_THRESHOLD) / MACRO_YAW_THRESHOLD
            )
            comp_angle *= decay_factor

        comp_angle = np.clip(comp_angle, -COMP_ANGLE_MAX, COMP_ANGLE_MAX)

        cos_yaw = np.cos(comp_angle)
        sin_yaw = np.sin(comp_angle)

        traj_x = x_inner * cos_yaw - y_inner * sin_yaw
        traj_y = x_inner * sin_yaw + y_inner * cos_yaw
    else:
        traj_x = x_inner
        traj_y = y_inner

    # Origin alignment: start from (0, 0)
    traj_x -= traj_x[0]
    traj_y -= traj_y[0]

    return traj_x, traj_y


def build_angular_peak_dz_mask(
    angular_velocity: np.ndarray,
    onset_idx: int,
    offset_idx: int,
    total_len: int,
) -> np.ndarray:
    """Build a dz mask spanning [onset, angular-velocity peak -> zero-crossing].

    Reuses :func:`_refine_offset_by_angular_velocity` to locate the first
    zero-crossing after the angular-velocity peak within the escape interval.
    Only dz frames in this direction-consistent range are included, filtering
    out the rebound phase of the air-floating ball.
    """
    mask = np.zeros(total_len, dtype=bool)
    refined = _refine_offset_by_angular_velocity(
        np.arange(total_len, dtype=float), angular_velocity, onset_idx, offset_idx,
    )
    end = refined if refined is not None else offset_idx
    mask[onset_idx:end + 1] = True
    return mask


def body_to_traj(
    grp: pd.DataFrame,
    mask_xy: np.ndarray,
    *,
    use_z: bool,
    use_rigid_rotation: bool = False,
    mask_z: np.ndarray | None = None,
    heading_dz_mask: np.ndarray | None = None,
    macro_yaw_override: float | None = None,
    heading_offset: float = 0.0,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """DataFrame wrapper around integrate_body_trajectory.

    Extracts body-frame dx/dy/dz from *grp* using boolean masks, then
    delegates to the shared integration core.
    """
    if mask_z is None:
        mask_z = mask_xy

    dx_body = -grp["dx"].fillna(0).values
    dy_body = -grp["dy"].fillna(0).values
    dz_body = grp["dz"].fillna(0).values

    burst_dx = dx_body[mask_xy]
    burst_dy = dy_body[mask_xy]
    burst_dz = dz_body[mask_z]

    heading_dz = dz_body[heading_dz_mask] if heading_dz_mask is not None else None

    _src = heading_dz_mask if heading_dz_mask is not None else mask_z
    _hd_len = len(heading_dz if heading_dz is not None else burst_dz)
    src_idx = np.flatnonzero(_src) if _hd_len != len(burst_dx) else None
    dst_idx = np.flatnonzero(mask_xy) if _hd_len != len(burst_dx) else None

    return integrate_body_trajectory(
        burst_dx, burst_dy, burst_dz,
        use_heading=use_z,
        use_rigid_rotation=use_rigid_rotation,
        macro_yaw_override=macro_yaw_override,
        heading_dz=heading_dz,
        heading_offset=heading_offset,
        src_idx=src_idx, dst_idx=dst_idx,
    )