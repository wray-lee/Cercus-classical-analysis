"""DataProcessor: Pure computation engine running in isolated process.

This module handles all heavy data processing, signal analysis, and file I/O.
It runs in a separate process from the GUI and communicates exclusively via
multiprocessing.Queue objects (cmd_queue and telemetry_queue).
"""

import json
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import CubicSpline

from src.models import (
    Command,
    CommandAction,
    EscapeBout,
    FreezingEpisode,
    KinematicData,
    PSTHResult,
    ProcessingParams,
    SessionResults,
    Telemetry,
    TelemetryStatus,
)

logger = logging.getLogger(__name__)


class DataProcessor:
    """Runs in isolated process. Communicates only via queues."""

    def __init__(
        self,
        cmd_queue: mp.Queue,
        telemetry_queue: mp.Queue,
    ) -> None:
        self._cmd_queue: mp.Queue = cmd_queue
        self._telemetry_queue: mp.Queue = telemetry_queue
        self._running: bool = True
        self._params: Optional[ProcessingParams] = None
        self._cached_results: List[SessionResults] = []

    def run(self) -> None:
        """Main loop: poll cmd_queue, dispatch, send telemetry."""
        logger.info("DataProcessor started")
        while self._running:
            try:
                cmd: Command = self._cmd_queue.get(timeout=0.1)
                self._dispatch(cmd)
            except Exception:
                # Empty queue or timeout — continue polling
                continue
        logger.info("DataProcessor shutdown")

    def _dispatch(self, cmd: Command) -> None:
        """Route command to appropriate handler."""
        handlers = {
            CommandAction.PROCESS_BATCH: self._handle_process_batch,
            CommandAction.PROCESS_GROUP_BATCH: self._handle_process_group_batch,
            CommandAction.COMPUTE_PSTH: self._handle_compute_psth,
            CommandAction.EXPORT_RESULTS: self._handle_export_results,
            CommandAction.SHUTDOWN: self._handle_shutdown,
        }
        handler = handlers.get(cmd.action)
        if handler is None:
            self._send_error(cmd.request_id, f"Unknown action: {cmd.action}")
            return
        try:
            handler(cmd)
        except Exception as e:
            logger.exception("Error handling command %s", cmd.action)
            self._send_error(cmd.request_id, str(e))

    def _send_telemetry(
        self,
        status: TelemetryStatus,
        request_id: str,
        data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        progress: float = 0.0,
    ) -> None:
        """Send telemetry message to GUI."""
        msg = Telemetry(
            status=status,
            request_id=request_id,
            data=data,
            error=error,
            progress=progress,
        )
        self._telemetry_queue.put(msg)

    def _send_error(self, request_id: str, error: str) -> None:
        """Convenience: send error telemetry."""
        self._send_telemetry(TelemetryStatus.ERROR, request_id, error=error)

    def _handle_shutdown(self, cmd: Command) -> None:
        """Graceful shutdown."""
        self._running = False

    # -------------------------------------------------------------------------
    # Data Loading
    # -------------------------------------------------------------------------

    def _discover_csv_pairs(
        self, input_dir: Path, recursive: bool = False,
    ) -> List[Tuple[Path, Optional[Path]]]:
        """Find kinematics CSV files and their matching events files.

        Supports naming patterns:
          - {name}_kinematics.csv / {name}_events.csv
          - {name}-kinematics.csv / {name}-events.csv
          - {name}kinematics.csv / {name}events.csv

        When *recursive* is ``True``, searches all subdirectories via
        ``rglob`` instead of a single-level ``glob``.
        """
        pairs: List[Tuple[Path, Optional[Path]]] = []
        kin_files = sorted(input_dir.rglob("*kinematics*.csv") if recursive else input_dir.glob("*kinematics*.csv"))

        for kin_file in kin_files:
            stem = kin_file.stem.lower()
            # Strip 'kinematics' to get base name
            base = stem.replace("_kinematics", "").replace("-kinematics", "").replace("kinematics", "")
            base = base.rstrip("_-")

            # Search for matching events file with any separator pattern
            evt_candidates = [
                kin_file.parent / f"{base}_events.csv",
                kin_file.parent / f"{base}-events.csv",
                kin_file.parent / f"{base}events.csv",
            ]
            evt_file = None
            for candidate in evt_candidates:
                if candidate.exists():
                    evt_file = candidate
                    break

            pairs.append((kin_file, evt_file))

        return pairs

    def _load_kinematics(self, filepath: Path) -> pd.DataFrame:
        """Load and validate a kinematics CSV, preserving hardware stim_state."""
        df = pd.read_csv(filepath)
        required = {"sys_time", "dx", "dy", "dz"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"Missing columns in {filepath.name}: {missing}")

        cols = ["sys_time", "dx", "dy", "dz"]
        agg_dict = {"dx": "sum", "dy": "sum", "dz": "sum"}

        # Preserve hardware state if available
        if "stim_state" in df.columns:
            cols.append("stim_state")
            agg_dict["stim_state"] = "max"

        return (
            df[cols]
            .dropna()
            .sort_values("sys_time")
            .groupby("sys_time", as_index=False)
            .agg(agg_dict)
            .sort_values("sys_time")
            .reset_index(drop=True)
        )

    def _load_events(self, filepath: Path) -> pd.DataFrame:
        """Load events CSV with state-machine parsing for trial anchoring.

        Hardware logs store stimulus metadata in ``trial_start`` rows
        (JSON ``details`` column).  This method walks the event table
        sequentially:

        1. On ``trial_start`` — parse ``details`` for base type +
           direction, then **immediately** emit a real event using the
           ``trial_start`` row's own ``sys_time`` as the stimulus anchor.
        2. ``iti_start`` rows are passed through unchanged (they mark the
           inter-trial interval, not the stimulus).
        3. All other event types pass through unchanged.

        For files that lack ``trial_start``/``iti_start`` semantics the
        method falls back to plain passthrough of all event rows.
        """
        df = pd.read_csv(filepath)

        # Map common column name variants
        if "timestamp" in df.columns and "sys_time" not in df.columns:
            df = df.rename(columns={"timestamp": "sys_time"})
        if "event_name" in df.columns and "event_type" not in df.columns:
            df = df.rename(columns={"event_name": "event_type"})

        required = {"sys_time", "event_type"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"Missing columns in {filepath.name}: {missing}")

        df = df.dropna(subset=["sys_time", "event_type"]).sort_values("sys_time")

        # Detect whether this file uses trial/iti anchoring
        event_types_lower = df["event_type"].str.lower()
        has_trial_start = event_types_lower.str.contains("trial_start").any()

        if not has_trial_start:
            # Fallback: plain passthrough
            return (
                df[["sys_time", "event_type"]]
                .drop_duplicates(subset=["sys_time"], keep="first")
                .reset_index(drop=True)
            )

        # ---- State-machine parsing ----
        parsed_events: List[Dict[str, Any]] = []

        for _, row in df.iterrows():
            evt_lower = str(row["event_type"]).lower().strip()

            if evt_lower == "trial_start":
                # Parse metadata from details column and emit immediately
                # using the trial_start time as the stimulus anchor
                composite = self._parse_trial_start_label(row)
                if composite:
                    parsed_events.append({
                        "sys_time": row["sys_time"],
                        "event_type": composite,
                    })

            elif evt_lower == "iti_start":
                # Pass through unchanged — iti marks the inter-trial
                # interval, not the stimulus onset
                parsed_events.append({
                    "sys_time": row["sys_time"],
                    "event_type": str(row["event_type"]).strip(),
                })

            else:
                # Any other event type passes through unchanged
                parsed_events.append({
                    "sys_time": row["sys_time"],
                    "event_type": str(row["event_type"]).strip(),
                })

        result = pd.DataFrame(parsed_events, columns=["sys_time", "event_type"])
        result = result.drop_duplicates(subset=["sys_time"], keep="first").reset_index(drop=True)

        # Enforce strict trial alignment: truncate trailing trials of
        # majority event types to match the minority count so that
        # downstream PSTH comparisons operate on balanced trial sets.
        if len(result) > 0:
            evt_counts = result["event_type"].value_counts()
            if len(evt_counts) > 1:
                min_count = evt_counts.min()
                mask = pd.Series(True, index=result.index)
                for evt_type, count in evt_counts.items():
                    if count > min_count:
                        evt_idx = result.index[result["event_type"] == evt_type]
                        # Drop the trailing (count - min_count) trials
                        drop_idx = evt_idx[min_count:]
                        mask.loc[drop_idx] = False
                        logger.warning(
                            "Truncating %d '%s' trials to match minority count (%d) — "
                            "likely hardware event drops",
                            count - min_count,
                            evt_type,
                            min_count,
                        )
                result = result.loc[mask].reset_index(drop=True)

        return result

    @staticmethod
    def _parse_trial_start_label(row: pd.Series) -> Optional[str]:
        """Extract composite label from a trial_start row's details JSON.

        Returns e.g. ``"looming_wind_left"`` or ``None`` on parse failure
        (caller should skip emitting an event for unparseable rows).
        """
        raw_details = row.get("details")
        if raw_details is None or (isinstance(raw_details, float) and np.isnan(raw_details)):
            return None

        try:
            details = json.loads(raw_details) if isinstance(raw_details, str) else raw_details
        except (json.JSONDecodeError, TypeError):
            return None

        if not isinstance(details, dict):
            return None

        # Extract base stimulus type from "type" field
        base_type = details.get("type", "")
        if not base_type:
            return None
        base_type = str(base_type).strip().lower().replace(" ", "_")

        # Extract direction / side (optional)
        direction = None
        for dir_key in ("wind_dir", "screen_side", "direction", "side"):
            if dir_key in details and details[dir_key]:
                direction = str(details[dir_key]).strip().lower()
                break

        if direction:
            return f"{base_type}_{direction}"
        return base_type

    # -------------------------------------------------------------------------
    # Signal Processing
    # -------------------------------------------------------------------------

    def _compute_uniform_dt(self, timestamps: np.ndarray) -> float:
        """Compute uniform time delta from timestamp array."""
        diffs = np.diff(timestamps)
        if len(diffs) == 0:
            raise ValueError("Timestamp array has fewer than 2 elements")
        median_dt = float(np.median(diffs))
        if median_dt <= 0:
            raise ValueError(f"Non-positive median dt: {median_dt}")
        return median_dt

    def _resample_to_uniform(
        self,
        timestamps: np.ndarray,
        values: np.ndarray,
        target_freq_hz: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resample unevenly sampled data to fixed frequency using cubic spline.

        Pads edges by replicating boundary values to prevent spline artifacts.
        """
        if len(timestamps) < 4:
            raise ValueError("Cubic spline requires at least 4 data points")

        t_start = timestamps[0]
        t_end = timestamps[-1]
        dt_target = 1.0 / target_freq_hz
        t_uniform = np.arange(t_start, t_end, dt_target)

        # Cubic spline with natural boundary conditions
        cs = CubicSpline(timestamps, values, bc_type="natural")
        values_uniform = cs(t_uniform)

        return t_uniform, values_uniform

    def _apply_lowpass_filter(
        self,
        data: np.ndarray,
        cutoff_hz: float,
        sample_rate_hz: float,
        order: int = 4,
    ) -> np.ndarray:
        """Zero-phase low-pass Butterworth filter.

        Uses filtfilt for zero-phase distortion. Pads signal edges to reduce
        boundary artifacts.
        """
        nyquist = sample_rate_hz / 2.0
        normalized_cutoff = cutoff_hz / nyquist

        if normalized_cutoff >= 1.0:
            logger.warning("Filter cutoff >= Nyquist, returning unfiltered data")
            return data
        if normalized_cutoff <= 0.0:
            raise ValueError(f"Invalid normalized cutoff: {normalized_cutoff}")

        b, a = signal.butter(order, normalized_cutoff, btype="low")

        # Pad signal to reduce edge effects (reflect 2x filter length)
        pad_len = min(3 * max(len(b), len(a)), len(data) // 2)
        if pad_len > 0:
            padded = np.pad(data, pad_len, mode="reflect")
            filtered = signal.filtfilt(b, a, padded)
            return filtered[pad_len:-pad_len] if pad_len > 0 else filtered
        return signal.filtfilt(b, a, data)

    def _compute_kinematics(
        self,
        df: pd.DataFrame,
        params: ProcessingParams,
    ) -> Tuple[KinematicData, float]:
        """Full signal processing pipeline for one recording.

        Returns (KinematicData, median_raw_dt) tuple.
        """
        timestamps = df["sys_time"].values.astype(np.float64)
        dx = df["dx"].values.astype(np.float64)
        dy = df["dy"].values.astype(np.float64)
        dz = df["dz"].values.astype(np.float64)

        # Uniform resampling
        raw_dt = self._compute_uniform_dt(timestamps)
        t_uniform, dx_uniform = self._resample_to_uniform(
            timestamps, dx, params.resample_freq_hz
        )
        _, dy_uniform = self._resample_to_uniform(
            timestamps, dy, params.resample_freq_hz
        )
        _, dz_uniform = self._resample_to_uniform(
            timestamps, dz, params.resample_freq_hz
        )

        # Hardware Correction: Hardware outputs -dy for forward motion.
        # Invert to standard Cartesian (+dy = forward).
        dy_uniform = dy_uniform * -1.0

        # Low-pass filter
        dx_filt = self._apply_lowpass_filter(
            dx_uniform, params.filter_cutoff_hz, params.resample_freq_hz, params.filter_order
        )
        dy_filt = self._apply_lowpass_filter(
            dy_uniform, params.filter_cutoff_hz, params.resample_freq_hz, params.filter_order
        )
        dz_filt = self._apply_lowpass_filter(
            dz_uniform, params.filter_cutoff_hz, params.resample_freq_hz, params.filter_order
        )

        # Speed (mm/s) and angular velocity (deg/s or rad/s depending on input)
        dt_resample = 1.0 / params.resample_freq_hz
        speed = np.sqrt(dx_filt**2 + dy_filt**2) / dt_resample
        angular_velocity = dz_filt / dt_resample

        # Trajectory reconstruction (cumulative displacement)
        traj_x = np.cumsum(dx_filt)
        traj_y = np.cumsum(dy_filt)

        # Heading angle: integrate Z-axis rotational displacement
        heading_angle = np.cumsum(dz_filt)
        steering_sine = np.sin(heading_angle)

        kin_data = KinematicData(
            session_id="",  # Set by caller
            time=t_uniform,
            speed=speed,
            angular_velocity=angular_velocity,
            trajectory_x=traj_x,
            trajectory_y=traj_y,
            raw_kinematics=df,
            heading_angle=heading_angle,
            steering_sine=steering_sine,
            decoupled_dx=dx_filt,
            decoupled_dy=dy_filt,
            decoupled_dz=dz_filt,
        )

        return kin_data, raw_dt

    # -------------------------------------------------------------------------
    # Ethological State Detection
    # -------------------------------------------------------------------------

    def _detect_freezing(
        self,
        speed: np.ndarray,
        time: np.ndarray,
        threshold_mm_s: float,
        min_duration_ms: float,
    ) -> List[FreezingEpisode]:
        """Detect freezing episodes: speed < threshold for > min_duration.

        Uses contiguous region detection with proper edge handling.
        """
        is_frozen = speed < threshold_mm_s
        episodes: List[FreezingEpisode] = []

        if len(is_frozen) == 0:
            return episodes

        # Find contiguous regions via diff
        padded = np.concatenate([[False], is_frozen, [False]])
        changes = np.diff(padded.astype(int))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]

        for s, e in zip(starts, ends):
            # e is exclusive; last frozen sample is at e-1
            if e - 1 >= len(time):
                e = len(time)
            duration_ms = (time[e - 1] - time[s]) * 1000.0
            if duration_ms >= min_duration_ms:
                episodes.append(FreezingEpisode(
                    start_time=float(time[s]),
                    end_time=float(time[e - 1]),
                    duration_ms=float(duration_ms),
                ))
        return episodes

    def _detect_escape_bouts(
        self,
        speed: np.ndarray,
        time: np.ndarray,
        acceleration_threshold: float,
        window_s: float = 0.5,
        min_prominence: float = 1.0,
    ) -> List[EscapeBout]:
        """Detect escape bouts from acceleration peaks.

        Identifies acceleration peaks above threshold, then finds the
        subsequent velocity peak within a window.
        """
        if len(speed) < 3:
            return []

        dt = float(np.median(np.diff(time)))
        acceleration = np.diff(speed) / dt
        # Pad to match time array length (acceleration[i] corresponds to time[i+1])
        accel_time = time[1:]

        peak_indices, properties = signal.find_peaks(
            acceleration,
            height=acceleration_threshold,
            prominence=min_prominence,
        )

        bouts: List[EscapeBout] = []
        window_samples = int(window_s / dt)

        for idx in peak_indices:
            # Find velocity peak within window after acceleration peak
            window_end = min(idx + window_samples, len(speed))
            vel_window = speed[idx:window_end]
            if len(vel_window) == 0:
                continue

            peak_vel_offset = int(np.argmax(vel_window))
            peak_vel_idx = idx + peak_vel_offset

            # Integrate speed over the bout window for total distance
            total_dist = float(np.trapz(np.abs(vel_window), dx=dt))

            bouts.append(EscapeBout(
                peak_time=float(time[peak_vel_idx]),
                latency_from_stimulus=0.0,  # Set during PSTH alignment
                peak_velocity=float(speed[peak_vel_idx]),
                total_distance=total_dist,
            ))
        return bouts

    def _compute_escape_latency_from_stimulus(
        self,
        escape_bouts: List[EscapeBout],
        stimulus_onset_time: float,
        max_latency_s: float = 2.0,
    ) -> List[EscapeBout]:
        """Set latency_from_stimulus for escape bouts near a stimulus."""
        updated: List[EscapeBout] = []
        for bout in escape_bouts:
            latency = bout.peak_time - stimulus_onset_time
            if 0 <= latency <= max_latency_s:
                updated.append(EscapeBout(
                    peak_time=bout.peak_time,
                    latency_from_stimulus=latency,
                    peak_velocity=bout.peak_velocity,
                    total_distance=bout.total_distance,
                ))
        return updated

    # -------------------------------------------------------------------------
    # PSTH Analysis
    # -------------------------------------------------------------------------

    def _compute_psth_for_event_type(
        self,
        event_type: str,
        event_times: np.ndarray,
        time_axis: np.ndarray,
        speed: np.ndarray,
        angular_velocity: np.ndarray,
        pre_s: float,
        post_s: float,
        resample_freq_hz: float,
        dx_filt: Optional[np.ndarray] = None,
        dy_filt: Optional[np.ndarray] = None,
    ) -> PSTHResult:
        """Compute PSTH for a single event type.

        Trials that extend beyond data boundaries are padded with NaN
        and excluded from the mean/SEM computation (pairwise deletion).

        When dx_filt and dy_filt are provided, per-trial spatial
        coordinates are extracted, cumulatively summed, and zero-aligned
        to the stimulus onset position (t=0 → origin).
        """
        n_pre = int(abs(pre_s) * resample_freq_hz)
        n_post = int(post_s * resample_freq_hz)
        n_total = n_pre + n_post

        speed_trials: List[np.ndarray] = []
        angvel_trials: List[np.ndarray] = []
        trial_x_list: List[np.ndarray] = []
        trial_y_list: List[np.ndarray] = []

        has_spatial = dx_filt is not None and dy_filt is not None

        for evt_time in event_times:
            idx = np.searchsorted(time_axis, evt_time, side="left")
            start_idx = idx - n_pre
            end_idx = idx + n_post

            # Extract with NaN padding for out-of-bounds
            speed_trial = np.full(n_total, np.nan)
            angvel_trial = np.full(n_total, np.nan)

            # Compute valid overlap region
            src_start = max(0, start_idx)
            src_end = min(len(speed), end_idx)
            dst_start = src_start - start_idx
            dst_end = dst_start + (src_end - src_start)

            if src_start < src_end:
                speed_trial[dst_start:dst_end] = speed[src_start:src_end]
                angvel_trial[dst_start:dst_end] = angular_velocity[src_start:src_end]

            speed_trials.append(speed_trial)
            angvel_trials.append(angvel_trial)

            # Spatial trajectory extraction with cumsum and origin reset
            if has_spatial:
                dx_trial = np.full(n_total, np.nan)
                dy_trial = np.full(n_total, np.nan)

                if src_start < src_end:
                    dx_trial[dst_start:dst_end] = dx_filt[src_start:src_end]
                    dy_trial[dst_start:dst_end] = dy_filt[src_start:src_end]

                # Replace NaN with 0 before cumsum to avoid propagation
                dx_clean = np.nan_to_num(dx_trial)
                dy_clean = np.nan_to_num(dy_trial)

                cum_x = np.cumsum(dx_clean)
                cum_y = np.cumsum(dy_clean)

                # Zero-align: stimulus onset (index n_pre) becomes origin (0, 0)
                origin_x = cum_x[n_pre] if n_pre < n_total else 0.0
                origin_y = cum_y[n_pre] if n_pre < n_total else 0.0
                cum_x -= origin_x
                cum_y -= origin_y

                # Mark NaN regions back for valid-overlap tracking
                nan_mask = np.isnan(dx_trial) | np.isnan(dy_trial)
                cum_x[nan_mask] = np.nan
                cum_y[nan_mask] = np.nan

                trial_x_list.append(cum_x)
                trial_y_list.append(cum_y)

        time_axis_psth = np.arange(-n_pre, n_post) / resample_freq_hz

        if not speed_trials:
            return PSTHResult(
                event_type=event_type,
                time_axis=time_axis_psth,
                mean_speed=np.zeros(n_total),
                sem_speed=np.zeros(n_total),
                mean_angular_velocity=np.zeros(n_total),
                sem_angular_velocity=np.zeros(n_total),
                n_trials=0,
            )

        speed_arr = np.array(speed_trials)
        angvel_arr = np.array(angvel_trials)

        # nanmean/nanstd for pairwise deletion (NaN-padded edges)
        mean_speed = np.nanmean(speed_arr, axis=0)
        sem_speed = np.nanstd(speed_arr, axis=0) / np.sqrt(
            np.sum(~np.isnan(speed_arr), axis=0)
        )
        mean_angvel = np.nanmean(angvel_arr, axis=0)
        sem_angvel = np.nanstd(angvel_arr, axis=0) / np.sqrt(
            np.sum(~np.isnan(angvel_arr), axis=0)
        )

        # Replace any remaining NaN with 0
        mean_speed = np.nan_to_num(mean_speed)
        sem_speed = np.nan_to_num(sem_speed)
        mean_angvel = np.nan_to_num(mean_angvel)
        sem_angvel = np.nan_to_num(sem_angvel)

        n_valid_trials = int(np.max(np.sum(~np.isnan(speed_arr), axis=0)))

        # Assemble spatial trial matrices if available
        trial_x_arr = np.array(trial_x_list) if trial_x_list else None
        trial_y_arr = np.array(trial_y_list) if trial_y_list else None

        return PSTHResult(
            event_type=event_type,
            time_axis=time_axis_psth,
            mean_speed=mean_speed,
            sem_speed=sem_speed,
            mean_angular_velocity=mean_angvel,
            sem_angular_velocity=sem_angvel,
            n_trials=n_valid_trials,
            speed_trials=speed_arr,
            trial_x=trial_x_arr,
            trial_y=trial_y_arr,
        )

    # -------------------------------------------------------------------------
    # Calibration Metrics
    # -------------------------------------------------------------------------

    @staticmethod
    def _compute_energy_ratio(
        primary: np.ndarray,
        cross_a: np.ndarray,
        cross_b: np.ndarray,
    ) -> float:
        """Compute Signal-to-Noise Energy Ratio for decoupled axes.

        SNR = Var(primary) / (Var(cross_a) + Var(cross_b))

        A high ratio indicates strong decoupling: the primary axis carries
        the dominant signal energy while cross-talk axes contribute only
        noise-floor variance.

        Parameters
        ----------
        primary : np.ndarray
            Displacement of the axis under active stimulation.
        cross_a, cross_b : np.ndarray
            Displacement of the two orthogonal cross-talk axes.

        Returns
        -------
        float
            Energy ratio (dimensionless). Higher = better decoupling.
        """
        var_primary = float(np.var(primary))
        var_cross = float(np.var(cross_a) + np.var(cross_b))

        # Guard against division by zero on dead-still recordings
        if var_cross == 0.0:
            return float("inf") if var_primary > 0.0 else 0.0
        return var_primary / var_cross

    def _compute_session_snr_metrics(
        self,
        dx: np.ndarray,
        dy: np.ndarray,
        dz: np.ndarray,
    ) -> Dict[str, float]:
        """Compute per-axis SNR energy ratios for a session.

        Returns dict with keys: snr_x, snr_y, snr_z.
        Each value is the energy ratio for that axis treated as primary.
        """
        return {
            "snr_x": self._compute_energy_ratio(dx, dy, dz),
            "snr_y": self._compute_energy_ratio(dy, dx, dz),
            "snr_z": self._compute_energy_ratio(dz, dx, dy),
        }

    # -------------------------------------------------------------------------
    # Batch Processing
    # -------------------------------------------------------------------------

    def _handle_process_batch(self, cmd: Command) -> None:
        """Process all CSV pairs in input directory."""
        params = ProcessingParams(**cmd.params)
        self._params = params
        input_dir = Path(params.input_dir)

        if not input_dir.exists():
            self._send_error(cmd.request_id, f"Input directory not found: {input_dir}")
            return

        pairs = self._discover_csv_pairs(input_dir)
        if not pairs:
            self._send_error(
                cmd.request_id,
                f"No *kinematics*.csv files found in {input_dir}",
            )
            return

        logger.info("Found %d recording(s) in %s", len(pairs), input_dir)
        results: List[SessionResults] = []

        for i, (kin_path, evt_path) in enumerate(pairs):
            progress = (i + 1) / len(pairs)
            self._send_telemetry(
                TelemetryStatus.PROGRESS,
                cmd.request_id,
                progress=progress,
            )

            try:
                session_result = self._process_single_session(
                    kin_path, evt_path, params
                )
                results.append(session_result)
                logger.info(
                    "Processed %s: speed=%.2f mm/s, distance=%.1f mm",
                    session_result.session_id,
                    session_result.mean_speed,
                    session_result.total_distance,
                )
            except Exception as e:
                logger.exception("Error processing %s", kin_path.name)
                self._send_error(
                    cmd.request_id,
                    f"Error processing {kin_path.name}: {e}",
                )
                continue

        self._cached_results = results
        self._send_telemetry(
            TelemetryStatus.COMPLETE,
            cmd.request_id,
            data={"results": results},
        )

    def _handle_process_group_batch(self, cmd: Command) -> None:
        """Process multiple directories and tag each session with its group label.

        Expected ``cmd.params`` keys::

            {
                "processing": { ... ProcessingParams fields ... },
                "groups": {
                    "treatment": "/path/to/treatment",
                    "base_visual": "/path/to/visual_baseline",
                    "base_wind": "/path/to/wind_baseline",
                },
            }

        Each path is recursively scanned for CSV pairs.  The resulting
        ``SessionResults`` objects are tagged with their corresponding
        ``group_label`` before being returned to the telemetry queue.
        """
        processing_cfg = cmd.params.get("processing", {})
        params = ProcessingParams(**{
            k: v for k, v in processing_cfg.items()
            if k in ProcessingParams.__dataclass_fields__
        })
        self._params = params

        groups: Dict[str, str] = cmd.params.get("groups", {})
        if not groups:
            self._send_error(cmd.request_id, "No group paths provided.")
            return

        all_results: List[SessionResults] = []
        total_pairs = 0

        # First pass: discover all pairs across groups for progress tracking
        group_pairs: Dict[str, List[Tuple[Path, Optional[Path]]]] = {}
        for label, path_str in groups.items():
            group_dir = Path(path_str)
            if not group_dir.exists():
                logger.warning("Group directory not found, skipping: %s", group_dir)
                continue
            pairs = self._discover_csv_pairs(group_dir, recursive=True)
            group_pairs[label] = pairs
            total_pairs += len(pairs)

        if total_pairs == 0:
            self._send_error(cmd.request_id, "No *kinematics*.csv files found in any group.")
            return

        logger.info("Found %d recording(s) across %d group(s)", total_pairs, len(group_pairs))
        processed = 0

        for label, pairs in group_pairs.items():
            for kin_path, evt_path in pairs:
                processed += 1
                self._send_telemetry(
                    TelemetryStatus.PROGRESS,
                    cmd.request_id,
                    progress=processed / total_pairs,
                )
                try:
                    session_result = self._process_single_session(
                        kin_path, evt_path, params,
                    )
                    # Tag with group label via dataclass replace
                    session_result = SessionResults(
                        session_id=session_result.session_id,
                        kinematic_data=session_result.kinematic_data,
                        psth_results=session_result.psth_results,
                        freezing_episodes=session_result.freezing_episodes,
                        escape_bouts=session_result.escape_bouts,
                        mean_speed=session_result.mean_speed,
                        mean_angular_velocity=session_result.mean_angular_velocity,
                        total_distance=session_result.total_distance,
                        total_freezing_time_s=session_result.total_freezing_time_s,
                        group_label=label,
                    )
                    all_results.append(session_result)
                    logger.info(
                        "[%s] Processed %s: speed=%.2f mm/s",
                        label, session_result.session_id, session_result.mean_speed,
                    )
                except Exception as e:
                    logger.exception("Error processing %s", kin_path.name)
                    self._send_error(
                        cmd.request_id,
                        f"Error processing {kin_path.name}: {e}",
                    )
                    continue

        self._cached_results = all_results
        self._send_telemetry(
            TelemetryStatus.COMPLETE,
            cmd.request_id,
            data={"results": all_results},
        )

    def _process_single_session(
        self,
        kin_path: Path,
        evt_path: Optional[Path],
        params: ProcessingParams,
    ) -> SessionResults:
        """Process a single kinematics+events pair."""
        kin_df = self._load_kinematics(kin_path)
        session_id = kin_path.stem.replace("_kinematics", "").replace("-kinematics", "")

        kin_data, raw_dt = self._compute_kinematics(kin_df, params)
        kin_data = KinematicData(
            session_id=session_id,
            time=kin_data.time,
            speed=kin_data.speed,
            angular_velocity=kin_data.angular_velocity,
            trajectory_x=kin_data.trajectory_x,
            trajectory_y=kin_data.trajectory_y,
            raw_kinematics=kin_data.raw_kinematics,
            heading_angle=kin_data.heading_angle,
            steering_sine=kin_data.steering_sine,
            decoupled_dx=kin_data.decoupled_dx,
            decoupled_dy=kin_data.decoupled_dy,
            decoupled_dz=kin_data.decoupled_dz,
        )

        # Detect ethological states
        freezing = self._detect_freezing(
            kin_data.speed,
            kin_data.time,
            params.freezing_threshold_mm_s,
            params.freezing_min_duration_ms,
        )
        escapes = self._detect_escape_bouts(
            kin_data.speed,
            kin_data.time,
            params.escape_acceleration_threshold,
        )

        # PSTH if events exist
        psth_results: Dict[str, PSTHResult] = {}
        events_df: Optional[pd.DataFrame] = None
        if evt_path and evt_path.exists():
            events_df = self._load_events(evt_path)
            kin_data = KinematicData(
                session_id=kin_data.session_id,
                time=kin_data.time,
                speed=kin_data.speed,
                angular_velocity=kin_data.angular_velocity,
                trajectory_x=kin_data.trajectory_x,
                trajectory_y=kin_data.trajectory_y,
                raw_kinematics=kin_data.raw_kinematics,
                raw_events=events_df,
                heading_angle=kin_data.heading_angle,
                steering_sine=kin_data.steering_sine,
                decoupled_dx=kin_data.decoupled_dx,
                decoupled_dy=kin_data.decoupled_dy,
                decoupled_dz=kin_data.decoupled_dz,
            )

            for evt_type in events_df["event_type"].unique():
                onset_mask = events_df["event_type"] == evt_type
                raw_onset_times = events_df.loc[onset_mask, "sys_time"].values

                if len(raw_onset_times) == 0:
                    continue

                # Hardware Cross-Reference: Find real physical trigger
                onset_times = []
                kin_df = kin_data.raw_kinematics
                has_stim_state = "stim_state" in kin_df.columns

                for onset_t in raw_onset_times:
                    real_t = onset_t
                    if has_stim_state:
                        # Scan a 5-second window post software-trigger for actual hardware execution
                        mask = (kin_df["sys_time"] >= onset_t) & (kin_df["sys_time"] <= onset_t + 5.0) & (kin_df["stim_state"] > 0)
                        valid_rows = kin_df[mask]
                        if not valid_rows.empty:
                            real_t = float(valid_rows["sys_time"].iloc[0])
                    onset_times.append(real_t)

                onset_times = np.array(onset_times)

                psth = self._compute_psth_for_event_type(
                    event_type=str(evt_type),
                    event_times=onset_times,
                    time_axis=kin_data.time,
                    speed=kin_data.speed,
                    angular_velocity=kin_data.angular_velocity,
                    pre_s=params.psth_window_pre_s,
                    post_s=params.psth_window_post_s,
                    resample_freq_hz=params.resample_freq_hz,
                    dx_filt=kin_data.decoupled_dx,
                    dy_filt=kin_data.decoupled_dy,
                )
                psth_results[str(evt_type)] = psth

                # Compute escape latencies relative to stimulus onsets
                for onset_t in onset_times:
                    escapes = self._compute_escape_latency_from_stimulus(
                        escapes, onset_t
                    )

        # Aggregate metrics
        total_freezing_ms = sum(ep.duration_ms for ep in freezing)
        dt_resample = 1.0 / params.resample_freq_hz

        # Calibration SNR metrics (energy ratio of decoupled axes)
        snr_metrics = self._compute_session_snr_metrics(
            kin_data.decoupled_dx,
            kin_data.decoupled_dy,
            kin_data.decoupled_dz,
        )
        logger.info(
            "SNR metrics for %s — X: %.1f, Y: %.1f, Z: %.1f",
            session_id,
            snr_metrics["snr_x"],
            snr_metrics["snr_y"],
            snr_metrics["snr_z"],
        )

        return SessionResults(
            session_id=session_id,
            kinematic_data=kin_data,
            psth_results=psth_results,
            freezing_episodes=freezing,
            escape_bouts=escapes,
            mean_speed=float(np.mean(kin_data.speed)),
            mean_angular_velocity=float(np.mean(kin_data.angular_velocity)),
            total_distance=float(np.trapz(np.abs(kin_data.speed), dx=dt_resample)),
            total_freezing_time_s=total_freezing_ms / 1000.0,
        )

    def _handle_compute_psth(self, cmd: Command) -> None:
        """Recompute PSTH with updated parameters on cached data."""
        if self._params is None or not self._cached_results:
            self._send_error(
                cmd.request_id,
                "No data loaded. Run process_batch first.",
            )
            return

        # Update PSTH window params
        new_pre = cmd.params.get("psth_window_pre_s", self._params.psth_window_pre_s)
        new_post = cmd.params.get("psth_window_post_s", self._params.psth_window_post_s)

        updated_results: List[SessionResults] = []
        for res in self._cached_results:
            new_psth: Dict[str, PSTHResult] = {}
            if res.kinematic_data.raw_events is not None:
                events_df = res.kinematic_data.raw_events
                for evt_type in events_df["event_type"].unique():
                    onset_mask = events_df["event_type"] == evt_type
                    onset_times = events_df.loc[onset_mask, "sys_time"].values
                    if len(onset_times) == 0:
                        continue

                    psth = self._compute_psth_for_event_type(
                        event_type=str(evt_type),
                        event_times=onset_times,
                        time_axis=res.kinematic_data.time,
                        speed=res.kinematic_data.speed,
                        angular_velocity=res.kinematic_data.angular_velocity,
                        pre_s=new_pre,
                        post_s=new_post,
                        resample_freq_hz=self._params.resample_freq_hz,
                        dx_filt=res.kinematic_data.decoupled_dx,
                        dy_filt=res.kinematic_data.decoupled_dy,
                    )
                    new_psth[str(evt_type)] = psth

            updated = SessionResults(
                session_id=res.session_id,
                kinematic_data=res.kinematic_data,
                psth_results=new_psth,
                freezing_episodes=res.freezing_episodes,
                escape_bouts=res.escape_bouts,
                mean_speed=res.mean_speed,
                mean_angular_velocity=res.mean_angular_velocity,
                total_distance=res.total_distance,
                total_freezing_time_s=res.total_freezing_time_s,
            )
            updated_results.append(updated)

        self._cached_results = updated_results
        self._send_telemetry(
            TelemetryStatus.COMPLETE,
            cmd.request_id,
            data={"results": updated_results},
        )

    def _handle_export_results(self, cmd: Command) -> None:
        """Export results to CSV."""
        results: List[SessionResults] = cmd.params.get("results", [])
        if not results:
            results = self._cached_results
        if not results:
            self._send_error(cmd.request_id, "No results to export.")
            return

        output_dir = Path(cmd.params.get("output_dir", "data/output"))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build summary statistics CSV
        rows: List[Dict[str, Any]] = []
        for res in results:
            kd = res.kinematic_data
            snr = self._compute_session_snr_metrics(
                kd.decoupled_dx, kd.decoupled_dy, kd.decoupled_dz,
            )
            rows.append({
                "session_id": res.session_id,
                "mean_speed_mm_s": round(res.mean_speed, 4),
                "mean_angular_velocity": round(res.mean_angular_velocity, 4),
                "total_distance_mm": round(res.total_distance, 2),
                "total_freezing_time_s": round(res.total_freezing_time_s, 3),
                "n_freezing_episodes": len(res.freezing_episodes),
                "n_escape_bouts": len(res.escape_bouts),
                "mean_escape_latency_s": round(
                    float(np.mean([
                        b.latency_from_stimulus
                        for b in res.escape_bouts
                        if b.latency_from_stimulus > 0
                    ])) if any(b.latency_from_stimulus > 0 for b in res.escape_bouts) else 0.0,
                    4,
                ),
                "snr_x_energy_ratio": round(snr["snr_x"], 2),
                "snr_y_energy_ratio": round(snr["snr_y"], 2),
                "snr_z_energy_ratio": round(snr["snr_z"], 2),
            })

        summary_df = pd.DataFrame(rows)
        summary_path = output_dir / "summary_statistics.csv"
        summary_df.to_csv(summary_path, index=False)

        # Export per-session freezing episodes
        for res in results:
            if res.freezing_episodes:
                freeze_df = pd.DataFrame([
                    {
                        "session_id": res.session_id,
                        "start_time": ep.start_time,
                        "end_time": ep.end_time,
                        "duration_ms": ep.duration_ms,
                    }
                    for ep in res.freezing_episodes
                ])
                freeze_path = output_dir / f"{res.session_id}_freezing.csv"
                freeze_df.to_csv(freeze_path, index=False)

        logger.info("Exported results to %s", output_dir)
        self._send_telemetry(
            TelemetryStatus.COMPLETE,
            cmd.request_id,
            data={"summary_path": str(summary_path)},
        )
