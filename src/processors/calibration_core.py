"""CalibrationProcessor: Standalone kinematics backend for hardware validation.

Pure physical kinematics only — no ethological logic (Freezing, Escape, PSTH).
Replicates core math from DataProcessor for independent calibration workflow.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import CubicSpline


@dataclass
class CalibrationResult:
    """Processed calibration data for a single recording."""
    session_id: str
    time: np.ndarray              # Uniform time axis (s)
    dx_raw: np.ndarray            # Raw lateral displacement
    dy_raw: np.ndarray            # Raw forward/back displacement
    dz_raw: np.ndarray            # Raw rotational displacement
    dx_filtered: np.ndarray       # Filtered lateral
    dy_filtered: np.ndarray       # Filtered forward/back
    dz_filtered: np.ndarray       # Filtered rotational
    speed: np.ndarray             # Instantaneous speed (mm/s)
    angular_velocity: np.ndarray  # Instantaneous angular velocity
    cumulative_yaw: np.ndarray    # Cumulative sum of dz
    raw_dt: float                 # Median raw sample interval
    resample_freq_hz: float       # Resampling frequency


class CalibrationProcessor:
    """Standalone kinematics processor for hardware calibration.

    Contains ONLY pure physical math:
    - Uniform time delta computation
    - Cubic spline resampling
    - Zero-phase Butterworth low-pass filter
    - Instantaneous speed (V = sqrt(dx² + dy²) / dt)
    - Cumulative yaw (cumsum of dz)

    No ethological logic. No multiprocessing. No queues.
    """

    def __init__(
        self,
        filter_cutoff_hz: float = 10.0,
        filter_order: int = 4,
        resample_freq_hz: float = 100.0,
    ) -> None:
        self.filter_cutoff_hz: float = filter_cutoff_hz
        self.filter_order: int = filter_order
        self.resample_freq_hz: float = resample_freq_hz

    def load_csv(self, filepath: Path) -> pd.DataFrame:
        """Load and validate a raw kinematics CSV.

        Parameters
        ----------
        filepath : Path
            Path to CSV with columns: sys_time, dx, dy, dz.

        Returns
        -------
        pd.DataFrame
            Cleaned dataframe sorted by sys_time.

        Raises
        ------
        ValueError
            If required columns are missing.
        """
        df = pd.read_csv(filepath)
        required = {"sys_time", "dx", "dy", "dz"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"Missing columns in {filepath.name}: {missing}")
        return (
            df[["sys_time", "dx", "dy", "dz"]]
            .dropna()
            .sort_values("sys_time")
            .reset_index(drop=True)
        )

    def compute_uniform_dt(self, timestamps: np.ndarray) -> float:
        """Compute median time delta from timestamp array.

        Parameters
        ----------
        timestamps : np.ndarray
            Array of timestamps.

        Returns
        -------
        float
            Median dt in same units as timestamps.

        Raises
        ------
        ValueError
            If fewer than 2 timestamps or dt is non-positive.
        """
        diffs = np.diff(timestamps)
        if len(diffs) == 0:
            raise ValueError("Timestamp array has fewer than 2 elements")
        median_dt = float(np.median(diffs))
        if median_dt <= 0:
            raise ValueError(f"Non-positive median dt: {median_dt}")
        return median_dt

    def resample_to_uniform(
        self,
        timestamps: np.ndarray,
        values: np.ndarray,
        target_freq_hz: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resample unevenly sampled data to fixed frequency.

        Uses cubic spline interpolation with natural boundary conditions.

        Parameters
        ----------
        timestamps : np.ndarray
            Original timestamps.
        values : np.ndarray
            Values at original timestamps.
        target_freq_hz : float
            Target resampling frequency.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (uniform_time, resampled_values).
        """
        if len(timestamps) < 4:
            raise ValueError("Cubic spline requires at least 4 data points")

        t_start = timestamps[0]
        t_end = timestamps[-1]
        dt_target = 1.0 / target_freq_hz
        t_uniform = np.arange(t_start, t_end, dt_target)

        cs = CubicSpline(timestamps, values, bc_type="natural")
        values_uniform = cs(t_uniform)

        return t_uniform, values_uniform

    def apply_lowpass_filter(
        self,
        data: np.ndarray,
        cutoff_hz: float,
        sample_rate_hz: float,
        order: int = 4,
    ) -> np.ndarray:
        """Zero-phase low-pass Butterworth filter.

        Uses filtfilt for zero-phase distortion. Reflect-pads signal
        edges to reduce boundary artifacts.

        Parameters
        ----------
        data : np.ndarray
            Input signal.
        cutoff_hz : float
            Filter cutoff frequency.
        sample_rate_hz : float
            Sample rate of the signal.
        order : int
            Filter order.

        Returns
        -------
        np.ndarray
            Filtered signal.
        """
        nyquist = sample_rate_hz / 2.0
        normalized_cutoff = cutoff_hz / nyquist

        if normalized_cutoff >= 1.0:
            return data
        if normalized_cutoff <= 0.0:
            raise ValueError(f"Invalid normalized cutoff: {normalized_cutoff}")

        b, a = signal.butter(order, normalized_cutoff, btype="low")

        # Reflect-pad to reduce edge effects
        pad_len = min(3 * max(len(b), len(a)), len(data) // 2)
        if pad_len > 0:
            padded = np.pad(data, pad_len, mode="reflect")
            filtered = signal.filtfilt(b, a, padded)
            return filtered[pad_len:-pad_len] if pad_len > 0 else filtered
        return signal.filtfilt(b, a, data)

    def compute_speed(
        self,
        dx: np.ndarray,
        dy: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Compute instantaneous speed.

        V = sqrt(dx² + dy²) / dt

        Parameters
        ----------
        dx, dy : np.ndarray
            Filtered displacement components.
        dt : float
            Time step (1 / sample_rate).

        Returns
        -------
        np.ndarray
            Speed in same displacement units per second.
        """
        return np.sqrt(dx**2 + dy**2) / dt

    def compute_cumulative_yaw(self, dz: np.ndarray) -> np.ndarray:
        """Compute cumulative yaw via cumulative sum.

        Parameters
        ----------
        dz : np.ndarray
            Rotational displacement array.

        Returns
        -------
        np.ndarray
            Cumulative yaw.
        """
        return np.cumsum(dz)

    def process(
        self,
        filepath: Path,
        session_id: Optional[str] = None,
    ) -> CalibrationResult:
        """Full calibration pipeline for a single CSV.

        Parameters
        ----------
        filepath : Path
            Path to raw kinematics CSV.
        session_id : Optional[str]
            Identifier. Defaults to filename stem.

        Returns
        -------
        CalibrationResult
            Processed calibration data.
        """
        # Load
        df = self.load_csv(filepath)
        if session_id is None:
            session_id = filepath.stem.replace("_kinematics", "").replace("-kinematics", "")

        timestamps = df["sys_time"].values.astype(np.float64)
        dx_raw = df["dx"].values.astype(np.float64)
        dy_raw = df["dy"].values.astype(np.float64)
        dz_raw = df["dz"].values.astype(np.float64)

        # Compute raw dt
        raw_dt = self.compute_uniform_dt(timestamps)

        # Resample to uniform
        t_uniform, dx_uniform = self.resample_to_uniform(timestamps, dx_raw, self.resample_freq_hz)
        _, dy_uniform = self.resample_to_uniform(timestamps, dy_raw, self.resample_freq_hz)
        _, dz_uniform = self.resample_to_uniform(timestamps, dz_raw, self.resample_freq_hz)

        # Low-pass filter
        dx_filtered = self.apply_lowpass_filter(dx_uniform, self.filter_cutoff_hz, self.resample_freq_hz, self.filter_order)
        dy_filtered = self.apply_lowpass_filter(dy_uniform, self.filter_cutoff_hz, self.resample_freq_hz, self.filter_order)
        dz_filtered = self.apply_lowpass_filter(dz_uniform, self.filter_cutoff_hz, self.resample_freq_hz, self.filter_order)

        # Kinematics
        dt_resample = 1.0 / self.resample_freq_hz
        speed = self.compute_speed(dx_filtered, dy_filtered, dt_resample)
        angular_velocity = dz_filtered / dt_resample
        cumulative_yaw = self.compute_cumulative_yaw(dz_filtered)

        return CalibrationResult(
            session_id=session_id,
            time=t_uniform,
            dx_raw=dx_raw,
            dy_raw=dy_raw,
            dz_raw=dz_raw,
            dx_filtered=dx_filtered,
            dy_filtered=dy_filtered,
            dz_filtered=dz_filtered,
            speed=speed,
            angular_velocity=angular_velocity,
            cumulative_yaw=cumulative_yaw,
            raw_dt=raw_dt,
            resample_freq_hz=self.resample_freq_hz,
        )
