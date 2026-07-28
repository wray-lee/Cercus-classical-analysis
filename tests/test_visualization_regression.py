"""
Regression tests for visualization module.
Generates plots with Agg backend and compares file hashes.

Usage:
    pytest tests/test_visualization_regression.py -v
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # must be before pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from cercus.visualization import (
    plot_behavior_probability,
    plot_escape_angle_distribution,
    plot_habituation_curve,
    plot_population_behavior_probability,
    plot_population_habituation,
    plot_population_polar_histogram,
    plot_population_spaghetti_kinetics,
    plot_population_speed_kinetics,
    plot_population_vmax_gmm,
    plot_population_vmax_response,
    plot_prewalk_stillness,
    plot_single_trial_kinetics,
    plot_spaghetti_kinetics,
    plot_spaghetti_kinetics_heatmap,
    plot_speed_kinetics,
    plot_trajectory_overlay,
    plot_trial_stacked_heatmap,
    plot_vmax_distribution,
)

# Directory for baseline hashes
_HASH_DIR = Path(__file__).parent / "baselines"
_HASH_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    """Create a minimal synthetic DataFrame for plot regression testing."""
    rng = np.random.default_rng(42)
    n_trials = 10
    n_frames = 200

    rows = []
    for tid in range(n_trials):
        t_rel = np.linspace(-1000, 500, n_frames) + rng.uniform(-5, 5, n_frames)
        speed = np.maximum(
            0,
            np.sin(np.linspace(0, np.pi, n_frames)) * rng.uniform(20, 80)
            + rng.normal(0, 3, n_frames),
        )
        # Give some trials a proper escape burst
        if tid < 5:
            speed += np.exp(-np.linspace(0, 3, n_frames)) * 60

        # Create valid onset/offset for escape-ish trials
        onset_idx = 100 if tid < 5 else np.nan
        offset_idx = 150 if tid < 5 else np.nan

        for f in range(n_frames):
            rows.append({
                "global_trial_id": tid,
                "global_trial_index": tid,
                "subject_id": f"subj_{tid % 3}",
                "type": "looming_wind" if tid < 7 else "baseline_visual_test",
                "t_rel": t_rel[f],
                "speed": speed[f],
                "angular_velocity": rng.normal(0, 0.5, 1)[0],
                "response_type": "Escape" if tid < 4 else "PreWalk" if tid < 7 else "NoResponse",
                "v_max": float(np.max(speed)),
                "latency_ms": 50.0 if tid < 7 else np.nan,
                "interval_onset_ms": float(t_rel[100]) if tid < 7 else np.nan,
                "interval_offset_ms": float(t_rel[150]) if tid < 7 else np.nan,
                "screen_side": "left" if tid % 2 == 0 else "right",
                "stim_state": 1 if tid < 7 else 0,
                "dx": rng.normal(0, 0.5, 1)[0],
                "dy": rng.normal(0, 0.5, 1)[0],
                "dz": rng.normal(0, 0.1, 1)[0],
            })

    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def _close_figs():
    """Ensure all figures are closed after each test."""
    yield
    plt.close("all")


# ══════════════════════════════════════════════════════════════════════
# Hash-based regression helpers
# ══════════════════════════════════════════════════════════════════════


def _fig_hash(fig: plt.Figure) -> str:
    """Compute SHA-256 of the rendered figure."""
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    return hashlib.sha256(buf).hexdigest()


def _assert_regression(
    fig: plt.Figure,
    name: str,
) -> None:
    """Assert pixel-identical output using hash comparison."""
    h = _fig_hash(fig)
    hash_file = _HASH_DIR / f"{name}.sha256"

    if hash_file.exists():
        expected = hash_file.read_text().strip()
        assert h == expected, (
            f"Figure hash mismatch for {name}!\n"
            f"  Expected: {expected}\n"
            f"  Got:      {h}\n"
            "Update baseline: rm tests/baselines/*.sha256 && pytest --update-baselines"
        )
    else:
        # First run: save baseline
        hash_file.write_text(h)


# ══════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════


class TestTrajectoryPlots:
    """Regression tests for trajectory overlay plots."""

    def test_trajectory_overlay(self, sample_df):
        fig = plot_trajectory_overlay(sample_df)
        _assert_regression(fig, "trajectory_overlay")

    def test_global_trajectory_overlay(self, sample_df):
        from cercus.visualization import plot_global_trajectory_overlay_fixed
        fig = plot_global_trajectory_overlay_fixed(sample_df)
        _assert_regression(fig, "global_trajectory_overlay")


class TestKineticsPlots:
    """Regression tests for kinetics plots."""

    def test_speed_kinetics(self, sample_df):
        fig = plot_speed_kinetics(sample_df)
        _assert_regression(fig, "speed_kinetics")

    def test_population_speed_kinetics(self, sample_df):
        fig = plot_population_speed_kinetics(sample_df)
        _assert_regression(fig, "population_speed_kinetics")

    def test_population_spaghetti_kinetics(self, sample_df):
        fig = plot_population_spaghetti_kinetics(sample_df)
        _assert_regression(fig, "population_spaghetti_kinetics")

    def test_spaghetti_kinetics(self, sample_df):
        fig = plot_spaghetti_kinetics(sample_df)
        _assert_regression(fig, "spaghetti_kinetics")

    def test_single_trial_kinetics(self, sample_df):
        trial = sample_df[sample_df["global_trial_id"] == 0].sort_values("t_rel")
        fig = plot_single_trial_kinetics(
            trial, latency_ms=50.0, v_max=80.0,
            global_trial_index=0, response_type="Escape",
        )
        _assert_regression(fig, "single_trial_kinetics")


class TestHeatmapPlots:
    """Regression tests for heatmap plots."""

    def test_spaghetti_kinetics_heatmap(self, sample_df):
        fig = plot_spaghetti_kinetics_heatmap(sample_df)
        _assert_regression(fig, "spaghetti_kinetics_heatmap")

    def test_trial_stacked_heatmap(self, sample_df):
        fig = plot_trial_stacked_heatmap(sample_df)
        _assert_regression(fig, "trial_stacked_heatmap")


class TestVmaxPlots:
    """Regression tests for Vmax distribution plots."""

    def test_vmax_distribution(self, sample_df):
        fig = plot_vmax_distribution(sample_df)
        _assert_regression(fig, "vmax_distribution")

    def test_population_vmax_gmm(self, sample_df):
        fig = plot_population_vmax_gmm(sample_df)
        _assert_regression(fig, "population_vmax_gmm")

    def test_population_vmax_response(self, sample_df):
        fig = plot_population_vmax_response(sample_df)
        _assert_regression(fig, "population_vmax_response")


class TestBehaviorPlots:
    """Regression tests for behavior probability plots."""

    def test_behavior_probability(self, sample_df):
        fig = plot_behavior_probability(sample_df)
        _assert_regression(fig, "behavior_probability")

    def test_habituation_curve(self, sample_df):
        fig = plot_habituation_curve(sample_df)
        _assert_regression(fig, "habituation_curve")

    def test_population_habituation(self, sample_df):
        fig = plot_population_habituation(sample_df)
        _assert_regression(fig, "population_habituation")

    def test_population_behavior_probability(self, sample_df):
        fig = plot_population_behavior_probability(sample_df)
        _assert_regression(fig, "population_behavior_probability")

    def test_prewalk_stillness(self, sample_df):
        fig = plot_prewalk_stillness(sample_df)
        _assert_regression(fig, "prewalk_stillness")


class TestPolarPlots:
    """Regression tests for polar / angle plots."""

    def test_escape_angle_distribution(self, sample_df):
        fig = plot_escape_angle_distribution(sample_df)
        _assert_regression(fig, "escape_angle_distribution")

    def test_population_polar_histogram(self, sample_df):
        fig = plot_population_polar_histogram(sample_df)
        _assert_regression(fig, "population_polar_histogram")
