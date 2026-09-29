"""Polar directions are relative to each trial's stimulus side."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from cercus.core.kinematics import trajectory_integration
from cercus.visualization import polar


@pytest.mark.parametrize("direction, expected_angle, bin_index", [
    ("contra", -90, 9), ("ipsi", 90, 27),
])
def test_polar_normalizes_wind_sides(monkeypatch, direction, expected_angle, bin_index):
    x_for_left = 1.0 if direction == "contra" else -1.0
    rows = []
    for trial_id, (wind_side, raw_x) in enumerate((("left", x_for_left), ("right", -x_for_left))):
        for t_rel in (0.0, 1.0):
            rows.append({
                "global_trial_id": trial_id,
                "t_rel": t_rel,
                "type": "baseline_wind",
                "wind_dir": wind_side,
                "screen_side": "right" if wind_side == "left" else "left",
                "response_type": "Escape",
                "interval_onset_ms": 0.0,
                "interval_offset_ms": 1.0,
                "dx": raw_x,
            })

    def trajectory(grp, *_args, **kwargs):
        assert kwargs["context"] == "polar"
        return np.array([0.0, grp["dx"].iloc[0]]), np.array([0.0, 0.0])

    monkeypatch.setattr(polar, "compute_trajectory_masks", trajectory)
    fig = polar.plot_population_polar_histogram(pd.DataFrame(rows))
    try:
        ax = fig.axes[0]
        legend = ax.get_legend().get_texts()[0].get_text()
        assert "Escape (n = 2)" in legend
        assert f"μ = {expected_angle}°, R = 1.00" in legend
        assert round(ax.patches[bin_index].get_height()) == 100
        labels = [tick.get_text() for tick in ax.get_xticklabels()]
        assert "−90°\nContra" in labels
        assert "90°\nIpsi" in labels
    finally:
        plt.close(fig)


def test_polar_skips_unknown_stimulus_side(monkeypatch):
    df = pd.DataFrame({
        "global_trial_id": [0, 0],
        "t_rel": [0.0, 1.0],
        "type": ["baseline_wind", "baseline_wind"],
        "response_type": ["Escape", "Escape"],
        "interval_onset_ms": [0.0, 0.0],
        "interval_offset_ms": [1.0, 1.0],
    })
    monkeypatch.setattr(polar, "compute_trajectory_masks", lambda *_a, **_kw: (np.array([0., 1.]), np.array([0., 0.])))
    fig = polar.plot_population_polar_histogram(df)
    try:
        assert "No valid escape trials" in fig.axes[0].texts[0].get_text()
    finally:
        plt.close(fig)


@pytest.mark.parametrize("direction, expected_angle, bin_index", [
    ("contra", -90, 9), ("ipsi", 90, 27),
])
def test_pre_movement_normalizes_wind_sides(monkeypatch, direction, expected_angle, bin_index):
    x_for_left = 1.0 if direction == "contra" else -1.0
    rows = []
    for trial_id, (wind_side, raw_x) in enumerate((("left", x_for_left), ("right", -x_for_left))):
        for t_rel in (0.0, 1.0, 2.0, 3.0):
            rows.append({
                "global_trial_id": trial_id,
                "t_rel": t_rel,
                "type": "baseline_wind",
                "wind_dir": wind_side,
                "screen_side": "right" if wind_side == "left" else "left",
                "response_type": "PreWalk",
                "interval_onset_ms": 3.0,
                "dx": raw_x,
            })

    def trajectory(grp, *_args, **kwargs):
        assert kwargs["context"] == "polar"
        return np.array([0.0, grp["dx"].iloc[0]]), np.array([0.0, 0.0])

    monkeypatch.setattr(trajectory_integration, "body_to_traj", trajectory)
    fig = polar.plot_population_pre_movement_prewalk(pd.DataFrame(rows))
    try:
        ax = fig.axes[0]
        assert len(ax.patches) == 36
        assert round(ax.patches[bin_index].get_height()) == 100
        legend = fig.legends[0].get_texts()[0].get_text()
        assert "PreWalk (n = 2)" in legend
        assert f"μ = {expected_angle}°, R = 1.00" in legend
        labels = [tick.get_text() for tick in ax.get_xticklabels()]
        assert "−90°\nContra" in labels
        assert "90°\nIpsi" in labels
    finally:
        plt.close(fig)
