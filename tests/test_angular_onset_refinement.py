"""
Tests for angular-velocity escape-onset refinement
(``escape.use_angular_onset_refinement``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cercus.core.kinematics.latency import (
    compute_escape_latency,
    get_angular_onset_idx,
)
from pipeline.classifier import classify_trial

_FPS = 200.0  # 5 ms/frame


def _synthetic() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """200 fps trial: linear 10 mm/s onset at 100 ms, angular spike 40 ms earlier."""
    dt = 1000.0 / _FPS
    t = np.arange(-500.0, 500.0, dt)  # 200 samples, t[120] = 100 ms
    n = len(t)
    speed = np.full(n, 2.0)
    i_coarse = 120  # 10 mm/s crossing at 100 ms
    i_ang = i_coarse - 8  # 40 ms earlier
    peak = 140
    speed[i_coarse:peak] = np.linspace(15.0, 903.0, peak - i_coarse)
    speed[peak:] = np.linspace(903.0, 2.0, n - peak)

    omega = np.zeros(n)
    spike = 18.0 * np.exp(-((np.arange(30) - 8) ** 2) / (2 * 4.0 ** 2))
    omega[i_ang:i_ang + 30] = spike
    dz = 0.01 * omega
    return t, speed, omega, dz, i_coarse, i_ang


def test_get_angular_onset_idx_finds_earlier_zero():
    _, _, omega, dz, i_coarse, i_ang = _synthetic()
    idx = get_angular_onset_idx(omega, dz, i_coarse, _FPS)
    # left bracket = last zero frame before the spike, i.e. one before i_ang
    assert idx == i_ang - 1
    assert idx < i_coarse


def test_latency_flag_off_equals_coarse():
    t, speed, omega, dz, i_coarse, _ = _synthetic()
    res = compute_escape_latency(
        t, speed, angular_velocity=omega, dz=dz,
        use_angular_onset_refinement=False,
    )
    assert res["latency_ms"] == pytest.approx(t[i_coarse])
    assert res["latency_coarse_ms"] == pytest.approx(t[i_coarse])


def test_latency_flag_on_refines_earlier():
    t, speed, omega, dz, i_coarse, i_ang = _synthetic()
    res = compute_escape_latency(
        t, speed, angular_velocity=omega, dz=dz,
        use_angular_onset_refinement=True,
    )
    assert res["latency_ms"] == pytest.approx(t[i_ang - 1])
    assert res["latency_ms"] < res["latency_coarse_ms"]
    # angular zero is 45 ms (9 frames) before the 10 mm/s point
    assert t[i_coarse] - res["latency_ms"] == pytest.approx(45.0)


def test_classifier_keeps_interval_and_prewalk_coarse():
    t, speed, omega, dz, i_coarse, _ = _synthetic()
    df = pd.DataFrame({
        "t_rel": t,
        "speed": speed,
        "angular_velocity": omega,
        "dz": dz,
        "target_ttc_ms": 0.0,
        "type": "looming_wind",
        "global_trial_id": 1,
    })

    off = classify_trial(df, use_angular_onset_refinement=False)
    on = classify_trial(df, use_angular_onset_refinement=True)

    assert off["response_type"] == "Escape"
    assert on["response_type"] == "Escape"
    assert on["latency_ms"] < off["latency_ms"]
    # Interval onset (PreWalk anchor) must stay on the coarse 10 mm/s point
    assert on["interval_onset_ms"] == pytest.approx(off["interval_onset_ms"])
    assert on["interval_onset_ms"] == pytest.approx(t[i_coarse])
