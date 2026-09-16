"""Reaction-time / distance pure-function checks (hand-computed trapezoid)."""

import numpy as np

from cercus.core.kinematics.distance import compute_reaction_and_distance


def test_constant_speed_hand_computed():
    t = np.arange(-1000, 2000, 10.0)
    v = np.where((t >= 0) & (t <= 1000), 100.0, 0.0)
    r = compute_reaction_and_distance(t, v, 0.0, 1000.0, -373.0)
    assert r["reaction_time_ms"] == 373.0          # onset − wind anchor
    assert abs(r["distance_mm"] - 100.0) < 1e-6    # 100 mm/s × 1 s
    assert abs(r["distance_500ms_mm"] - 50.0) < 1e-6


def test_negative_rt_for_prewind_onset():
    t = np.arange(-1000, 500, 10.0)
    v = np.where((t >= -500) & (t <= -200), 200.0, 0.0)
    r = compute_reaction_and_distance(t, v, -500.0, -200.0, -373.0)
    assert r["reaction_time_ms"] == -127.0         # started 127 ms before wind


def test_nan_propagation():
    t = np.arange(0, 1000, 10.0)
    r = compute_reaction_and_distance(t, np.zeros_like(t), np.nan, np.nan, -373.0)
    assert all(np.isnan(x) for x in r.values())
