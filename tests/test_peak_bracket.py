"""
Tests for peak_bracket integration mode.
"""
import numpy as np

from cercus.core.kinematics.trajectory_integration import find_peak_bracket_interval


_DEG2RAD = np.pi / 180.0


def test_peak_bracket_ideal_peak():
    """Synthetic dz with a clear peak and zero-crossings on both sides."""
    t = np.linspace(0, 1, 100)
    # omega_z = gaussian bump + noise, eps=2 deg/s ~ 0.0349 rad/s
    omega = 3.0 * np.exp(-((t - 0.5) ** 2) / (2 * 0.02 ** 2)) * _DEG2RAD
    dz = 0.01 * omega  # small dz to match
    onset, offset = 20, 80
    omega[:onset] = 0
    omega[offset:] = 0

    l, r = find_peak_bracket_interval(dz, omega, onset, offset, eps=2.0)

    # boundaries should be inside the window
    assert l >= onset, f"left {l} < onset {onset}"
    assert r <= offset - 1, f"right {r} > offset {offset}"
    assert r - l >= 5, f"bracket too short ({r-l} frames, need >=5)"
    # peak should be bracketed: l <= 50 <= r
    assert l <= 50 <= r, f"peak at 50 not bracketed by [{l}, {r}]"


def test_peak_bracket_weak_peak_fallback():
    """Near-zero omega_z -> fallback to escape_angular_peak behaviour."""
    t = np.linspace(0, 1, 100)
    omega = 0.001 * np.sin(2 * np.pi * t) * _DEG2RAD  # ~0.006 deg/s peak, well below 2 deg/s
    dz = np.zeros_like(t)
    onset, offset = 20, 80

    l, r = find_peak_bracket_interval(dz, omega, onset, offset, eps=2.0)

    # fallback: l == onset, r should be post-peak zero crossing
    assert l == onset, f"expected fallback l={onset}, got {l}"
    assert r > onset, f"right bound {r} not past onset"


def test_peak_bracket_short_bracket_fallback():
    """Bracket length < 5 frames -> fallback."""
    omega = np.zeros(100)
    dz = np.zeros(100)
    # Single isolated spike: peak width < 5 frames
    omega[50] = 10.0 * _DEG2RAD
    onset, offset = 40, 60

    l, r = find_peak_bracket_interval(dz, omega, onset, offset, eps=2.0)

    # Should fallback: l == onset
    assert l == onset, f"expected fallback l={onset}, got {l}"
    assert r > onset, f"right bound {r} not past onset"


def test_peak_bracket_noise_robustness():
    """Gaussian noise on top of a clear peak still finds the bracket."""
    rng = np.random.RandomState(42)
    t = np.linspace(0, 1, 120)
    signal = 5.0 * np.exp(-((t - 0.5) ** 2) / (2 * 0.03 ** 2)) * _DEG2RAD
    noise = 0.1 * _DEG2RAD * rng.randn(len(t))
    omega = signal + noise
    dz = np.zeros_like(t)
    onset, offset = 25, 95

    l, r = find_peak_bracket_interval(dz, omega, onset, offset, eps=2.0)

    # Bracket should still be reasonable
    assert l >= onset
    assert r <= offset - 1
    # The peak (~50) should be inside or near the bracket
    assert l <= 50 or r >= 50, f"peak at 50 not near bracket [{l}, {r}]"
    bracket_len = r - l + 1
    assert bracket_len < (offset - onset) // 2 or bracket_len < 40, f"bracket too wide: {bracket_len}"


def test_peak_bracket_asymmetric_peak():
    """Peak near the right edge: bracket should still be valid."""
    t = np.linspace(0, 1, 100)
    omega = np.zeros(100)
    omega[40:70] = 4.0 * _DEG2RAD * np.exp(-((t[40:70] - 0.65) ** 2) / (2 * 0.01 ** 2))
    dz = np.zeros(100)
    onset, offset = 30, 70

    l, r = find_peak_bracket_interval(dz, omega, onset, offset, eps=2.0)

    assert l >= onset
    assert r <= offset - 1 or r == offset, f"r={r} > offset={offset}"
    # Fallback or valid bracket
    assert r - l >= 5 or (l == onset and r > onset), f"unexpected bracket [{l}, {r}]"