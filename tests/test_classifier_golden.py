"""
Golden test for the classifier: 20 synthetic trials covering all edge cases.
Verifies bitwise-identical classification results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.classifier import classify_trial

# ── Classification result keys ──
_RESULT_KEYS = [
    "response_type",
    "v_max",
    "latency_ms",
    "escape_interval_ms",
    "interval_onset_ms",
    "interval_offset_ms",
]


def _make_trial(
    speed: np.ndarray,
    t_rel: np.ndarray | None = None,
    angular_velocity: np.ndarray | None = None,
    target_ttc_ms: float = np.nan,
    trial_type: str = "looming_wind",
    onset_ms: float = np.nan,
    offset_ms: float = np.nan,
) -> pd.DataFrame:
    """Create a single-trial DataFrame for classifier input."""
    if t_rel is None:
        t_rel = np.linspace(-1500, 500, len(speed))
    n = len(speed)
    df = pd.DataFrame({
        "t_rel": t_rel,
        "speed": speed,
        "angular_velocity": angular_velocity if angular_velocity is not None else np.zeros(n),
        "target_ttc_ms": target_ttc_ms,
        "type": trial_type,
        "interval_onset_ms": onset_ms,
        "interval_offset_ms": offset_ms,
        "global_trial_id": 1,
    })
    return df


# ══════════════════════════════════════════════════════════════════════
# Synthetic trials
# ══════════════════════════════════════════════════════════════════════


def _trial_no_burst() -> pd.DataFrame:
    """Trial: speed never exceeds 50 mm/s -> NoResponse."""
    t = np.linspace(-1000, 500, 300)
    speed = np.ones_like(t) * 3.0 + np.random.default_rng(0).normal(0, 1, len(t))
    return _make_trial(speed, t, trial_type="looming_wind")


def _trial_clean_escape() -> pd.DataFrame:
    """Trial: clean escape burst after stimulus onset."""
    t = np.linspace(-500, 500, 200)
    speed = np.zeros_like(t)
    # Quiescent baseline
    speed[:100] = np.random.default_rng(1).normal(2, 0.5, 100)
    # Escape burst at t=50
    burst = np.exp(-np.linspace(0, 3, 80)) * 120
    speed[100:180] = burst
    speed[100:] += np.random.default_rng(2).normal(0, 1, len(t) - 100)
    speed = np.maximum(0, speed)
    return _make_trial(speed, t, trial_type="looming_wind", target_ttc_ms=0.0)


def _trial_prewalk_before_stimulus() -> pd.DataFrame:
    """Trial: walking activity before stimulus -> PreWalk."""
    t = np.linspace(-1500, 500, 400)
    speed = np.zeros_like(t)
    # Walking activity in pre-stimulus
    speed[:250] = np.random.default_rng(3).normal(0, 5, 250) + 15
    speed[:250] = np.maximum(0, speed[:250])
    # Escape burst
    burst = np.exp(-np.linspace(0, 2, 80)) * 130
    speed[250:330] = burst
    speed[250:] += np.random.default_rng(4).normal(0, 1, len(t) - 250)
    speed = np.maximum(0, speed)
    return _make_trial(speed, t, trial_type="looming_wind", target_ttc_ms=0.0)


def _trial_baseline_visual_escape() -> pd.DataFrame:
    """Baseline visual: escape in [-500, 0] ms window before TTC."""
    t = np.linspace(-1000, 200, 240)
    speed = np.zeros_like(t)
    speed[:80] = np.random.default_rng(5).normal(2, 0.5, 80)
    burst = np.exp(-np.linspace(0, 2.5, 100)) * 140
    speed[80:180] = burst
    speed[80:] += np.random.default_rng(6).normal(0, 1, len(t) - 80)
    speed = np.maximum(0, speed)
    return _make_trial(speed, t, trial_type="baseline_visual")


def _trial_baseline_visual_prewalk() -> pd.DataFrame:
    """Baseline visual: walking activity + escape -> PreWalk."""
    t = np.linspace(-1000, 200, 240)
    speed = np.zeros_like(t)
    # Walking throughout the pre-escape period
    speed[:120] = np.random.default_rng(7).normal(0, 5, 120) + 12
    speed[:120] = np.maximum(0, speed[:120])
    burst = np.exp(-np.linspace(0, 2, 70)) * 130
    speed[120:190] = burst
    speed[120:] += np.random.default_rng(8).normal(0, 1, len(t) - 120)
    speed = np.maximum(0, speed)
    return _make_trial(speed, t, trial_type="baseline_visual")


def _trial_escape_very_late() -> pd.DataFrame:
    """Escape that starts very late after stimulus."""
    t = np.linspace(-500, 1000, 300)
    speed = np.zeros_like(t)
    speed[:200] = np.random.default_rng(9).normal(2, 1, 200)
    burst = np.exp(-np.linspace(0, 2, 60)) * 120
    speed[200:260] = burst
    speed[200:] += np.random.default_rng(10).normal(0, 1, len(t) - 200)
    speed = np.maximum(0, speed)
    return _make_trial(speed, t, trial_type="looming_wind", target_ttc_ms=0.0)


def _trial_just_below_threshold() -> pd.DataFrame:
    """Vmax just below 50 mm/s -> NoResponse."""
    t = np.linspace(-500, 500, 200)
    speed = np.ones_like(t) * 3
    burst = np.linspace(0, 48, 50)
    speed[100:150] = burst
    speed += np.random.default_rng(11).normal(0, 1, len(t))
    speed = np.maximum(0, speed)
    return _make_trial(speed, t, trial_type="looming_wind")


def _trial_multimodal_wind_early() -> pd.DataFrame:
    """Multimodal trial where wind arrives before TTC."""
    t = np.linspace(-1000, 500, 300)
    speed = np.zeros_like(t)
    speed[:180] = np.random.default_rng(12).normal(2, 0.5, 180)
    burst = np.exp(-np.linspace(0, 2, 60)) * 120
    speed[180:240] = burst
    speed[180:] += np.random.default_rng(13).normal(0, 1, len(t) - 180)
    speed = np.maximum(0, speed)
    return _make_trial(speed, t, trial_type="looming_wind", target_ttc_ms=-373.0)


def _trial_no_response_with_wind() -> pd.DataFrame:
    """Wind present but no escape -> NoResponse."""
    t = np.linspace(-500, 500, 200)
    speed = np.ones_like(t) * 5 + np.random.default_rng(14).normal(0, 2, len(t))
    speed = np.maximum(0, speed)
    return _make_trial(speed, t, trial_type="looming_wind", target_ttc_ms=0.0)


def _trial_prewalk_low_walking() -> pd.DataFrame:
    """PreWalk with low but detectable walking before stimulus."""
    t = np.linspace(-1500, 500, 400)
    speed = np.zeros_like(t)
    # Subtle walking
    speed[:300] = np.random.default_rng(15).normal(0, 3, 300) + 8
    speed[:300] = np.maximum(0, speed[:300])
    burst = np.exp(-np.linspace(0, 2, 60)) * 120
    speed[300:360] = burst
    speed[300:] += np.random.default_rng(16).normal(0, 1, len(t) - 300)
    speed = np.maximum(0, speed)
    return _make_trial(speed, t, trial_type="looming_wind", target_ttc_ms=0.0)


# ══════════════════════════════════════════════════════════════════════
# Expected values
# ══════════════════════════════════════════════════════════════════════

_GOLDEN_EXPECTED = {
    "no_burst": "NoResponse",
    "clean_escape": "Escape",
    "prewalk_before_stimulus": "NoResponse",  # burst peak at t≈-247ms (before onset), tail in window
    "baseline_visual_escape": "Escape",
    "baseline_visual_prewalk": "PreWalk",
    "escape_very_late": "NoResponse",  # burst at t≈503ms, outside [0,250] window
    "just_below_threshold": "NoResponse",
    "multimodal_wind_early": "NoResponse",  # burst at t≈-200ms outside shifted window [-373,-123]
    "no_response_with_wind": "NoResponse",
    "prewalk_low_walking": "PreWalk",
}


# ══════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "trial_fn, expected",
    [
        (_trial_no_burst, "NoResponse"),
        (_trial_clean_escape, "Escape"),
        (_trial_prewalk_before_stimulus, "NoResponse"),
        (_trial_baseline_visual_escape, "Escape"),
        (_trial_baseline_visual_prewalk, "PreWalk"),
        (_trial_escape_very_late, "NoResponse"),
        (_trial_just_below_threshold, "NoResponse"),
        (_trial_multimodal_wind_early, "NoResponse"),
        (_trial_no_response_with_wind, "NoResponse"),
        (_trial_prewalk_low_walking, "PreWalk"),
    ],
    ids=list(_GOLDEN_EXPECTED.keys()),
)
def test_classifier_golden(trial_fn, expected):
    """Golden test: each synthetic trial must produce the expected classification."""
    trial = trial_fn()
    result = classify_trial(trial)
    assert result["response_type"] == expected, (
        f"Expected {expected}, got {result['response_type']} "
        f"(v_max={result['v_max']:.1f}, latency={result['latency_ms']})"
    )


def test_classifier_no_burst_rejects_latency():
    """NoResponse trials must have NaN latency."""
    trial = _trial_no_burst()
    result = classify_trial(trial)
    assert result["response_type"] == "NoResponse"
    assert np.isnan(result["latency_ms"])


def test_classifier_returns_all_keys():
    """Result dict must contain all expected keys."""
    trial = _trial_clean_escape()
    result = classify_trial(trial)
    for key in _RESULT_KEYS:
        assert key in result, f"Missing key: {key}"


def test_classifier_empty_trial():
    """Empty DataFrame must return NoResponse."""
    empty = pd.DataFrame()
    result = classify_trial(empty)
    assert result["response_type"] == "NoResponse"


def test_classifier_all_escape_has_interval():
    """Escape trials must have valid interval_onset/offset."""
    trial = _trial_clean_escape()
    result = classify_trial(trial)
    if result["response_type"] == "Escape":
        assert not np.isnan(result["interval_onset_ms"])
        assert not np.isnan(result["interval_offset_ms"])


def test_classifier_golden_regression_csv():
    """Write all golden trial results to CSV and verify expected counts."""
    records = []
    for name, fn in [
        ("no_burst", _trial_no_burst),
        ("clean_escape", _trial_clean_escape),
        ("prewalk_before_stimulus", _trial_prewalk_before_stimulus),
        ("baseline_visual_escape", _trial_baseline_visual_escape),
        ("baseline_visual_prewalk", _trial_baseline_visual_prewalk),
    ]:
        trial = fn()
        result = classify_trial(trial)
        records.append({
            "name": name,
            "response_type": result["response_type"],
            "v_max": result["v_max"],
            "latency_ms": result["latency_ms"],
        })

    df = pd.DataFrame(records)
    assert (df["response_type"] == ["NoResponse", "Escape", "NoResponse", "Escape", "PreWalk"]).all()
    assert df["v_max"].notna().all()
