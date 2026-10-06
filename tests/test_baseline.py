"""Independent pre-stimulus baseline: raw session, reference kind, boundaries."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cercus.analysis.baseline import (
    KIND_UNOBSERVED,
    KIND_VISUAL,
    KIND_WIND,
    build_session_blocks,
    measure_window_prestim,
)


class _Cfg:
    class prewalk:
        window_ms = 1000.0

    class stillness:
        speed_window_ms = 20.0

    class reaction_time:
        wind_arrival_delay_ms = 0.0


def _session(n=1000, session_id=1, ard_start=0.0, step=5.0, amp=0.001, active=None):
    """Uniform acquisition session; ``amp`` per-frame displacement (mm)."""
    ard = ard_start + np.arange(n) * step
    kin = pd.DataFrame({
        "sys_time": 100.0 + ard / 1000.0,
        "ard_time": ard,
        "dx": np.full(n, amp),
        "dy": np.zeros(n),
        "dz": np.zeros(n),
        "stim_state": np.zeros(n),
        "global_trial_id": 1,
        "session_id": session_id,
    })
    if active is not None:
        kin.loc[active, "stim_state"] = 1.0
    return kin


def _window(kin, **over):
    w = {
        "global_trial_id": 1,
        "session_id": int(kin["session_id"].iloc[0]),
        "t_start": float(kin["sys_time"].iloc[0]),
        "t_stop": float(kin["sys_time"].iloc[-1]),
        "visual_onset_sys": np.nan,
    }
    w.update(over)
    return w


def test_visual_reference_maps_last_sample_at_or_before_event():
    kin = _session(n=1000, amp=0.001)  # ~0.2 mm/s, stationary
    blocks = build_session_blocks(kin)
    event = 100.0 + 2.5  # ard_time = 2500 ms → index 500
    row = measure_window_prestim(
        blocks, _window(kin, visual_onset_sys=event), "baseline_visual", t_zero_sys=event, cfg=_Cfg,
    )
    assert row["prestim_reference_kind"] == KIND_VISUAL
    assert row["prestim_status"] == "stationary"
    assert row["prestim_moving_fraction"] == pytest.approx(0.0)
    assert row["prestim_reference_sample_offset_ms"] == pytest.approx(0.0)


def test_missing_visual_event_stays_unobserved():
    kin = _session()
    blocks = build_session_blocks(kin)
    row = measure_window_prestim(
        blocks, _window(kin, visual_onset_sys=np.nan), "baseline_visual", t_zero_sys=100.0, cfg=_Cfg,
    )
    assert row["prestim_reference_kind"] == KIND_VISUAL
    assert row["prestim_status"] == KIND_UNOBSERVED
    assert np.isnan(row["prestim_moving_fraction"])


def test_wind_reference_is_first_active_frame_plus_delay():
    kin = _session(n=1000, active=slice(600, 700))
    blocks = build_session_blocks(kin)
    row = measure_window_prestim(
        blocks, _window(kin), "baseline_wind", t_zero_sys=100.0, cfg=_Cfg,
    )
    assert row["prestim_reference_kind"] == KIND_WIND
    # Reference host time equals the first active sample; delay 0 → offset is
    # exactly that sample's host time minus t_zero.
    expected = (float(kin["sys_time"].iloc[600]) - 100.0) * 1000.0
    assert row["prestim_reference_offset_ms"] == pytest.approx(expected)


def test_delay_is_included_in_wind_offset():
    class CfgDelay(_Cfg):
        class reaction_time:
            wind_arrival_delay_ms = 40.0

    kin = _session(n=1000, active=slice(600, 700))
    blocks = build_session_blocks(kin)
    row = measure_window_prestim(
        blocks, _window(kin), "baseline_wind", t_zero_sys=100.0, cfg=CfgDelay,
    )
    expected = (float(kin["sys_time"].iloc[600]) - 100.0) * 1000.0 + 40.0
    assert row["prestim_reference_offset_ms"] == pytest.approx(expected)


def test_short_history_is_unobserved():
    """A reference near the session start lacks the averaging margin."""
    kin = _session(n=1000, amp=0.001)
    blocks = build_session_blocks(kin)
    # Event at ard_time 300 ms → only 300 ms of history, less than 1000+20.
    event = 100.0 + 0.3
    row = measure_window_prestim(
        blocks, _window(kin, visual_onset_sys=event), "baseline_visual", t_zero_sys=event, cfg=_Cfg,
    )
    assert row["prestim_status"] == KIND_UNOBSERVED
    assert np.isnan(row["prestim_moving_fraction"])


def test_moving_history_fraction_in_unit_interval():
    kin = _session(n=1000, amp=1.0)  # 200 mm/s → clearly moving
    blocks = build_session_blocks(kin)
    event = 100.0 + 4.0  # ard_time 4000 ms
    row = measure_window_prestim(
        blocks, _window(kin, visual_onset_sys=event), "baseline_visual", t_zero_sys=event, cfg=_Cfg,
    )
    assert row["prestim_status"] in {"continuous_moving", "intermittent_moving"}
    assert 0.0 <= row["prestim_moving_fraction"] <= 1.0


def test_sessions_are_restricted_and_not_bridged():
    """Two sessions with restarted ard_time stay separate components."""
    a = _session(n=800, session_id=1, ard_start=0.0)
    b = _session(n=800, session_id=2, ard_start=0.0)
    kin = pd.concat([a, b], ignore_index=True)
    blocks = build_session_blocks(kin)
    assert set(blocks) == {1, 2}
    # A reference in session 2 must not see session 1's frames: a 1-s history
    # exists only if the walk stayed inside session 2.
    event = 100.0 + 3.0  # ard_time 3000 ms in session 2
    row = measure_window_prestim(
        blocks, _window(b, visual_onset_sys=event), "baseline_visual", t_zero_sys=event, cfg=_Cfg,
    )
    assert row["prestim_reference_kind"] == KIND_VISUAL
    assert row["prestim_status"] != KIND_UNOBSERVED


def test_corrupt_acquisition_gap_leaves_history_unobserved():
    kin = _session(n=1000, amp=0.001)
    # Inject an invalid (non-increasing) acquisition timestamp inside the
    # 1-s history window (reference at index 800, window start ~index 596).
    kin.loc[600, "ard_time"] = kin.loc[599, "ard_time"]
    blocks = build_session_blocks(kin)
    event = 100.0 + 4.0  # reference after the corruption
    row = measure_window_prestim(
        blocks, _window(kin, visual_onset_sys=event), "baseline_visual", t_zero_sys=event, cfg=_Cfg,
    )
    assert row["prestim_status"] == KIND_UNOBSERVED


def test_irregular_left_boundary_retains_carry_forward_average():
    # The left boundary is 1005 ms. Its carried observation at 999 ms needs
    # history from 979 ms, earlier than boundary-minus-width (985 ms).
    from cercus.analysis.baseline import _component_bounds, _session_speed_gap, measure_prestim_baseline

    times = np.arange(3.0, 2510.0, 6.0)
    dx = np.full(times.shape, 0.003)
    speed, gap = _session_speed_gap(times, dx, np.zeros_like(dx), 3.0)
    reference = 2005.0
    mapped = int(np.flatnonzero(times <= reference)[-1])
    left, right = _component_bounds(times, mapped, gap, reference, 1000.0, 20.0)
    full = measure_prestim_baseline(times, speed, reference, 1000.0)
    cropped = measure_prestim_baseline(times[left:right + 1], speed[left:right + 1], reference, 1000.0)
    assert full[0] == cropped[0] == "stationary"
    assert full[1] == cropped[1] == 0.0
    assert cropped[2] == pytest.approx(full[2], abs=1e-10)


def test_multisensory_baseline_uses_visual_event_and_no_wind_fallback():
    kin = _session(n=1000, active=slice(800, 900))
    kin.loc[:450, "dx"] = 1.0
    blocks = build_session_blocks(kin)
    visual = float(kin.sys_time.iloc[400])
    row = measure_window_prestim(
        blocks, _window(kin, visual_onset_sys=visual), "looming_wind", visual, cfg=_Cfg,
    )
    assert row["prestim_reference_kind"] == KIND_VISUAL
    assert row["prestim_status"] == "continuous_moving"
    assert row["prestim_moving_fraction"] == pytest.approx(1.0)
    missing = measure_window_prestim(blocks, _window(kin), "looming_wind", visual, cfg=_Cfg)
    assert missing["prestim_status"] == KIND_UNOBSERVED


def test_host_clock_drift_does_not_change_source_speed():
    kin = _session(n=1000, amp=0.06)
    # 12 mm/s on acquisition time, 6 mm/s if host intervals were used.
    kin["sys_time"] = 100.0 + kin.ard_time / 500.0
    visual = float(kin.sys_time.iloc[500])
    row = measure_window_prestim(
        build_session_blocks(kin), _window(kin, visual_onset_sys=visual),
        "baseline_visual", visual, cfg=_Cfg,
    )
    assert row["prestim_status"] == "continuous_moving"
    assert row["prestim_reference_speed_mm_s"] == pytest.approx(12.0)


def test_arrival_delay_changes_measured_history():
    class CfgDelay(_Cfg):
        class reaction_time:
            wind_arrival_delay_ms = 100.0

    kin = _session(n=1000, amp=0.001, active=slice(600, 700))
    kin.loc[601:650, "dx"] = 0.1
    blocks = build_session_blocks(kin)
    zero = measure_window_prestim(blocks, _window(kin), "baseline_wind", 100.0, cfg=_Cfg)
    delayed = measure_window_prestim(blocks, _window(kin), "baseline_wind", 100.0, cfg=CfgDelay)
    assert zero["prestim_status"] == "stationary"
    assert zero["prestim_moving_fraction"] == 0.0
    assert delayed["prestim_status"] == "intermittent_moving"
    assert delayed["prestim_moving_fraction"] > 0.0
    assert delayed["prestim_reference_speed_mm_s"] == pytest.approx(20.0)


def test_session_start_cannot_borrow_complete_history_from_previous_session():
    a = _session(n=800, session_id=1)
    b = _session(n=800, session_id=2)
    visual = float(b.sys_time.iloc[60])
    row = measure_window_prestim(
        build_session_blocks(pd.concat([a, b], ignore_index=True)),
        _window(b, visual_onset_sys=visual), "baseline_visual", visual, cfg=_Cfg,
    )
    assert row["prestim_status"] == KIND_UNOBSERVED
    assert np.isnan(row["prestim_moving_fraction"])


def test_unknown_trial_type_unobserved():
    kin = _session()
    blocks = build_session_blocks(kin)
    row = measure_window_prestim(blocks, _window(kin), "trial", t_zero_sys=100.0, cfg=_Cfg)
    assert row["prestim_reference_kind"] == KIND_UNOBSERVED
    assert row["prestim_status"] == KIND_UNOBSERVED
