"""Stopping measurement must not invent a response from optical count pulses."""
import numpy as np
import pytest

from cercus.core.kinematics.latency import find_stillness_onset, measure_stopping


def test_quantized_constant_slow_walk_has_no_stopping_event():
    t = np.arange(-1200.0, 300.0, 4.0)
    quantum = 0.04712389
    # Constant 6 mm/s, not a change of state. Individual counts alias across 10.
    position = np.floor((t - t[0]) * 6.0 / 1000.0 / quantum) * quantum
    speed = np.diff(position, prepend=position[0]) / 0.004
    assert np.isnan(find_stillness_onset(t, speed, 250.0, 0.0))


def test_prestopped_animal_is_not_rearmed_by_one_postwind_count():
    t = np.arange(-1200.0, 300.0, 5.0)
    speed = np.where(t < -30.0, 20.0, 0.0)
    speed[t == 10.0] = 12.0
    assert np.isnan(find_stillness_onset(t, speed, 120.0, 0.0))


def test_genuine_short_stop_is_not_clamped_to_biological_prior():
    t = np.arange(-1200.0, 300.0, 5.0)
    speed = np.where(t < 30.0, 20.0, 0.0)
    stop = find_stillness_onset(t, speed, 120.0, 0.0)
    assert 30.0 <= stop < 50.0


def test_short_history_is_reported_as_insufficient_baseline():
    t = np.arange(-500.0, 101.0, 5.0)
    result = measure_stopping(t, np.full_like(t, 20.0), np.nan, 0.0)
    assert np.isnan(result["onset_ms"])
    assert result["status"] == "insufficient_baseline"


def test_escape_onset_at_or_before_wind_is_not_a_stopping_trial():
    t = np.arange(-1100.0, 1.0, 5.0)
    result = measure_stopping(t, np.full_like(t, 20.0), -50.0, 0.0)
    assert np.isnan(result["onset_ms"])
    assert result["status"] == "escape_first"


def test_invalid_pre_wind_acquisition_interval_is_not_bridged():
    t = np.arange(-1100.0, 301.0, 5.0)
    t[100] = t[99]
    result = measure_stopping(t, np.full_like(t, 20.0), np.nan, 0.0)
    assert np.isnan(result["onset_ms"])
    assert result["status"] == "invalid_data"


def test_earlier_interruption_does_not_erase_observed_postwind_stop():
    t = np.arange(-1100.0, 301.0, 5.0)
    speed = np.where(t < 60.0, 20.0, 0.0)
    speed[(t >= -800.0) & (t < -600.0)] = 0.0
    result = measure_stopping(t, speed, 120.0, 0.0)
    assert result["status"] == "observed"
    assert result["onset_ms"] == 70.0
    assert result["baseline_status"] == "intermittent_moving"


@pytest.mark.parametrize("velocity", [9.9, 10.1, 11.0])
def test_near_threshold_quantization_is_not_a_stopping_response(velocity):
    t = np.arange(-1100.0, 301.0, 5.0)
    quantum = 0.04712389
    position = np.floor((t - t[0]) * velocity / 1000.0 / quantum) * quantum
    speed = np.diff(position, prepend=position[0]) / 0.005
    result = measure_stopping(t, speed, 200.0, 0.0)
    assert np.isnan(result["onset_ms"])
    assert result["status"] in {"not_moving_at_wind", "threshold_unresolved"}


@pytest.mark.parametrize("velocity", [8.0, 10.0, 12.0])
def test_count_scale_baseline_is_unresolved_on_either_side_of_threshold(velocity):
    t = np.arange(-1100.0, 301.0, 5.0)
    speed = np.where(t < 60.0, velocity, 0.0)
    result = measure_stopping(t, speed, 120.0, 0.0)
    assert np.isnan(result["onset_ms"])
    assert result["status"] == "threshold_unresolved"


def test_unresolved_first_dip_cannot_be_replaced_by_a_later_stop():
    t = np.arange(-1100.0, 301.0, 5.0)
    speed = np.full_like(t, 20.0)
    speed[(t >= 40.0) & (t < 80.0)] = 9.0
    speed[t >= 150.0] = 0.0
    result = measure_stopping(t, speed, 250.0, 0.0)
    assert np.isnan(result["onset_ms"])
    assert result["status"] == "threshold_unresolved"


def test_resolution_confirmation_keeps_the_first_crossing_timestamp():
    t = np.arange(-1100.0, 301.0, 5.0)
    speed = np.where(t < 60.0, 20.0, 5.0)
    result = measure_stopping(t, speed, 120.0, 0.0)
    # The first strict crossing at 70 ms is 8.75 mm/s. The same excursion
    # resolves below the count margin at 75 ms; do not move its endpoint.
    assert result["status"] == "observed"
    assert result["onset_ms"] == 70.0
    assert result["baseline_status"] == "continuous_moving"


def test_resolution_confirmation_can_follow_multiple_subthreshold_samples():
    t = np.arange(-1100.0, 301.0, 5.0)
    speed = np.where(t < 60.0, 20.0, 9.0)
    speed[t >= 95.0] = 5.0
    result = measure_stopping(t, speed, 120.0, 0.0)
    # Crossing at 75 ms remains below 10 until confirmation at 105 ms.
    assert result["status"] == "observed"
    assert result["onset_ms"] == 75.0


@pytest.mark.parametrize("escape_ms,status", [(70.0, "no_preescape_stop"), (75.0, "threshold_unresolved")])
def test_stopping_confirmation_must_strictly_precede_escape(escape_ms, status):
    t = np.arange(-1100.0, 301.0, 5.0)
    speed = np.where(t < 60.0, 20.0, 5.0)
    # Crossing is at 70 ms, confirmation at 75 ms. Neither may be borrowed
    # from the escape interval to establish a pre-escape stopping event.
    result = measure_stopping(t, speed, escape_ms, 0.0)
    assert np.isnan(result["onset_ms"])
    assert result["status"] == status


@pytest.mark.parametrize("invalid_time", [np.nan, np.inf, -500.0, 75.0])
def test_bad_timestamp_after_confirmation_preserves_observed_stop(invalid_time):
    t = np.arange(-1100.0, 301.0, 5.0)
    speed = np.where(t < 60.0, 20.0, 5.0)
    t[t == 80.0] = invalid_time
    result = measure_stopping(t, speed, 120.0, 0.0)
    assert result["status"] == "observed"
    assert result["onset_ms"] == 70.0


@pytest.mark.parametrize("corruption", ["nan", "inf", "backward", "duplicate"])
@pytest.mark.parametrize("corrupt_ms", [-500.0, 60.0, 75.0])
def test_bad_timestamp_before_confirmation_is_not_bridged(corrupt_ms, corruption):
    t = np.arange(-1100.0, 301.0, 5.0)
    speed = np.where(t < 60.0, 20.0, 5.0)
    index = int(np.flatnonzero(t == corrupt_ms)[0])
    t[index] = {
        "nan": np.nan, "inf": np.inf,
        "backward": t[index - 1] - 5.0, "duplicate": t[index - 1],
    }[corruption]
    result = measure_stopping(t, speed, 120.0, 0.0)
    assert result["status"] == "invalid_data"
    assert np.isnan(result["onset_ms"])


def test_low_speed_presence_is_separate_from_stopping_endpoint():
    t = np.arange(-1100.0, 301.0, 5.0)
    speed = np.where(t < -60.0, 20.0, 0.0)
    result = measure_stopping(t, speed, 120.0, 0.0)
    assert result["presence"] == "low_speed"
    assert result["status"] == "not_moving_at_wind"
    assert np.isnan(result["onset_ms"])


def test_no_low_speed_presence_is_distinct_from_unobserved_data():
    t = np.arange(-1100.0, 301.0, 5.0)
    result = measure_stopping(t, np.full_like(t, 20.0), 120.0, 0.0)
    assert result["presence"] == "no_low_speed"
    assert result["status"] == "no_preescape_stop"


def test_low_speed_presence_requires_wholly_postreference_support():
    t = np.arange(-1100.0, 301.0, 5.0)
    speed = np.where(t <= 0.0, 0.0, 20.0)
    result = measure_stopping(t, speed, 120.0, 0.0)
    assert result["presence"] == "no_low_speed"


@pytest.mark.parametrize("missing", ["start", "end", "gap"])
def test_incomplete_state_observations_cannot_establish_absence(missing):
    t = np.arange(-1100.0, 301.0, 5.0)
    speed = np.full_like(t, 20.0)
    if missing == "start":
        speed[(t > 0.0) & (t <= 25.0)] = np.nan
    elif missing == "end":
        t, speed = t[t <= 80.0], speed[t <= 80.0]
    else:
        keep = (t <= 75.0) | (t >= 120.0)
        t, speed = t[keep], speed[keep]
    result = measure_stopping(t, speed, 120.0, 0.0)
    assert result["presence"] == "unobserved"


def test_stop_to_escape_interval_is_not_stopping_reaction_time():
    t = np.arange(-1100.0, 301.0, 5.0)
    speed = np.where(t < 60.0, 20.0, 5.0)
    result = measure_stopping(t, speed, 120.0, 0.0)
    assert result["onset_ms"] == 70.0
    assert 120.0 - result["onset_ms"] == 50.0
