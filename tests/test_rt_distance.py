"""Reaction-time / distance pure-function checks (hand-computed trapezoid)."""

import numpy as np
import pandas as pd
import pytest

from cercus.core.kinematics.distance import compute_reaction_and_distance
from cercus.core.kinematics.latency import compute_escape_latency
from pipeline.classifier import classify_trial, label_trials


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


def _wind_trial(wind_ms=0.0, pause_start_ms=60.0, early_burst=False):
    # The default trace steps 20 → 2 mm/s at ``pause_start_ms`` (60 ms), but the
    # reported stopping endpoint is the *causal 20 ms averaged-speed crossing*,
    # which lags the true step by one frame (60 → 70 ms).  Tests assert that
    # filtered endpoint — the value the pipeline actually reports — and never
    # treat it as the exact instant the animal stopped.
    t = np.arange(-1100.0, 400.0, 10.0)
    speed = np.full_like(t, 20.0)
    if early_burst:
        speed[(t >= -40.0) & (t < 60.0)] = 120.0
    speed[(t >= pause_start_ms) & (t < 120.0)] = 2.0
    speed[(t >= 120.0) & (t < 220.0)] = 120.0
    speed[t >= 220.0] = 2.0
    return pd.DataFrame({
        "t_rel": t + wind_ms,
        "speed": speed,
        "speed_raw": speed,
        "t_acquisition_rel": t + wind_ms,
        "type": "baseline_wind" if wind_ms == 0 else "looming_wind",
        "target_ttc_ms": np.nan if wind_ms == 0 else wind_ms,
        "global_trial_id": 1,
    })


@pytest.mark.parametrize("wind_ms", [0.0, -373.0])
def test_wind_prewalk_rt_is_filtered_stop_endpoint_not_true_step(wind_ms):
    trial = _wind_trial(wind_ms)
    result = classify_trial(trial)
    assert result["response_type"] == "PreWalk"
    # Raw step is 60 ms; the causal 20 ms averaged-speed crossing lands one
    # frame later at 70 ms.  This is the filtered endpoint the pipeline
    # reports, not an exact reaction time.
    assert result["reaction_time_ms"] == 70.0
    assert result["escape_reaction_time_ms"] == 120.0
    assert result["stillness_reaction_time_ms"] == 70.0
    assert result["stop_to_escape_interval_ms"] == 50.0
    assert result["stillness_status"] == "observed"
    assert not result["short_rt"]
    assert result["interval_onset_ms"] == wind_ms + 120.0
    assert result["latency_ms"] == wind_ms + 120.0
    expected = compute_reaction_and_distance(
        trial["t_rel"].values, trial["speed"].values,
        wind_ms + 120.0, wind_ms + 220.0, wind_ms,
    )
    assert result["distance_mm"] == expected["distance_mm"]
    assert result["distance_500ms_mm"] == expected["distance_500ms_mm"]


@pytest.mark.parametrize("wind_ms", [0.0, -373.0])
@pytest.mark.parametrize("history", ["missing_speed", "gap", "short"])
def test_unobserved_cohort_retains_classifier_escape_timing(wind_ms, history):
    from cercus.analysis.reaction_time import select_escape_latency

    trial = _wind_trial(wind_ms)
    if history == "missing_speed":
        trial.loc[trial["t_rel"] == wind_ms - 500.0, "speed_raw"] = np.nan
    elif history == "gap":
        trial = trial.loc[~trial["t_rel"].between(wind_ms - 550.0, wind_ms - 500.0)]
    else:
        trial = trial.loc[trial["t_rel"] >= wind_ms - 500.0]
    result = classify_trial(trial)
    assert result["response_type"] == "Escape"
    assert result["pause_baseline_status"] == "unobserved"
    assert result["stillness_baseline_status"] == "unobserved"
    assert result["pause_status"] == result["stillness_status"] == "observed"
    assert result["pause_stopping_time_ms"] == result["stillness_reaction_time_ms"] == 70.0
    assert result["pause_to_escape_time_ms"] == result["stop_to_escape_interval_ms"] == 50.0
    assert result["pause_reaction_time_ms"] == result["escape_reaction_time_ms"] == 120.0
    assert select_escape_latency(pd.DataFrame([{**result, "type": trial["type"].iloc[0]}])).iloc[0] == 120.0


def test_wind_prewalk_skips_ongoing_burst_before_stillness():
    result = classify_trial(_wind_trial(early_burst=True))
    assert result["response_type"] == "PreWalk"
    assert result["reaction_time_ms"] == 70.0
    assert result["interval_onset_ms"] == 120.0


def test_stop_to_escape_interval_uses_source_clock_for_both_endpoints():
    trial = _wind_trial()
    # Keep the wind reference aligned at zero but use a different acquisition
    # clock; T2 must not be computed from the host ``t_rel`` values.
    trial["t_acquisition_rel"] = trial["t_rel"] * 1.5
    result = classify_trial(trial)
    # At the first low row, the 20-ms average is (20*5 + 2*15)/20 = 6.5.
    assert result["stillness_reaction_time_ms"] == 90.0
    assert result["escape_reaction_time_ms"] == 120.0
    assert result["stop_to_escape_interval_ms"] == 90.0


def test_stop_to_escape_interval_rejects_duplicate_at_actual_escape_row():
    trial = _wind_trial()
    trial.loc[trial["t_rel"] == 110.0, "t_acquisition_rel"] = 120.0
    result = classify_trial(trial)
    assert result["stillness_status"] == "observed"
    assert result["stillness_reaction_time_ms"] == 70.0
    assert np.isnan(result["stop_to_escape_interval_ms"])


def test_stop_to_escape_interval_is_nan_without_either_endpoint():
    trial = _wind_trial()
    trial["t_acquisition_rel"] = np.nan
    result = classify_trial(trial)
    assert np.isnan(result["stop_to_escape_interval_ms"])


def test_prewind_stillness_is_not_clipped_into_postwind_reaction():
    trial = _wind_trial(pause_start_ms=-60.0)
    trial["speed_raw"] = np.where(trial["t_rel"] < -60.0, 20.0, 2.0)
    result = classify_trial(trial)
    assert result["response_type"] == "Escape"
    assert result["reaction_time_ms"] == 120.0
    assert result["stillness_status"] == "not_moving_at_wind"
    assert result["interval_onset_ms"] == 120.0
    assert np.isfinite(result["distance_mm"])


def test_missing_speed_after_stopping_does_not_erase_observed_crossing():
    trial = _wind_trial()
    trial.loc[trial["t_rel"] == 90.0, "speed"] = np.nan
    result = classify_trial(trial)
    assert result["response_type"] == "PreWalk"
    assert result["stillness_reaction_time_ms"] == 70.0
    assert result["stillness_status"] == "observed"
    assert np.isfinite(result["distance_mm"])


@pytest.mark.parametrize("missing_ms", [50.0, 60.0])
def test_missing_speed_at_crossing_does_not_create_stopping(missing_ms):
    trial = _wind_trial()
    trial.loc[trial["t_rel"] == missing_ms, "speed_raw"] = np.nan
    trial.loc[trial["t_rel"] >= 220.0, "speed"] = 20.0
    result = classify_trial(trial)
    assert result["response_type"] == "PreWalk"
    assert np.isnan(result["stillness_reaction_time_ms"])
    assert result["stillness_status"] == "invalid_data"


def test_missing_frames_after_stopping_do_not_erase_observed_crossing():
    trial = _wind_trial()
    trial = trial[~trial["t_rel"].isin([80.0, 90.0, 100.0])]
    result = classify_trial(trial)
    assert result["response_type"] == "PreWalk"
    assert result["stillness_reaction_time_ms"] == 70.0
    assert result["stillness_status"] == "observed"
    assert result["escape_reaction_time_ms"] == 120.0


@pytest.mark.parametrize("invalid_time", [np.nan, 70.0, -500.0])
def test_bad_source_timestamp_after_stop_keeps_classifier_endpoint(invalid_time):
    trial = _wind_trial()
    trial.loc[trial["t_rel"] == 80.0, "t_acquisition_rel"] = invalid_time
    result = classify_trial(trial)
    assert result["response_type"] == "PreWalk"
    assert result["stillness_status"] == "observed"
    assert result["stillness_reaction_time_ms"] == 70.0
    assert result["escape_reaction_time_ms"] == 120.0
    assert np.isnan(result["stop_to_escape_interval_ms"])
    assert np.isfinite(result["distance_mm"])


def test_missing_frames_at_crossing_do_not_create_stopping():
    trial = _wind_trial()
    trial = trial[~trial["t_rel"].isin([50.0, 60.0, 70.0, 80.0])].copy()
    trial.loc[(trial["t_rel"] >= 220.0) & (trial["t_rel"] < 300.0), "speed"] = 20.0
    result = classify_trial(trial)
    assert np.isnan(result["stillness_reaction_time_ms"])
    assert result["stillness_status"] == "invalid_data"
    assert result["escape_reaction_time_ms"] == 120.0


def test_observed_frame_jitter_does_not_invalidate_stopping():
    trial = _wind_trial()
    trial = trial[trial["t_rel"] != 90.0]
    result = classify_trial(trial)
    assert result["stillness_reaction_time_ms"] == 70.0
    assert result["escape_reaction_time_ms"] == 120.0


def test_wind_escape_keeps_movement_rt_without_50ms_clipping():
    trial = _wind_trial()
    trial.loc[trial["t_rel"] < 120.0, "speed"] = 2.0
    trial.loc[(trial["t_rel"] >= 30.0) & (trial["t_rel"] < 220.0), "speed"] = 120.0
    trial["speed_raw"] = trial["speed"]
    result = classify_trial(trial)
    assert result["response_type"] == "Escape"
    assert result["reaction_time_ms"] == 30.0
    assert result["escape_reaction_time_ms"] == 30.0
    assert np.isnan(result["stillness_reaction_time_ms"])
    assert result["short_rt"]


def test_wind_continuous_movement_has_no_fabricated_onset():
    trial = _wind_trial()
    trial["speed"] = 120.0
    trial["speed_raw"] = 120.0
    result = classify_trial(trial)
    assert result["response_type"] == "PreWalk"
    assert np.isnan(result["reaction_time_ms"])
    assert np.isnan(result["interval_onset_ms"])
    assert result["stillness_status"] == "no_preescape_stop"


@pytest.mark.parametrize("early_burst", [False, True])
def test_wind_onset_precedes_gradual_acceleration_to_burst(early_burst):
    trial = _wind_trial(early_burst=early_burst)
    trial.loc[trial["t_rel"] == 120.0, "speed"] = 15.0
    trial.loc[trial["t_rel"] == 130.0, "speed"] = 40.0
    result = classify_trial(trial)
    assert result["interval_onset_ms"] == 120.0
    assert result["escape_reaction_time_ms"] == 120.0
    assert result["stillness_reaction_time_ms"] == 70.0


def test_wind_onset_at_boundary_does_not_reuse_earlier_walking_onset():
    trial = _wind_trial()
    trial.loc[trial["t_rel"] < 0.0, "speed"] = 2.0
    trial.loc[(trial["t_rel"] >= 0.0) & (trial["t_rel"] < 220.0), "speed"] = 120.0
    trial["speed_raw"] = trial["speed"]
    result = classify_trial(trial)
    assert result["response_type"] == "Escape"
    assert result["interval_onset_ms"] == 0.0
    assert result["escape_reaction_time_ms"] == 0.0
    assert result["short_rt"]


@pytest.mark.parametrize("tail_speed", [np.nan, 20.0])
def test_observed_stopping_is_independent_of_later_tail(tail_speed):
    trial = _wind_trial()
    trial["speed_raw"] = trial["speed"]
    trial.loc[(trial["t_rel"] >= 100.0) & (trial["t_rel"] < 120.0), "speed_raw"] = tail_speed
    result = classify_trial(trial)
    assert result["response_type"] == "PreWalk"
    assert result["stillness_reaction_time_ms"] == 70.0
    assert result["escape_reaction_time_ms"] == 120.0
    assert np.isfinite(result["distance_mm"])


@pytest.mark.parametrize("pause_start_ms", [0.0, 10.0])
def test_first_quiet_frame_spanning_wind_arrival_is_not_postwind_stillness(pause_start_ms):
    from cercus.core.kinematics.latency import find_stillness_onset

    trial = _wind_trial(pause_start_ms=pause_start_ms)
    trial.loc[trial["t_rel"] >= 120.0, "speed"] = 20.0
    # Each raw speed sample represents displacement since the previous frame.
    wind_arrival_ms = pause_start_ms - 5.0
    assert np.isnan(find_stillness_onset(
        trial["t_rel"].to_numpy(), trial["speed"].to_numpy(),
        120.0, wind_arrival_ms,
    ))


def test_stillness_allows_observed_slow_acceleration_tail():
    trial = _wind_trial()
    trial["speed_raw"] = trial["speed"]
    trial.loc[(trial["t_rel"] >= 100.0) & (trial["t_rel"] < 120.0), "speed_raw"] = 5.0
    result = classify_trial(trial)
    assert result["stillness_reaction_time_ms"] == 70.0
    assert result["escape_reaction_time_ms"] == 120.0


def test_raw_speed_uses_observed_frame_dt_and_rejects_invalid_timestamps():
    from pipeline.kinematics import _integrate_trial

    # Optical deltas belong to acquisition intervals (ard_time), not to the
    # host receive clock (sys_time): the two deliberately differ here.
    sys_time = np.arange(40) * 0.01
    sys_time[5:] += 0.01
    sys_time[8] = sys_time[7]
    ard_time = np.arange(40) * 10.0
    ard_time[5:] += 10.0
    ard_time[8] = ard_time[7]
    trial = pd.DataFrame({
        "sys_time": sys_time,
        "ard_time": ard_time,
        "dx": np.full(40, 0.2),
        "dy": np.zeros(40),
        "dz": np.zeros(40),
    })
    integrated = _integrate_trial(trial, 0.0)
    assert integrated["speed_raw"].iloc[:2].isna().all()
    assert integrated["speed_raw"].iloc[4] == pytest.approx(20.0)
    assert integrated["speed_raw"].iloc[5] == pytest.approx(10.0)
    assert np.isnan(integrated["speed_raw"].iloc[8])


def test_acquisition_time_keeps_source_intervals_at_wind_reference():
    from pipeline.kinematics import _integrate_trial

    trial = pd.DataFrame({
        "sys_time": np.arange(40) * 0.01,
        "ard_time": np.arange(40) * 5.0,
        "dx": np.full(40, 0.1),
        "dy": np.zeros(40),
        "dz": np.zeros(40),
        "stim_state": np.arange(40) >= 10,
    })
    integrated = _integrate_trial(trial, 0.0)
    assert integrated["t_acquisition_rel"].iloc[10] == 100.0
    assert integrated["t_acquisition_rel"].iloc[11] == 105.0
    assert integrated["t_rel"].iloc[11] == 110.0
    assert integrated["speed_raw"].iloc[11] == pytest.approx(20.0)


@pytest.mark.parametrize("trial_type,wind_ms", [("baseline_wind", 0.0), ("looming_wind", -373.0)])
def test_nonbinary_stimulus_spike_does_not_shift_wind_reference(trial_type, wind_ms):
    from pipeline.constants import DETAILS_KEYS
    from pipeline.kinematics import preprocess

    sys_time = np.arange(100) * 0.01
    stimulus = np.where((np.arange(100) >= 40) & (np.arange(100) < 60), 1.0, 0.0)
    stimulus[10] = 967.0
    trial = pd.DataFrame({
        "sys_time": sys_time, "ard_time": np.arange(100) * 5.0,
        "dx": np.full(100, 0.1), "dy": 0.0, "dz": 0.0,
        "stim_state": stimulus, "global_trial_id": 1,
    })
    meta = pd.DataFrame([{**dict.fromkeys(DETAILS_KEYS, np.nan),
                          "global_trial_id": 1, "type": trial_type,
                          "target_ttc_ms": wind_ms}])
    windows = [{"global_trial_id": 1, "t_start": 0.0, "t_stop": 0.99}]
    integrated = preprocess(meta, windows, {}, trial)
    assert integrated["t_rel"].iloc[40] == pytest.approx(wind_ms)
    assert integrated["t_acquisition_rel"].iloc[40] == pytest.approx(wind_ms)
    assert integrated["t_acquisition_rel"].iloc[41] == pytest.approx(wind_ms + 5.0)
    # Do not rewrite raw telemetry or let clipping a spike to 1 create an onset.
    assert integrated["stim_state"].iloc[10] == 967.0


def test_integrate_trial_ignores_nonbinary_stimulus_spike():
    from pipeline.kinematics import _integrate_trial

    trial = pd.DataFrame({
        "sys_time": np.arange(40) * 0.01,
        "ard_time": np.arange(40) * 5.0,
        "dx": np.full(40, 0.1), "dy": 0.0, "dz": 0.0,
        "stim_state": np.where(np.arange(40) >= 10, 1.0, 0.0),
    })
    trial.loc[3, "stim_state"] = 967.0
    integrated = _integrate_trial(trial, 0.0)
    assert integrated["t_acquisition_rel"].iloc[10] == pytest.approx(100.0)
    assert integrated["t_acquisition_rel"].iloc[11] == pytest.approx(105.0)


def test_raw_speed_prevents_smoothed_early_stopping_time():
    trial = _wind_trial(pause_start_ms=30.0)
    trial["speed_raw"] = _wind_trial(pause_start_ms=60.0)["speed"]
    result = classify_trial(trial)
    # The smoothed trace steps at 30 ms but raw speed (which gates stopping)
    # steps at 60 ms; the averaged endpoint is 70 ms.
    assert result["reaction_time_ms"] == 70.0
    assert result["escape_reaction_time_ms"] == 120.0
    expected = classify_trial(_wind_trial(pause_start_ms=30.0))
    assert result["interval_onset_ms"] == expected["interval_onset_ms"]
    assert result["distance_mm"] == expected["distance_mm"]


def test_early_stopping_is_not_short_escape_latency():
    result = classify_trial(_wind_trial(pause_start_ms=30.0))
    # Raw step at 30 ms; averaged endpoint one frame later at 40 ms.
    assert result["reaction_time_ms"] == 40.0
    assert result["stillness_reaction_time_ms"] == 40.0
    assert result["stillness_status"] == "observed"
    assert result["escape_reaction_time_ms"] == 120.0
    assert not result["short_rt"]


def test_prewalk_rt_accepts_sustained_sub10_speed():
    trial = _wind_trial()
    trial.loc[(trial["t_rel"] >= 60.0) & (trial["t_rel"] < 120.0), "speed"] = 5.0
    trial["speed_raw"] = trial["speed"]
    result = classify_trial(trial)
    assert result["response_type"] == "PreWalk"
    assert result["reaction_time_ms"] == 70.0
    assert result["stillness_reaction_time_ms"] == 70.0
    assert result["stillness_status"] == "observed"
    assert result["escape_reaction_time_ms"] == 120.0
    assert np.isfinite(result["distance_mm"])


def test_single_frame_pause_before_escape_does_not_establish_stopping():
    # A single sub-threshold frame is not a sustained stop: the causal
    # trailing average is still moving, so no stopping event may be reported.
    trial = _wind_trial(pause_start_ms=110.0)
    result = classify_trial(trial)
    assert result["response_type"] == "PreWalk"
    assert np.isnan(result["stillness_reaction_time_ms"])
    assert result["stillness_status"] == "no_preescape_stop"
    assert result["escape_reaction_time_ms"] == 120.0


def test_label_trials_exports_separate_latencies_and_qc():
    labeled = label_trials(_wind_trial())
    assert (labeled["escape_reaction_time_ms"] == 120.0).all()
    assert (labeled["stillness_reaction_time_ms"] == 70.0).all()
    assert (labeled["stillness_status"] == "observed").all()
    assert (labeled["stillness_presence"] == "low_speed").all()
    assert (labeled["stop_to_escape_interval_ms"] == 50.0).all()
    assert (labeled["stillness_baseline_status"] == "continuous_moving").all()
    assert labeled["pause_moving_eligible"].all()
    assert labeled["pause_moving_fraction"].eq(1.0).all()
    assert not labeled["short_rt"].any()


def test_stillness_presence_does_not_require_a_resolvable_stopping_rt():
    trial = _wind_trial(pause_start_ms=-60.0)
    trial["speed_raw"] = np.where(trial["t_rel"] < -60.0, 20.0, 2.0)
    labeled = label_trials(trial)
    assert labeled["stillness_presence"].eq("low_speed").all()
    assert labeled["stillness_status"].eq("not_moving_at_wind").all()
    assert labeled["stillness_reaction_time_ms"].isna().all()


def test_stillness_plot_separates_state_presence_from_stopping_rt():
    import matplotlib.pyplot as plt
    from cercus.visualization.behavior import plot_prewalk_stillness

    # Keep the full moving history, but the first post-wind excursion is
    # count-scale unresolved: a low-speed level exists without a stopping RT.
    source = _wind_trial()
    source.loc[(source["t_rel"] >= 40.0) & (source["t_rel"] < 80.0), "speed_raw"] = 9.0
    trial = label_trials(source)
    trial["stillness_reaction_time_ms"] = np.nan
    trial["global_trial_index"] = 1
    trial["subject_id"] = "a"
    fig = plot_prewalk_stillness(trial)
    assert [bar.get_height() for bar in fig.axes[0].patches] == [1.0, 0.0, 0.0]
    plt.close(fig)


def test_stillness_plot_does_not_call_missing_data_absence():
    import matplotlib.pyplot as plt
    from cercus.visualization.behavior import plot_prewalk_stillness

    trial = _wind_trial()
    trial.loc[trial["t_rel"] > 0.0, "speed_raw"] = np.nan
    trial = label_trials(trial)
    trial["global_trial_index"] = 1
    trial["subject_id"] = "a"
    fig = plot_prewalk_stillness(trial)
    assert [bar.get_height() for bar in fig.axes[0].patches] == [0.0, 0.0, 1.0]
    plt.close(fig)


def test_stopping_threshold_is_configurable(monkeypatch):
    from cercus.config import get_thresholds

    trial = _wind_trial()
    trial.loc[(trial["t_rel"] >= 60.0) & (trial["t_rel"] < 120.0), "speed"] = 5.0
    trial["speed_raw"] = trial["speed"]
    # 5 mm/s is below the default 10 mm/s quiet threshold → observed stop.
    assert classify_trial(trial)["stillness_reaction_time_ms"] == 70.0
    # Tighten the threshold below the pause speed: the same trace no longer
    # crosses it, so the config value is what gates the event.
    monkeypatch.setitem(get_thresholds().baseline._data, "quiet_mm_s", 3.0)
    result = classify_trial(trial)
    assert np.isnan(result["stillness_reaction_time_ms"])
    assert result["stillness_status"] == "no_preescape_stop"


def test_isolated_sub_threshold_frame_does_not_establish_stopping():
    trial = _wind_trial()
    trial.loc[trial["t_rel"] == 20.0, "speed"] = 5.0
    result = classify_trial(trial)
    assert result["stillness_reaction_time_ms"] == 70.0
    assert result["reaction_time_ms"] == 70.0
    assert result["escape_reaction_time_ms"] == 120.0
    assert not result["short_rt"]


def test_stopping_uses_first_crossing_not_final_pause_before_escape():
    trial = _wind_trial()
    trial["speed_raw"] = np.where(trial["t_rel"] >= 40.0, 5.0, 20.0)
    result = classify_trial(trial)
    # Raw step at 40 ms; averaged endpoint at 50 ms. The later 60 ms pause is
    # never reached because the first crossing already fired.
    assert result["stillness_reaction_time_ms"] == 50.0
    assert result["escape_reaction_time_ms"] == 120.0


@pytest.mark.parametrize("stop_ms", [120.0, 220.0])
def test_escape_deceleration_is_not_prewalk_stopping(stop_ms):
    trial = _wind_trial()
    trial["speed_raw"] = np.where(trial["t_rel"] < stop_ms, 20.0, 5.0)
    result = classify_trial(trial)
    assert result["response_type"] == "PreWalk"
    assert np.isnan(result["stillness_reaction_time_ms"])
    assert np.isnan(result["reaction_time_ms"])
    # Deceleration at/after the escape onset is not a pre-escape stop.
    assert result["stillness_status"] == "no_preescape_stop"
    assert result["escape_reaction_time_ms"] == 120.0
    assert np.isfinite(result["distance_mm"])


def test_stopping_is_measurable_with_unresolved_escape_onset():
    trial = _wind_trial()
    trial["speed"] = 120.0
    trial["speed_raw"] = np.where(trial["t_rel"] >= 60.0, 5.0, 20.0)
    result = classify_trial(trial)
    assert result["response_type"] == "PreWalk"
    # Raw step at 60 ms; averaged endpoint at 70 ms — stopping is measured even
    # though the escape onset cannot be resolved.
    assert result["stillness_reaction_time_ms"] == 70.0
    assert result["stillness_status"] == "observed"
    assert np.isnan(result["escape_reaction_time_ms"])
    assert np.isnan(result["stop_to_escape_interval_ms"])
    assert np.isnan(result["distance_mm"])


@pytest.mark.parametrize("stop_ms,expected", [(240.0, 250.0), (250.0, np.nan)])
@pytest.mark.parametrize("wind_ms", [0.0, -373.0])
def test_stopping_uses_postwind_250ms_window(stop_ms, expected, wind_ms):
    from cercus.core.kinematics.latency import find_stillness_onset

    # Raw step at ``stop_ms``; the averaged crossing is one frame later. A step
    # at 240 ms is still inside the post-wind 250 ms window; a step at the
    # window edge (250 ms) cannot establish a wholly post-wind event.
    t = np.arange(-1100.0, 400.0, 10.0) + wind_ms
    speed = np.where(t < wind_ms + stop_ms, 20.0, 5.0)
    actual = find_stillness_onset(t, speed, np.nan, wind_ms) - wind_ms
    if np.isnan(expected):
        assert np.isnan(actual)
    else:
        assert actual == expected


@pytest.mark.parametrize("invalid_time", [np.nan, 40.0, 50.0])
def test_invalid_crossing_timestamp_cannot_establish_stopping(invalid_time):
    from cercus.core.kinematics.latency import find_stillness_onset

    t = np.arange(-1100.0, 200.0, 10.0)
    speed = np.where(t < 60.0, 20.0, 5.0)
    t[t == 60.0] = invalid_time
    assert np.isnan(find_stillness_onset(t, speed, np.nan, 0.0))


def test_stopping_requires_strictly_below_threshold():
    from cercus.core.kinematics.latency import find_stillness_onset

    t = np.arange(-1100.0, 300.0, 10.0)
    speed = np.where(t < 70.0, 20.0, 10.0)
    assert np.isnan(find_stillness_onset(t, speed, np.nan, 0.0))
    speed[t >= 70.0] = 9.0
    # A 1 mm/s dip is smaller than the default count-resolution margin.
    assert np.isnan(find_stillness_onset(t, speed, np.nan, 0.0))
    speed[t >= 70.0] = 5.0
    assert find_stillness_onset(t, speed, np.nan, 0.0) == 80.0


def test_mismatched_stopping_arrays_do_not_establish_a_crossing():
    from cercus.core.kinematics.latency import find_stillness_onset

    assert np.isnan(find_stillness_onset(
        np.array([0.0, 10.0, 20.0]), np.array([20.0, 5.0]), np.nan, 0.0,
    ))


def test_missing_wind_timestamp_does_not_establish_stopping():
    from cercus.core.kinematics.latency import find_stillness_onset

    trial = _wind_trial()
    assert np.isnan(find_stillness_onset(
        trial["t_rel"].to_numpy(), trial["speed"].to_numpy(), 120.0, np.nan,
    ))


def test_missing_acquisition_clock_does_not_establish_stopping():
    trial = _wind_trial()
    trial["t_acquisition_rel"] = np.nan
    result = classify_trial(trial)
    assert result["response_type"] == "Escape"
    assert result["pause_baseline_status"] == "unobserved"
    assert np.isnan(result["stillness_reaction_time_ms"])
    assert result["stillness_status"] == result["pause_status"] == "missing_acquisition_clock"
    assert result["escape_reaction_time_ms"] == 120.0


def test_boundary_straddling_stop_is_not_rearmed_as_later_stillness():
    from cercus.core.kinematics.latency import measure_stopping

    # The averaged crossing lands at t=15, whose 20 ms window reaches back to
    # t=-5 — not a wholly post-wind stop. The animal must not be re-armed on
    # the later re-stop.
    t = np.arange(-1100.0, 400.0, 5.0)
    speed = np.where(t < 5.0, 20.0, 4.0)
    speed[(t >= 100.0) & (t < 200.0)] = 20.0
    speed[t >= 200.0] = 4.0
    result = measure_stopping(t, speed, np.nan, 0.0)
    assert np.isnan(result["onset_ms"])
    assert result["status"] == "wind_boundary"


def test_wind_arrival_calibration_applies_to_both_latencies(monkeypatch):
    from cercus.config import get_thresholds
    monkeypatch.setitem(get_thresholds().reaction_time._data, "wind_arrival_delay_ms", 10.0)
    result = classify_trial(_wind_trial())
    assert result["reaction_time_ms"] == 60.0
    assert result["escape_reaction_time_ms"] == 110.0
    assert result["interval_onset_ms"] == 120.0


def test_summary_export_preserves_separate_latencies_and_bool_qc(tmp_path):
    from pipeline.io import export_summary_metrics

    labeled = label_trials(_wind_trial(pause_start_ms=30.0))
    labeled["global_trial_index"] = 1
    labeled["session_id"] = 1
    output = export_summary_metrics(labeled, tmp_path / "summary.csv")
    summary = pd.read_csv(output)
    row = summary.iloc[0]
    # Raw step at 30 ms; averaged endpoint at 40 ms.
    assert row["reaction_time_ms"] == 40.0
    assert row["stillness_reaction_time_ms"] == 40.0
    assert row["escape_reaction_time_ms"] == 120.0
    assert row["stillness_presence"] == "low_speed"
    assert row["stop_to_escape_interval_ms"] == 80.0
    assert row["stillness_baseline_status"] == "continuous_moving"
    assert row["pause_moving_eligible"]
    assert row["pause_moving_fraction"] == 1.0
    assert summary["pause_moving_eligible"].dtype == bool
    assert not row["short_rt"]
    assert summary["short_rt"].dtype == bool


def test_calibration_rejects_pause_started_before_air_arrival(monkeypatch):
    from cercus.config import get_thresholds

    monkeypatch.setitem(get_thresholds().reaction_time._data, "wind_arrival_delay_ms", 80.0)
    trial = _wind_trial()
    trial["speed_raw"] = np.where(trial["t_rel"] < 60.0, 20.0, 2.0)
    result = classify_trial(trial)
    assert result["response_type"] == "Escape"
    assert result["pause_baseline_status"] == "intermittent_moving"
    assert np.isnan(result["stillness_reaction_time_ms"])
    assert result["reaction_time_ms"] == 40.0
    assert result["escape_reaction_time_ms"] == 40.0
    assert result["short_rt"]


def test_pause_response_does_not_use_centered_escape_endpoint():
    trial = _wind_trial()
    # Centered smoothing suggests escape 30 ms before source displacement.
    trial.loc[(trial["t_rel"] >= 90.0) & (trial["t_rel"] < 120.0), "speed"] = 120.0
    result = classify_trial(trial)
    assert result["escape_reaction_time_ms"] == 90.0
    assert result["pause_reaction_time_ms"] == 120.0
    assert result["pause_stopping_time_ms"] == 70.0
    assert result["pause_to_escape_time_ms"] == 50.0
    assert result["pause_escape_status"] == "observed"


def test_pause_response_finds_escape_after_wind_window_without_changing_class():
    trial = _wind_trial()
    t = trial["t_rel"].to_numpy()
    speed = np.where(t < 180.0, 20.0, 2.0)
    speed[(t >= 320.0) & (t < 370.0)] = 120.0
    trial["speed"] = speed
    trial["speed_raw"] = speed
    result = classify_trial(trial)
    assert result["response_type"] == "NoResponse"
    assert np.isnan(result["distance_mm"])
    assert np.isnan(result["escape_reaction_time_ms"])
    assert result["pause_stopping_time_ms"] == 190.0
    assert result["pause_reaction_time_ms"] == 320.0
    assert result["pause_to_escape_time_ms"] == 130.0


@pytest.mark.parametrize("bad_time", [np.nan, 100.0])
def test_pause_response_does_not_bridge_bad_source_clock(bad_time):
    trial = _wind_trial()
    trial.loc[trial["t_rel"] == 110.0, "t_acquisition_rel"] = bad_time
    result = classify_trial(trial)
    assert result["pause_stopping_time_ms"] == 70.0
    assert np.isnan(result["pause_to_escape_time_ms"])
    assert result["pause_escape_status"] == "invalid_data"


def test_pause_response_without_escape_is_not_imputed():
    trial = _wind_trial()
    trial["speed_raw"] = np.where(trial["t_rel"] < 60.0, 20.0, 2.0)
    result = classify_trial(trial)
    assert result["pause_stopping_time_ms"] == 70.0
    assert np.isnan(result["pause_reaction_time_ms"])
    assert np.isnan(result["pause_to_escape_time_ms"])
    assert result["pause_escape_status"] == "no_escape"
    assert result["response_type"] == "PreWalk"
    assert np.isfinite(result["distance_mm"])


def test_pause_response_missing_clock_does_not_fall_back_to_host_speed():
    trial = _wind_trial().drop(columns="t_acquisition_rel")
    result = classify_trial(trial)
    assert np.isnan(result["pause_stopping_time_ms"])
    assert np.isnan(result["pause_reaction_time_ms"])
    assert result["pause_status"] == "missing_acquisition_clock"


def test_pause_response_keeps_first_stop_without_rearming_and_preserves_clock_sum():
    trial = _wind_trial()
    trial["t_acquisition_rel"] = trial["t_rel"] * 1.5
    result = classify_trial(trial)
    assert result["pause_stopping_time_ms"] == 90.0
    assert result["pause_reaction_time_ms"] == 180.0
    assert result["pause_to_escape_time_ms"] == 90.0
    assert result["pause_reaction_time_ms"] == result["pause_stopping_time_ms"] + result["pause_to_escape_time_ms"]


@pytest.mark.parametrize("early_escape", [False, True])
def test_pause_response_rejects_prestopped_and_escape_deceleration(early_escape):
    trial = _wind_trial()
    if early_escape:
        trial["speed_raw"] = np.where(trial["t_rel"] < 60.0, 120.0, 2.0)
        trial.loc[(trial["t_rel"] >= 120.0) & (trial["t_rel"] < 220.0), "speed_raw"] = 120.0
    else:
        trial["speed_raw"] = np.where(trial["t_rel"] < -60.0, 20.0, 2.0)
        trial.loc[(trial["t_rel"] >= 120.0) & (trial["t_rel"] < 220.0), "speed_raw"] = 120.0
    result = classify_trial(trial)
    assert np.isnan(result["pause_stopping_time_ms"])
    assert np.isnan(result["pause_reaction_time_ms"])
    assert result["pause_status"] in ("not_moving_at_wind", "escape_first", "no_preescape_stop")


def test_pause_response_rejects_missing_followup_and_exports_fields(tmp_path):
    from pipeline.io import export_summary_metrics
    trial = _wind_trial()
    trial.loc[trial["t_rel"] == 100.0, "speed_raw"] = np.nan
    labeled = label_trials(trial)
    assert labeled.pause_stopping_time_ms.eq(70.0).all()
    assert labeled.pause_to_escape_time_ms.isna().all()
    assert labeled.pause_escape_status.eq("invalid_data").all()
    labeled["global_trial_index"] = 1
    labeled["session_id"] = 1
    summary = pd.read_csv(export_summary_metrics(labeled, tmp_path / "summary.csv"))
    assert summary.pause_stopping_time_ms.iloc[0] == 70.0
    assert summary.pause_to_escape_time_ms.isna().all()
    assert summary.pause_escape_status.iloc[0] == "invalid_data"


def test_pause_response_gap_at_later_window_boundary_is_unobserved():
    trial = _wind_trial()
    trial["speed_raw"] = np.where(trial["t_rel"] < 60.0, 20.0, 2.0)
    # The stop is 70; later window ends at 320. The interval crossing its end
    # has no speed, even though there are observations on both sides.
    trial.loc[trial["t_rel"] == 320.0, "speed_raw"] = np.nan
    result = classify_trial(trial)
    assert result["pause_stopping_time_ms"] == 70.0
    assert result["pause_escape_status"] == "invalid_data"


def test_pause_response_corruption_before_stop_is_not_observed_absence():
    trial = _wind_trial()
    trial.loc[trial["t_rel"] == 60.0, "t_acquisition_rel"] = np.nan
    result = classify_trial(trial)
    assert np.isnan(result["pause_stopping_time_ms"])
    assert result["pause_status"] == "invalid_data"


def test_pause_response_escape_onset_must_strictly_exceed_threshold():
    trial = _wind_trial()
    trial.loc[(trial["t_rel"] >= 100.0) & (trial["t_rel"] < 120.0), "speed_raw"] = 10.0
    result = classify_trial(trial)
    assert result["pause_stopping_time_ms"] == 70.0
    # Averaged speed is exactly 10 at t=110, not a >10 response onset.
    assert result["pause_reaction_time_ms"] == 120.0


def test_pause_response_multimodal_first_sample_fallback_is_not_escape_first():
    trial = _wind_trial(wind_ms=-373.0)
    t = trial["t_rel"].to_numpy()
    speed = np.full_like(t, 20.0)
    speed[(t >= -253.0) & (t < -203.0)] = 120.0
    trial["speed_raw"] = speed

    first = compute_escape_latency(
        t, speed, stim_onset_t_rel=-373.0, trial_type="looming_wind",
        use_angular_onset_refinement=False,
    )
    assert first["latency_ms"] == t[0]  # no observed pre-burst low-speed onset

    result = classify_trial(trial)
    assert result["pause_status"] == "no_preescape_stop"
    assert np.isnan(result["pause_stopping_time_ms"])


def test_pause_response_keeps_genuine_ongoing_multimodal_burst_as_escape_first():
    trial = _wind_trial(wind_ms=-373.0)
    trial["speed_raw"] = 120.0
    result = classify_trial(trial)
    assert result["pause_status"] == "escape_first"
    assert np.isnan(result["pause_stopping_time_ms"])


@pytest.mark.parametrize("plateau", [10.0, np.nextafter(10.0, np.inf)])
def test_baseline_visual_escape_onset_tolerates_threshold_roundoff(plateau):
    t = np.array([-30.0, -20.0, -10.0, 0.0, 10.0])
    speed = np.array([20.0, plateau, 120.0, 2.0, 2.0])
    result = compute_escape_latency(
        t, speed, trial_type="baseline_visual",
        use_angular_onset_refinement=False,
    )
    assert result["latency_ms"] == -10.0


def test_pause_response_observed_prewind_onset_is_not_a_first_sample_fallback():
    trial = _wind_trial(wind_ms=-373.0)
    t = trial["t_rel"].to_numpy()
    speed = np.where(t < -393.0, 2.0, 20.0)
    speed[(t >= -353.0) & (t < -313.0)] = 120.0
    trial["speed_raw"] = speed
    result = classify_trial(trial)
    assert result["pause_status"] == "escape_first"
    assert np.isnan(result["pause_stopping_time_ms"])


def test_pause_response_and_legacy_stopping_have_different_escape_caps():
    trial = _wind_trial()
    trial.loc[trial["t_rel"] == 50.0, "speed"] = 2.0
    trial.loc[(trial["t_rel"] >= 60.0) & (trial["t_rel"] < 120.0), "speed"] = 120.0
    result = classify_trial(trial)
    assert result["response_type"] == "PreWalk"
    assert result["stillness_status"] == "no_preescape_stop"
    assert np.isnan(result["stillness_reaction_time_ms"])
    assert result["pause_status"] == "observed"
    assert result["pause_stopping_time_ms"] == 70.0
    assert result["pause_to_escape_time_ms"] == 50.0


@pytest.mark.parametrize("missing", ["short", "gap"])
def test_pause_response_unrecorded_stopping_window_is_not_absence(missing):
    trial = _wind_trial()
    if missing == "short":
        trial = trial.loc[trial["t_rel"] <= 40.0].copy()
    else:
        trial.loc[trial["t_rel"] >= 50.0, "t_acquisition_rel"] += 500.0
    result = classify_trial(trial)
    assert result["pause_status"] == "invalid_data"
    assert np.isnan(result["pause_stopping_time_ms"])


@pytest.mark.parametrize("speed,status", [(2.0, "not_moving_at_wind"), (10.0, "threshold_unresolved"), (120.0, "escape_first")])
def test_pause_response_later_corruption_preserves_resolved_ineligibility(speed, status):
    trial = _wind_trial(wind_ms=-373.0)
    trial["speed_raw"] = speed
    trial.loc[trial["t_rel"] == -273.0, "t_acquisition_rel"] = np.nan
    result = classify_trial(trial)
    assert result["pause_status"] == status
    assert np.isnan(result["pause_stopping_time_ms"])


@pytest.mark.parametrize("plateau", [10.0, np.nextafter(10.0, np.inf), 10.0 + 1e-9])
def test_pause_cohort_requires_strictly_above_threshold_history(plateau):
    trial = _wind_trial()
    trial.loc[(trial["t_rel"] >= -600.0) & (trial["t_rel"] < -400.0), "speed_raw"] = plateau
    result = classify_trial(trial)
    assert result["pause_status"] == "observed"  # valid local transition
    assert result["pause_stopping_time_ms"] == 70.0
    assert result["pause_baseline_status"] == "intermittent_moving"


def test_prestimulus_slow_level_is_not_a_pause_transition():
    trial = _wind_trial()
    # Earlier movement is retained as intermittent history, not strict PreWalk.
    trial.loc[(trial["t_rel"] >= -300.0) & (trial["t_rel"] < 120.0), "speed_raw"] = 8.0
    result = classify_trial(trial)
    assert result["response_type"] == "Escape"
    assert result["pause_baseline_status"] == "intermittent_moving"
    assert result["pause_status"] == "threshold_unresolved"
    assert np.isnan(result["pause_stopping_time_ms"])
    assert np.isnan(result["pause_to_escape_time_ms"])


@pytest.mark.parametrize("wind_ms", [0.0, -373.0])
@pytest.mark.parametrize("history", ["continuous", "interrupted", "quiet_at_wind", "equal", "quiet_history", "invalid_history", "missing_clock", "short"])
def test_wind_prewalk_uses_same_moving_eligibility_as_pause_rt(wind_ms, history):
    trial = _wind_trial(wind_ms=wind_ms)
    t = trial["t_acquisition_rel"] - wind_ms
    if history == "interrupted":
        trial.loc[(t >= -600.0) & (t < -400.0), "speed_raw"] = 2.0
    elif history == "quiet_at_wind":
        trial.loc[(t >= -100.0) & (t <= 0.0), "speed_raw"] = 2.0
    elif history == "equal":
        trial.loc[(t >= -600.0) & (t < -400.0), "speed_raw"] = 10.0
    elif history == "quiet_history":
        trial.loc[t <= 0.0, "speed_raw"] = 2.0
    elif history == "invalid_history":
        trial.loc[t == -500.0, "speed_raw"] = np.nan
    elif history == "missing_clock":
        trial = trial.drop(columns="t_acquisition_rel")
    elif history == "short":
        trial = trial.loc[t >= -500.0].copy()
    result = classify_trial(trial)
    expected = {
        "continuous": "continuous_moving", "interrupted": "intermittent_moving",
        "quiet_at_wind": "intermittent_moving", "equal": "intermittent_moving",
        "quiet_history": "stationary", "invalid_history": "unobserved",
        "missing_clock": "unobserved", "short": "unobserved",
    }
    assert result["pause_baseline_status"] == expected[history]
    assert result["stillness_baseline_status"] == expected[history]
    strict = result["pause_baseline_status"] == "continuous_moving"
    assert strict == (history == "continuous")
    moving = history in {"continuous", "interrupted", "equal"}
    assert result["pause_moving_eligible"] == moving
    assert (result["response_type"] == "PreWalk") == moving
    from cercus.analysis.reaction_time import select_escape_latency
    selected = select_escape_latency(pd.DataFrame([{**result, "type": trial["type"].iloc[0]}])).iloc[0]
    if moving:
        assert selected == result["pause_reaction_time_ms"] == 120.0
    else:
        assert selected == result["escape_reaction_time_ms"] == 120.0
    assert result["escape_reaction_time_ms"] == 120.0
    assert np.isfinite(result["distance_mm"])


def test_strict_prewalk_does_not_require_observed_stop():
    trial = _wind_trial()
    trial["speed_raw"] = 20.0
    result = classify_trial(trial)
    assert result["pause_baseline_status"] == "continuous_moving"
    assert result["pause_status"] == "no_preescape_stop"
    assert result["response_type"] == "PreWalk"
    assert np.isnan(result["pause_stopping_time_ms"])


def test_prewind_interruption_keeps_local_stop_but_excludes_paper_moving_cohort():
    trial = _wind_trial()
    trial.loc[(trial["t_rel"] >= -600.0) & (trial["t_rel"] < -400.0), "speed_raw"] = 2.0
    result = classify_trial(trial)
    assert result["pause_status"] == "observed"
    assert result["pause_to_escape_time_ms"] == 50.0
    assert result["pause_baseline_status"] == "intermittent_moving"


def test_preescape_retains_negative_lead_time_without_short_escape_flag():
    trial = _wind_trial(wind_ms=-373.0)
    trial.loc[(trial["t_rel"] >= -393.0) & (trial["t_rel"] < -303.0), "speed"] = 2.0
    trial.loc[(trial["t_rel"] >= -383.0) & (trial["t_rel"] < -303.0), "speed"] = 120.0
    result = classify_trial(trial)
    assert result["response_type"] == "PreEscape"
    assert result["escape_reaction_time_ms"] == -10.0
    assert not result["short_rt"]
    assert np.isnan(result["stillness_reaction_time_ms"])


@pytest.mark.parametrize("reference", [0.0, -3.0])
def test_moving_fraction_uses_duration_and_causal_boundary_support(reference):
    from cercus.core.kinematics.latency import _movement_history_metrics

    t = np.arange(-1040.0, 21.0, 10.0)
    averaged = np.full_like(t, 20.0)
    averaged[(t >= -600.0) & (t < -400.0)] = 2.0
    status, fraction, at_reference = _movement_history_metrics(t, averaged, reference, 1000.0, 30.0)
    assert status == "intermittent_moving"
    assert fraction == pytest.approx(0.8)
    assert at_reference == 20.0
    # The last observation BEFORE the left boundary covers its first 7 ms,
    # not the first observation AFTER that boundary.
    t = np.r_[-1040.0, -1020.0, -1003.0, -993.0, np.arange(-990.0, 21.0, 10.0)]
    averaged = np.full_like(t, 20.0)
    averaged[t == -1003.0] = 2.0
    status, fraction, _ = _movement_history_metrics(t, averaged, 0.0, 1000.0, 30.0)
    assert status == "continuous_moving"  # original observation-based strict status
    assert fraction == pytest.approx(0.993)


@pytest.mark.parametrize("minimum,eligible", [
    (0.8, True), (np.nextafter(0.8, np.inf), True), (0.801, False),
    (-0.1, False), (1.1, False), (np.nan, False),
])
def test_moving_fraction_cutoff_is_configurable_and_roundoff_safe(monkeypatch, minimum, eligible):
    from cercus.config import get_thresholds

    trial = _wind_trial()
    # 210 ms raw dip gives 200 ms below threshold after causal averaging.
    trial.loc[trial["t_rel"].between(-600.0, -400.0), "speed_raw"] = 2.0
    monkeypatch.setitem(get_thresholds().prewalk._data, "min_moving_fraction", minimum)
    result = classify_trial(trial)
    assert result["pause_moving_fraction"] == pytest.approx(0.8)
    assert result["pause_moving_eligible"] == eligible
    assert (result["response_type"] == "PreWalk") == eligible
    assert result["pause_stopping_time_ms"] == 70.0  # local measurement independent


@pytest.mark.parametrize("speed,eligible", [(10.0, False), (10.0 + 1e-9, False), (12.0, True)])
def test_moving_reference_threshold_does_not_borrow_stopping_resolution_margin(speed, eligible):
    from cercus.analysis.reaction_time import select_escape_latency

    trial = _wind_trial()
    trial.loc[trial["t_rel"] <= 0.0, "speed_raw"] = speed
    result = classify_trial(trial)
    assert result["pause_moving_eligible"] == eligible
    assert result["pause_status"] == "threshold_unresolved"
    assert np.isnan(result["pause_reaction_time_ms"])
    selected = select_escape_latency(pd.DataFrame([{**result, "type": "baseline_wind"}])).iloc[0]
    if eligible:
        assert np.isnan(selected)
    else:
        assert selected == 120.0


def test_summary_export_keeps_fraction_precision_at_cutoff(tmp_path):
    from pipeline.io import export_summary_metrics

    labeled = label_trials(_wind_trial())
    labeled["global_trial_index"] = 1
    labeled["session_id"] = 1
    labeled["pause_moving_fraction"] = 0.149
    labeled["pause_moving_eligible"] = False
    summary = pd.read_csv(export_summary_metrics(labeled, tmp_path / "fraction.csv"))
    assert summary["pause_moving_fraction"].iloc[0] == 0.149
    assert not summary["pause_moving_eligible"].iloc[0]
