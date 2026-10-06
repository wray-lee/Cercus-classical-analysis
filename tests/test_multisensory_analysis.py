"""MS parent-directory mode: coverage, classifier endpoints and subject bootstrap."""
import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from cercus.analysis import multisensory as ms


def _pair(path, subject="animal"):
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{subject}_session_1_events.csv").write_text(
        "event_name,timestamp,global_trial_id,details\n", encoding="utf-8"
    )
    (path / f"{subject}_session_1_kinematics.csv").write_text(
        "sys_time,ard_time,dx,dy,dz,stim_state,global_trial_id\n", encoding="utf-8"
    )


def _trials():
    rows = []
    for paradigm, kind, ttc in (("bv", "baseline_visual", np.nan),
                                ("bw", "baseline_wind", np.nan),
                                ("-225 48°", "looming_wind", -225)):
        for subject in ("a", "b"):
            for index, label in enumerate(("Escape", "PreWalk", "PreEscape", "NoResponse")):
                rows.append(dict(paradigm=paradigm, subject_id=subject,
                    global_trial_index=index, type=kind, target_ttc_ms=ttc,
                    lv_ratio_ms=120 if kind != "baseline_wind" else np.nan,
                    init_half_angle_deg=2 if kind != "baseline_wind" else np.nan,
                    response_type=label, v_max=100.,
                    escape_reaction_time_ms=[40., 50., -20., np.nan][index],
                    pause_reaction_time_ms=[np.nan, np.nan, np.nan, np.nan][index],
                    distance_mm=[10., 20., 30., np.nan][index],
                    prestim_status="stationary" if index == 0 else "unobserved"))
    return pd.DataFrame(rows)


def test_discovery_keeps_missing_and_excludes_all(tmp_path):
    _pair(tmp_path / "bv")
    (tmp_path / "+200").mkdir()
    _pair(tmp_path / "all")
    (tmp_path / "all" / "animal_session_1_events.csv").write_text(
        "session_id,trial_id,time_ms,event_type,event_value\n", encoding="utf-8"
    )
    tasks, coverage, ignored = ms.discover_ms_inputs(tmp_path)
    assert len(tasks) == 1
    assert tasks[0][0] == "bv"
    statuses = coverage.set_index("paradigm")["status"]
    assert statuses["bv"] == "pending"
    assert statuses["+200"] == "empty"
    assert statuses["bw"] == "missing"
    assert "all" not in set(coverage.paradigm)
    assert any(row["paradigm"] == "all" for row in ignored)


def test_unsupported_pair_is_reported(tmp_path):
    _pair(tmp_path / "bw")
    (tmp_path / "bw" / "animal_session_1_kinematics.csv").write_text(
        "session_id,trial_id,time_ms,velocity\n", encoding="utf-8"
    )
    tasks, coverage, ignored = ms.discover_ms_inputs(tmp_path)
    assert not tasks
    assert coverage.set_index("paradigm").loc["bw", "status"] == "unsupported"
    assert "missing" in coverage.set_index("paradigm").loc["bw", "message"]


def test_summary_preserves_raw_endpoint_and_missing_cohort():
    raw = _trials()
    trials, subjects, responses = ms.summarize_ms_trials(raw)
    pd.testing.assert_series_equal(trials.response_type, raw.response_type)
    assert (trials.response_group[trials.response_type == "PreWalk"] == "Escape").all()
    wind = subjects[subjects.modality != "visual"]
    assert (wind.rt_trials == 3).all()  # Escape + PreWalk + PreEscape per subject
    assert (wind.rt_observed == 2).all()
    assert (wind.rt_missing == 1).all()
    assert (subjects.escape_probability == .5).all()
    assert (subjects.burst_probability == .75).all()
    assert (subjects.prestim_observed == 1).all()
    assert set(responses.response_type) == {"Escape", "PreWalk", "PreEscape", "NoResponse"}
    assert trials.loc[trials.response_type == "NoResponse", "selected_rt_ms"].isna().all()
    assert trials.loc[(trials.type == "looming_wind") & (trials.response_type == "PreWalk"), "selected_rt_ms"].isna().all()


def test_enhancement_uses_equal_subject_weights_and_missing_baseline():
    _, subjects, _ = ms.summarize_ms_trials(_trials())
    # Unequal trial totals must not change the subject mean.
    subjects.loc[subjects.subject_id == "a", "n_trials"] = 100
    comparisons = ms.compare_ms_probabilities(subjects)
    row = comparisons.iloc[0]
    assert row.status == "available"
    assert row.ms_probability == row.visual_probability == row.wind_probability == .75
    assert row.independence_probability == .9375
    assert row.delta_best == 0
    assert row.delta_independence == -.1875
    assert row.probability_endpoint == "any_burst"
    assert row.delta_best_ci_low == row.delta_best_ci_high == 0
    missing = ms.compare_ms_probabilities(subjects[subjects.modality != "wind"])
    assert missing.iloc[0].status == "missing_baseline"
    assert pd.isna(missing.iloc[0].delta_best)


def test_visual_controls_must_match_looming_parameters():
    _, subjects, _ = ms.summarize_ms_trials(_trials())
    subjects.loc[subjects.modality == "visual", "lv_ratio_ms"] = 240
    comparison = ms.compare_ms_probabilities(subjects).iloc[0]
    assert comparison.status == "missing_baseline"
    assert pd.isna(comparison.delta_best)


def test_ms_aggregation_reports_subject_failures(tmp_path, monkeypatch):
    _pair(tmp_path / "bv", "good")
    _pair(tmp_path / "bv", "bad")
    raw = _trials()
    def process(task):
        if task[1] == "bad":
            return pd.DataFrame()
        return raw[(raw.paradigm == "bv") & (raw.subject_id == "a")].assign(subject_id="good")
    monkeypatch.setattr(ms, "_process_subject", process)
    trials, coverage, meta = ms.aggregate_ms_trials(tmp_path, workers=1)
    assert len(trials) == 4
    row = coverage.set_index("paradigm").loc["bv"]
    assert row.status == "partial"
    assert row.n_subjects == 1 and row.n_subjects_failed == 1
    assert meta["failed_subjects"] == [{"paradigm": "bv", "subject_id": "bad"}]


def test_cli_ms_routes_to_analysis(tmp_path, monkeypatch):
    from cercus.cli.app import app
    seen = {}
    def run(input_dir, output_dir, workers=None):
        seen.update(input=input_dir, output=output_dir, workers=workers)
    monkeypatch.setattr(ms, "run_ms_analysis", run)
    result = CliRunner().invoke(app, ["ms", "--input", str(tmp_path),
                                     "--output", str(tmp_path / "out"), "--workers", "1"])
    assert result.exit_code == 0, result.output
    assert seen == dict(input=tmp_path, output=tmp_path / "out", workers=1)


@pytest.mark.parametrize("second_id", ["1", "01"])
def test_duplicate_session_names_are_rejected(tmp_path, second_id):
    _pair(tmp_path / "bv" / "first")
    second = tmp_path / "bv" / "second"
    _pair(second)
    if second_id != "1":
        for kind in ("events", "kinematics"):
            (second / f"animal_session_1_{kind}.csv").rename(second / f"animal_session_{second_id}_{kind}.csv")
    tasks, coverage, _ = ms.discover_ms_inputs(tmp_path)
    assert not tasks
    row = coverage.set_index("paradigm").loc["bv"]
    assert row.status == "unsupported"
    assert "duplicate session" in row.message


def test_excluded_session_copies_do_not_reject_valid_subject(tmp_path):
    _pair(tmp_path / "bv")
    for folder in ("garbage", "pilot", "_backup"):
        _pair(tmp_path / "bv" / folder)
    tasks, coverage, _ = ms.discover_ms_inputs(tmp_path)
    assert len(tasks) == 1
    assert coverage.set_index("paradigm").loc["bv", "status"] == "pending"


def test_probability_reference_includes_preescape_and_ignores_merge(monkeypatch):
    from cercus.analysis.response_groups import derive_response_group
    raw = _trials()
    merged, subjects_merged, _ = ms.summarize_ms_trials(raw)
    monkeypatch.setattr(ms, "response_group_series", lambda frame: derive_response_group(frame.response_type, False))
    split, subjects_split, _ = ms.summarize_ms_trials(raw)
    assert (subjects_merged.escape_probability == .5).all()
    assert (subjects_split.escape_probability == .25).all()
    pd.testing.assert_frame_equal(ms.compare_ms_probabilities(subjects_merged),
                                  ms.compare_ms_probabilities(subjects_split))
    pd.testing.assert_series_equal(merged.selected_rt_ms, split.selected_rt_ms)


def test_missing_angle_matching_is_explicit():
    raw = _trials()
    raw["init_half_angle_deg"] = np.nan
    _, subjects, _ = ms.summarize_ms_trials(raw)
    comparison = ms.compare_ms_probabilities(subjects).iloc[0]
    assert comparison.status == "available"
    assert comparison.visual_match_status == "lv_ratio_only_angle_unrecorded"
    subjects.loc[subjects.modality == "visual", "init_half_angle_deg"] = 2
    assert ms.compare_ms_probabilities(subjects).iloc[0].status == "missing_baseline"


def test_plot_keeps_missing_patterns_and_non_straddling_ci():
    import matplotlib.pyplot as plt
    from cercus.visualization.multisensory import plot_ms_overview, plot_ms_enhancement
    _, subjects, _ = ms.summarize_ms_trials(_trials())
    coverage = pd.DataFrame(dict(paradigm=["bv", "bw", "-225 48°", "+200"],
                                 status=["available", "available", "available", "empty"]))
    overview = plot_ms_overview(subjects, coverage)
    assert all(len(ax.get_xticklabels()) == 4 for ax in overview.axes)
    assert all(any(text.get_text() == "empty" for text in ax.texts) for ax in overview.axes)
    comparisons = ms.compare_ms_probabilities(subjects)
    comparisons["delta_best_ci_low"] = .1
    comparisons["delta_best_ci_high"] = .2
    figure = plot_ms_enhancement(comparisons)
    figure.canvas.draw()
    empty = plot_ms_enhancement(comparisons.iloc[:0])
    empty.canvas.draw()
    plt.close("all")


@pytest.mark.parametrize("lv_pair", [(120, 240), (120.00001, 120.00002)])
def test_overview_keeps_different_looming_settings_separate(lv_pair):
    import matplotlib.pyplot as plt
    from cercus.visualization.multisensory import _overview_categories, plot_ms_enhancement
    _, subjects, _ = ms.summarize_ms_trials(_trials())
    one = subjects[subjects.modality == "multisensory"].copy()
    one["lv_ratio_ms"] = lv_pair[0]
    two = one.copy()
    two["lv_ratio_ms"] = lv_pair[1]
    combined = pd.concat([one, two])
    categories = _overview_categories(combined, ["-225 48°"])
    assert len(categories) == 2
    assert len({cat["label"] for cat in categories}) == 2
    assert all(len(cat["rows"]) == 2 for cat in categories)
    figure = plot_ms_enhancement(ms.compare_ms_probabilities(combined))
    assert all(len({tick.get_text() for tick in ax.get_xticklabels()}) == 2 for ax in figure.axes)
    plt.close(figure)


def test_empty_parent_exports_coverage(tmp_path):
    output = tmp_path.parent / (tmp_path.name + "_ms_output")
    meta = ms.run_ms_analysis(tmp_path, output, workers=1)
    assert meta["status"] == "no_data"
    assert meta["vmax_threshold"] is None
    assert pd.read_csv(output / "ms_summary.csv").empty
    assert set(pd.read_csv(output / "ms_coverage.csv").status) == {"missing"}
    assert (output / "ms_overview.svg").is_file()


def test_workers_and_output_boundary(tmp_path):
    with pytest.raises(ValueError, match="workers"):
        ms.aggregate_ms_trials(tmp_path, workers=0)
    with pytest.raises(ValueError, match="output"):
        ms.run_ms_analysis(tmp_path, tmp_path / "bv", workers=1)
