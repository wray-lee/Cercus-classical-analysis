"""T3: 跨范式连线图 — summarize_subjects 口径 + dumbbell 出图结构。"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cercus.visualization.paradigm import plot_paradigm_dumbbell, summarize_subjects


def _synthetic() -> pd.DataFrame:
    """2 范式 × 若干动物 × 若干 trial 的 trial 级表。"""
    rows = []
    rng = np.random.default_rng(1)
    for paradigm, rate in (("bv", 0.8), ("-373 30°", 0.4)):
        for s in range(5):
            for t in range(20):
                esc = rng.random() < rate
                rows.append({
                    "paradigm": paradigm,
                    "subject_id": f"subj{s}",
                    "global_trial_index": t,
                    "response_type": "Escape" if esc else "NoResponse",
                    "reaction_time_ms": float(rng.normal(50, 10)) if esc else np.nan,
                    "distance_mm": float(rng.normal(40, 5)) if esc else np.nan,
                })
    return pd.DataFrame(rows)


def test_summarize_subjects_semantics():
    subj = summarize_subjects(_synthetic())
    assert list(subj.columns) == [
        "paradigm", "subject_id", "n_trials", "response_rate", "response_group_rate",
        "rt_mean", "dist_mean", "rt_observed", "rt_trials", "rt_missing",
    ]
    bv = subj[subj["paradigm"] == "bv"]
    # 每动物 20 trial；响应率∈(0,1]；RT 均值仅在逃逸 trial 上 → 有限
    assert (bv["n_trials"] == 20).all()
    assert bv["response_rate"].between(0, 1).all()
    assert bv["response_group_rate"].between(0, 1).all()
    assert bv["rt_mean"].notna().all()
    assert (subj["rt_observed"] + subj["rt_missing"]).eq(subj["rt_trials"]).all()
    assert subj["rt_missing"].eq(0).all()  # NoResponse is not a missing response endpoint.
    assert 0.2 < (subj[subj["paradigm"] == "bv"]["response_rate"].mean()) <= 1.0
    assert (subj[subj["paradigm"] == "-373 30°"]["response_rate"].mean()
            < subj[subj["paradigm"] == "bv"]["response_rate"].mean())


def test_dumbbell_structure():
    fig = plot_paradigm_dumbbell(_synthetic())
    assert len(fig.axes) == 3
    for ax in fig.axes:
        # 每范式一个黑方块（均值）+ 竖线；2 范式 → 2 方块
        sq = [c for c in ax.collections if c.get_offsets().shape[0] > 0
              and c.get_paths() and c.get_paths()[0].vertices.shape[0] == 5]
        assert len(sq) >= 1
    plt.close(fig)


def _synthetic_by_class() -> pd.DataFrame:
    """含 PreEscape / PreWalk 的 trial 级表（范式 × 行为类对比用）。"""
    rows = []
    rng = np.random.default_rng(2)
    classes3 = ("Escape", "PreEscape", "PreWalk")
    for paradigm in ("bv", "-373 30°", "-308 36°"):
        for s in range(4):
            for t in range(12):
                rt = float(rng.normal(40, 10))
                rows.append({
                    "paradigm": paradigm, "subject_id": f"subj{s}",
                    "global_trial_index": t,
                    "response_type": classes3[t % 3],
                    "reaction_time_ms": rt, "distance_mm": rt + 30.0,
                })
    return pd.DataFrame(rows)


def test_rt_dist_by_class():
    from cercus.visualization.paradigm import (
        plot_paradigm_rt_dist, summarize_subjects_by_class,
    )
    subj = summarize_subjects_by_class(_synthetic_by_class())
    # 三类有 burst 行为（含 PreWalk）都保留；NoResponse 无 rt/dist 被排除
    assert subj["response_type"].isin(["Escape", "PreEscape", "PreWalk"]).all()
    assert set(subj["response_type"]) == {"Escape", "PreEscape", "PreWalk"}
    # 每 (paradigm,subject,class) = 4 trial → 中位数逐动物一行
    assert len(subj) == 3 * 4 * 3 and (subj["n"] == 4).all()
    fig = plot_paradigm_rt_dist(_synthetic_by_class())
    assert len(fig.axes) == 2
    plt.close(fig)


def test_cross_paradigm_rt_uses_escape_not_stopping_latency():
    from cercus.visualization.paradigm import summarize_subjects_by_class

    df = _synthetic_by_class()
    df["escape_reaction_time_ms"] = 120.0
    df["stillness_reaction_time_ms"] = 30.0
    df.loc[df["response_type"] == "PreWalk", "reaction_time_ms"] = 30.0
    assert (summarize_subjects_by_class(df)["rt_med"] == 120.0).all()
    assert (summarize_subjects(df)["rt_mean"] == 120.0).all()


def test_cross_paradigm_rt_selects_state_specific_endpoints_without_imputation():
    from cercus.visualization.paradigm import summarize_subjects_by_class

    df = pd.DataFrame({
        "paradigm": ["bw"] * 8,
        "subject_id": list("abcdefgh"),
        "global_trial_index": range(8),
        "response_type": ["PreWalk", "PreWalk", "Escape", "Escape",
                          "Escape", "PreEscape", "NoResponse", "Escape"],
        "type": ["baseline_wind"] * 5 + ["looming_wind", "baseline_wind", "baseline_visual"],
        "pause_baseline_status": ["continuous_moving", "continuous_moving",
                                  "intermittent_moving", "unobserved", "stationary",
                                  "continuous_moving", "continuous_moving", "not_applicable"],
        "escape_reaction_time_ms": [90.0] * 5 + [-40.0, 90.0, 90.0],
        "pause_reaction_time_ms": [120.0, np.nan] + [120.0] * 6,
        "distance_mm": [10.0] * 8,
    })
    expected = pd.Series([120.0, np.nan, 90.0, 90.0, 90.0, -40.0, np.nan, 90.0],
                         index=list("abcdefgh"), name="rt_mean")
    # Replicated frame rows must not add trials or borrow another endpoint.
    frames = pd.concat([df, df], ignore_index=True)
    summary = summarize_subjects(frames).set_index("subject_id")
    pd.testing.assert_series_equal(summary["rt_mean"], expected, check_names=False)
    assert summary["n_trials"].eq(1).all()
    assert summary["dist_mean"].eq(10.0).all()
    by_class = summarize_subjects_by_class(frames).set_index("subject_id")
    pd.testing.assert_series_equal(by_class["rt_med"], expected.drop("g"), check_names=False)

    legacy = df.drop(columns=["type", "pause_baseline_status", "pause_reaction_time_ms"])
    expected = df.set_index("subject_id")["escape_reaction_time_ms"].copy()
    expected.loc["g"] = np.nan
    pd.testing.assert_series_equal(
        summarize_subjects(legacy).set_index("subject_id")["rt_mean"],
        expected, check_names=False,
    )


def test_shared_rt_selector_uses_final_class_without_missing_value_fallback():
    from cercus.analysis.reaction_time import select_escape_latency

    trials = pd.DataFrame({
        "type": ["baseline_wind"] * 6 + ["looming_wind", "baseline_wind"],
        "response_type": ["PreWalk"] * 6 + ["PreEscape", "NoResponse"],
        "pause_baseline_status": ["intermittent_moving", "intermittent_moving",
                                  "continuous_moving", "continuous_moving",
                                  "continuous_moving", "stationary",
                                  "intermittent_moving", "continuous_moving"],
        "pause_moving_eligible": [True, False, np.nan, False, True, False, True, True],
        "pause_reaction_time_ms": [120.0] * 4 + [np.nan, 120.0, 120.0, 120.0],
        "escape_reaction_time_ms": [90.0] * 6 + [-40.0, 90.0],
    })
    expected = pd.Series([120.0, 120.0, 120.0, 120.0, np.nan, 120.0, -40.0, np.nan])
    pd.testing.assert_series_equal(select_escape_latency(trials), expected, check_names=False)

    no_baseline = trials.drop(columns="pause_baseline_status")
    pd.testing.assert_series_equal(
        select_escape_latency(no_baseline), expected, check_names=False,
    )

    no_causal = trials.drop(columns="pause_reaction_time_ms")
    pd.testing.assert_series_equal(
        select_escape_latency(no_causal), expected.mask(expected.eq(120.0)), check_names=False,
    )


def test_rt_cohort_is_final_classifier_class_not_history_or_pair_coverage():
    from cercus.analysis.reaction_time import select_escape_latency
    from cercus.visualization.paradigm import summarize_subjects_by_class
    from cercus.visualization.behavior import plot_reaction_distance_panel

    trials = pd.DataFrame({
        "paradigm": ["bw"] * 6,
        "subject_id": ["a"] * 6,
        "global_trial_id": range(6), "global_trial_index": range(6),
        "type": ["baseline_wind"] * 3 + ["looming_wind"] + ["baseline_wind"] * 2,
        "response_type": ["PreWalk"] * 3 + ["PreEscape", "NoResponse", "Escape"],
        "pause_moving_eligible": [True, False, True, True, True, False],
        "pause_baseline_status": ["intermittent_moving"] * 6,
        "pause_reaction_time_ms": [120.0, 140.0, np.nan, 160.0, 180.0, 200.0],
        "pause_stopping_time_ms": [70.0, 80.0, np.nan, 90.0, 100.0, 110.0],
        "pause_to_escape_time_ms": [50.0, 60.0, np.nan, 70.0, 80.0, 90.0],
        "escape_reaction_time_ms": [90.0] * 3 + [-40.0, 90.0, 90.0],
        "distance_mm": [10.0] * 6,
    })
    # Deliberately contradictory history metadata must not reclassify trials.
    expected = pd.Series([120.0, 140.0, np.nan, -40.0, np.nan, 90.0])
    pd.testing.assert_series_equal(select_escape_latency(trials), expected, check_names=False)
    summary = summarize_subjects_by_class(pd.concat([trials, trials])).set_index("response_type")
    assert summary.loc["PreWalk", "n"] == 3
    assert summary.loc["PreWalk", "rt_observed"] == 2
    assert summary.loc["PreWalk", "rt_missing"] == 1
    assert summary.loc["PreWalk", "rt_med"] == 130.0
    fig = plot_reaction_distance_panel(trials)
    text = "\n".join(t.get_text() for t in fig.axes[2].texts)
    assert "Pure: PreWalk 3; paired 2/3" in text
    assert "Multimodal" not in text
    assert "PreWalk" in fig.axes[2].get_title()
    plt.close(fig)


def test_grouped_subject_median_preserves_raw_endpoint_and_missingness(monkeypatch):
    from cercus.config import get_config
    from cercus.visualization.paradigm import (
        plot_paradigm_rt_dist, summarize_subjects_by_class,
    )

    trials = pd.DataFrame({
        "paradigm": ["bw"] * 4, "subject_id": ["a"] * 4,
        "global_trial_index": range(4), "type": ["baseline_wind"] * 4,
        "response_type": ["Escape", "PreWalk", "PreWalk", "NoResponse"],
        "escape_reaction_time_ms": [90.0] * 4,
        "pause_reaction_time_ms": [np.nan, 120.0, np.nan, 100.0],
        "distance_mm": [10.0, 20.0, 30.0, np.nan],
    })
    before = trials.copy(deep=True)
    settings = get_config().analysis.response_grouping._data
    for merge in (True, False):
        monkeypatch.setitem(settings, "merge_prewalk", merge)
        summary = summarize_subjects_by_class(
            pd.concat([trials, trials]), grouped=True,
        ).set_index("response_group")
        if merge:
            assert list(summary.index) == ["Escape"]
            assert summary.loc["Escape", "n"] == 3
            assert summary.loc["Escape", "rt_med"] == 105.0
            assert summary.loc["Escape", "dist_med"] == 20.0
            assert summary.loc["Escape", "rt_observed"] == 2
            assert summary.loc["Escape", "rt_missing"] == 1
        else:
            assert set(summary.index) == {"Escape", "PreWalk"}
            assert summary.loc["PreWalk", "n"] == 2
            assert summary.loc["PreWalk", "rt_med"] == 120.0
            assert summary.loc["PreWalk", "rt_missing"] == 1
        fig = plot_paradigm_rt_dist(trials)
        labels = [t.get_text() for t in fig.legends[0].get_texts()]
        assert ("PreWalk" in labels) == (not merge)
        plt.close(fig)
    pd.testing.assert_frame_equal(trials, before)


def test_partial_causal_schema_never_borrows_centered_rt():
    from cercus.analysis.reaction_time import select_escape_latency

    trials = pd.DataFrame({
        "type": ["baseline_wind"], "response_type": ["PreWalk"],
        "escape_reaction_time_ms": [90.0],
        "pause_stopping_time_ms": [70.0], "pause_to_escape_time_ms": [50.0],
    })
    assert select_escape_latency(trials).isna().all()
    legacy = trials.drop(columns=["pause_stopping_time_ms", "pause_to_escape_time_ms"])
    assert select_escape_latency(legacy).iloc[0] == 90.0


def test_plot_coverage_uses_classifier_trials_and_finite_observed_subjects():
    from cercus.visualization.paradigm import plot_paradigm_rt_dist, summarize_subjects_by_class

    trials = pd.DataFrame({
        "paradigm": ["bw"] * 4, "subject_id": ["a", "b", "c", "d"],
        "global_trial_index": [1] * 4,
        "type": ["baseline_wind"] * 4,
        "response_type": ["PreWalk"] * 3 + [None],
        "pause_reaction_time_ms": [120.0, np.inf, np.nan, 100.0],
        "escape_reaction_time_ms": [90.0] * 4,
        "distance_mm": [10.0] * 4,
    })
    before = trials.copy(deep=True)
    frames = pd.concat([trials, trials], ignore_index=True)
    summary = summarize_subjects(frames)
    assert summary.rt_trials.sum() == 3
    assert summary.rt_observed.sum() == 1
    assert summary.rt_missing.sum() == 2
    by_class = summarize_subjects_by_class(frames)
    assert by_class.n.sum() == 3
    assert by_class.rt_observed.sum() == 1
    assert by_class.rt_missing.sum() == 2
    for plot, axis_index in ((plot_paradigm_rt_dist, 0), (plot_paradigm_dumbbell, 1)):
        fig = plot(frames)
        text = "\n".join(t.get_text() for t in fig.axes[axis_index].texts)
        assert "N=3; observed=1; missing=2" in text
        assert "subjects=1/3" in text
        plt.close(fig)
    pd.testing.assert_frame_equal(trials, before)


def test_rt_dist_ticks_and_offsets():
    from cercus.visualization.paradigm import (
        plot_paradigm_rt_dist, _compute_zero_anchored_ticks, _compute_distance_ticks,
    )
    vals_rt = np.array([-2200.0, 100.0])
    (t_min, t_max), ticks = _compute_zero_anchored_ticks(vals_rt)
    assert 0.0 in ticks
    assert t_min <= vals_rt.min()
    assert t_max >= vals_rt.max()

    vals_dist = np.array([5.0, 320.0])
    (d_min, d_max), d_ticks = _compute_distance_ticks(vals_dist)
    assert d_min == 0.0
    assert d_ticks[0] == 0.0
    assert d_max >= vals_dist.max()

    fig = plot_paradigm_rt_dist(_synthetic_by_class())
    assert len(fig.axes) == 2
    rt_ax, dist_ax = fig.axes
    assert rt_ax.get_ylabel() == "Selected timing endpoint vs reference (ms)"
    assert 0.0 in rt_ax.get_yticks()
    assert 0.0 in dist_ax.get_yticks()
    plt.close(fig)
