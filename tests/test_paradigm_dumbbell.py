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
        "paradigm", "subject_id", "n_trials", "response_rate", "rt_mean", "dist_mean",
    ]
    bv = subj[subj["paradigm"] == "bv"]
    # 每动物 20 trial；响应率∈(0,1]；RT 均值仅在逃逸 trial 上 → 有限
    assert (bv["n_trials"] == 20).all()
    assert bv["response_rate"].between(0, 1).all()
    assert bv["rt_mean"].notna().all()
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
    expected = pd.Series([120.0, np.nan, np.nan, np.nan, 90.0, -40.0, np.nan, 90.0],
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
