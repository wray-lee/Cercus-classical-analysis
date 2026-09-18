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
    assert 0.0 in rt_ax.get_yticks()
    assert 0.0 in dist_ax.get_yticks()
    plt.close(fig)
