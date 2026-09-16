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
