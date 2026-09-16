"""
Cercus Framework — Cross-Paradigm Dumbbell Plots (full mode)
============================================================
实验室传统连线图（cf. Frontiers fphys.2023.1153913 Fig 2）：
点 = 单只动物均值，竖线连 black square = 范式均值（± SD），横轴 = 范式梯度。
范式间为 between-subject，故不画跨范式折线——每范式一簇。
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cercus.analysis.full import _paradigm_sort_key
from cercus.constants.colors import NPG_PALETTE

log = logging.getLogger(__name__)

_ESCAPE_CLASSES = ("Escape", "PreEscape")


def summarize_subjects(df: pd.DataFrame) -> pd.DataFrame:
    """Trial-level full-mode frame → per-(paradigm, subject) summary.

    Columns: paradigm, subject_id, n_trials, response_rate, rt_mean, dist_mean.
    ``response_rate`` = (Escape+PreEscape)/all trials; RT/distance 均值仅在逃逸 trial 上。
    """
    trial = (
        df.groupby(["paradigm", "subject_id", "global_trial_index"])
        .agg(
            response_type=("response_type", "first"),
            rt=("reaction_time_ms", "first"),
            dist=("distance_mm", "first"),
        )
        .reset_index()
    )
    trial["is_escape"] = trial["response_type"].isin(_ESCAPE_CLASSES)
    out = (
        trial.groupby(["paradigm", "subject_id"])
        .agg(
            n_trials=("global_trial_index", "count"),
            response_rate=("is_escape", "mean"),
            rt_mean=("rt", "mean"),
            dist_mean=("dist", "mean"),
        )
        .reset_index()
    )
    return out


def plot_paradigm_dumbbell(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (7.5, 3.2),
) -> plt.Figure:
    """3-panel dumbbell: response rate / RT / distance across paradigms.

    输入 = full 模式的 trial 级聚合 df（需含 paradigm / subject_id 列）。
    """
    subj = summarize_subjects(df)
    paradigms = sorted(subj["paradigm"].unique(), key=_paradigm_sort_key)
    n_p = len(paradigms)
    colors = {p: NPG_PALETTE[i % len(NPG_PALETTE)] for i, p in enumerate(paradigms)}

    panels = (
        ("response_rate", "Escape probability", "P(Escape+PreEscape)"),
        ("rt_mean", "Reaction time", "stimulus-anchored RT (ms)"),
        ("dist_mean", "Escape distance", "distance (mm)"),
    )

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    rng = np.random.default_rng(42)
    for ax, (col, title, ylab) in zip(axes, panels):
        for i, p in enumerate(paradigms):
            vals = subj.loc[subj["paradigm"] == p, col].dropna().values
            if len(vals) == 0:
                continue
            mean, sd = float(np.mean(vals)), float(np.std(vals, ddof=1))
            c = colors[p]
            # 个体点（微横向 jitter 防重叠）→ 竖线 → 均值黑方块 ± SD
            ax.scatter(
                i + rng.uniform(-0.09, 0.09, len(vals)), vals,
                s=16, c=c, alpha=0.85, edgecolors="white", linewidths=0.3, zorder=4,
            )
            ax.plot([i, i], [mean - sd, mean + sd], color="black", lw=1.0,
                    solid_capstyle="butt", zorder=3)
            ax.plot([i, i], [vals.min(), vals.max()], color=c, lw=0.6, alpha=0.55,
                    zorder=2)
            ax.scatter([i], [mean], marker="s", s=34, c="black", zorder=5,
                       edgecolors="white", linewidths=0.6)
        n_per = subj.groupby("paradigm")["subject_id"].nunique()
        ax.set_xticks(range(n_p))
        ax.set_xticklabels(
            [f"{p}\n(n={int(n_per[p])})" for p in paradigms], fontsize=5.5, rotation=45,
            ha="right",
        )
        ax.set_xlim(-0.5, n_p - 0.5)
        ax.set_title(title, fontweight="bold", fontsize=8)
        ax.set_ylabel(ylab, fontsize=7)
        ax.tick_params(axis="y", labelsize=6)
        ax.grid(axis="y", color="#E5E7EB", lw=0.5, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if col == "rt_mean":
            ax.axhline(0, color="0.5", ls="--", lw=0.7)

    fig.tight_layout(pad=1.0)
    return fig


__all__ = ["summarize_subjects", "plot_paradigm_dumbbell"]
