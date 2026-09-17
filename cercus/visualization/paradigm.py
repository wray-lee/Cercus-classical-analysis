"""
Cercus Framework — Cross-Paradigm Dumbbell Plots (full mode)
============================================================
实验室传统连线图（cf. Frontiers fphys.2023.1153913 Fig 2）：
点 = 单只动物均值，竖线连 black square = 范式均值（± SD），横轴 = 范式梯度。
范式间为 between-subject，故不画跨范式折线——每范式一簇。
"""

from __future__ import annotations

import logging

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cercus.analysis.full import _paradigm_sort_key
from cercus.constants.colors import NPG_PALETTE
from cercus.constants.response_types import (
    BURST_CLASSES as _BURST_CLASSES,
    ESCAPE_CLASSES as _ESCAPE_CLASSES,
    RESPONSE_COLORS,
    RESPONSE_TYPES,
)

log = logging.getLogger(__name__)

# RT/distance 由逃逸 burst 定义：有 burst 的三类（Escape/PreEscape/PreWalk）均有值，
# NoResponse 无 burst（rt/dist=NaN）故排除。


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


def summarize_subjects_by_class(df: pd.DataFrame) -> pd.DataFrame:
    """Trial 级表 → per-(paradigm, subject, response_type) 的中位数汇总。

    保留三类有 burst 的行为（Escape / PreEscape / PreWalk，classifier 对
    三者都产出刺激锚定 RT 与逃逸区间行程）；NoResponse 无 burst → 排除。
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
    esc = trial[trial["response_type"].isin(_BURST_CLASSES)]
    return (
        esc.groupby(["paradigm", "subject_id", "response_type"])
        .agg(
            n=("global_trial_index", "count"),
            rt_med=("rt", "median"),
            dist_med=("dist", "median"),
        )
        .reset_index()
    )


def plot_paradigm_rt_dist(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (9.5, 4.0),
) -> plt.Figure:
    """2-panel 范式 × 行为类对比：RT / escape distance（三类有 burst 行为）。

    箱线建立在动物级中位数上（每动物一个数据点，避免 trial 级伪重复），
    逐动物点叠加；底部 Kruskal-Wallis（跨范式，动物级）按类报告。
    """
    from scipy import stats as sps

    subj = summarize_subjects_by_class(df)
    paradigms = sorted(subj["paradigm"].unique(), key=_paradigm_sort_key)
    classes = [c for c in RESPONSE_TYPES
               if c in _BURST_CLASSES and (subj["response_type"] == c).any()]
    n_p = len(paradigms)
    offsets = np.linspace(-0.19, 0.19, max(len(classes), 2))

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    rng = np.random.default_rng(42)
    for ax, (col, ylab) in zip(
        axes,
        (("rt_med", "Reaction time (ms)"), ("dist_med", "Escape-interval distance (mm)")),
    ):
        data, positions, colors = [], [], []
        for i, p in enumerate(paradigms):
            for c, off in zip(classes, offsets):
                vals = (
                    subj.loc[
                        (subj["paradigm"] == p) & (subj["response_type"] == c), col
                    ]
                    .dropna()
                    .values
                )
                if len(vals) == 0:
                    continue
                positions.append(i + off)
                data.append(vals)
                colors.append(RESPONSE_COLORS[c])
        if data:
            bp = ax.boxplot(
                data, positions=positions, widths=0.16,
                patch_artist=True, showfliers=False,
                whiskerprops=dict(color="0.2", lw=0.9),
                capprops=dict(color="0.2", lw=0.9),
                medianprops=dict(color="black", lw=1.2),
            )
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(mcolors.to_rgba(c, 0.30))
                patch.set_edgecolor(c)
                patch.set_linewidth(1.0)
            for pos, vals, c in zip(positions, data, colors):
                ax.scatter(
                    pos + rng.uniform(-0.05, 0.05, len(vals)), vals,
                    s=10, c=c, edgecolors="black", linewidths=0.3,
                    alpha=0.85, zorder=6,
                )
        ax.set_xticks(range(n_p))
        n_per = subj.groupby("paradigm")["subject_id"].nunique()
        ax.set_xticklabels(
            [f"{p}\n(n={int(n_per[p])})" for p in paradigms],
            fontsize=5.5, rotation=45, ha="right",
        )
        ax.set_xlim(-0.5, n_p - 0.5)
        ax.set_ylabel(ylab, fontsize=7)
        ax.tick_params(axis="y", labelsize=6)
        ax.grid(axis="y", color="#E5E7EB", lw=0.5, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if col == "rt_med":
            ax.axhline(0, color="0.5", ls="--", lw=0.7)
        # 跨范式检验：动物级中位数 Kruskal-Wallis（≥3 组且有数据才报）
        notes = []
        for c in classes:
            groups = [
                g[col].dropna().values
                for _p, g in subj[subj["response_type"] == c].groupby("paradigm")
            ]
            groups = [g for g in groups if len(g) > 0]
            if len(groups) >= 3 and any(len(g) >= 2 for g in groups):
                kw = sps.kruskal(*groups)
                p_str = "p<0.001" if kw.pvalue < 0.001 else f"p={kw.pvalue:.3f}"
                notes.append(f"{c}: H={kw.statistic:.1f}, {p_str}")
        if notes:
            # 逐类换行 + 右 panel 右对齐，避免长文本出右边界
            side, x = ("left", 0.0) if ax is axes[0] else ("right", 1.0)
            ax.text(
                x, -0.42,
                "KW across paradigms (subject medians):\n" + "\n".join(notes),
                transform=ax.transAxes, fontsize=5, color="0.3",
                ha=side, va="top", linespacing=1.4,
            )
    handles = [
        mpatches.Patch(
            facecolor=mcolors.to_rgba(RESPONSE_COLORS[c], 0.30),
            edgecolor=RESPONSE_COLORS[c], label=c,
        )
        for c in classes
    ]
    # 图例放 panel 上方（figure 级），避免遮挡散点
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.005),
        ncol=len(classes), fontsize=6.5, frameon=False,
    )
    fig.tight_layout(pad=1.0, rect=(0, 0, 1, 0.95))
    return fig


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
            mean = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
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


__all__ = [
    "summarize_subjects",
    "summarize_subjects_by_class",
    "plot_paradigm_dumbbell",
    "plot_paradigm_rt_dist",
]
