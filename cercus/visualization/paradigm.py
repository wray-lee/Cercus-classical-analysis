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
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from cercus.analysis.full import _paradigm_sort_key
from cercus.analysis.response_groups import effective_main_types, response_group_series
from cercus.config import get_visualization
from cercus.constants.colors import NPG_PALETTE
from cercus.constants.response_types import (
    BURST_CLASSES as _BURST_CLASSES,
    ESCAPE_CLASSES as _ESCAPE_CLASSES,
    RESPONSE_COLORS,
)
from cercus.analysis.reaction_time import select_escape_latency

log = logging.getLogger(__name__)

# Burst 行程独立保留；RT 按分类器的最终类别取 endpoint，缺失不回填。


def _latency_trials(df: pd.DataFrame) -> pd.DataFrame:
    trials = df.drop_duplicates(["paradigm", "subject_id", "global_trial_index"]).copy()
    # Select each raw class's endpoint before deriving the main display group.
    trials["rt"] = select_escape_latency(trials).where(trials["response_type"].isin(_BURST_CLASSES))
    trials["response_group"] = response_group_series(trials)
    trials["dist"] = trials["distance_mm"]
    return trials


def summarize_subjects(df: pd.DataFrame) -> pd.DataFrame:
    """Trial-level full-mode frame → per-(paradigm, subject) summary.

    Columns include classifier trial N and separate RT observed/missing counts.
    ``response_rate`` = (Escape+PreEscape)/all trials; timing uses final classes.
    """
    trial = _latency_trials(df)
    trial["is_escape"] = trial["response_type"].isin(_ESCAPE_CLASSES)
    trial["is_escape_group"] = trial["response_group"].eq("Escape")
    trial["rt_trial"] = trial["response_type"].isin(_BURST_CLASSES)
    out = (
        trial.groupby(["paradigm", "subject_id"])
        .agg(
            n_trials=("global_trial_index", "size"),
            response_rate=("is_escape", "mean"),
            response_group_rate=("is_escape_group", "mean"),
            rt_mean=("rt", "mean"),
            dist_mean=("dist", "mean"),
            rt_observed=("rt", "count"),
            rt_trials=("rt_trial", "sum"),
        )
        .reset_index()
    )
    out["rt_missing"] = out["rt_trials"] - out["rt_observed"]
    return out


def summarize_subjects_by_class(
    df: pd.DataFrame, *, grouped: bool = False,
) -> pd.DataFrame:
    """动物级中位数；默认保留原始分类，主图用 grouped=True。

    RT 先由原始分类选择 endpoint，再合并 trial；n 和 RT 覆盖分开报告。
    """
    trial = _latency_trials(df)
    esc = trial[trial["response_type"].isin(_BURST_CLASSES)]
    class_col = "response_group" if grouped else "response_type"
    out = (
        esc.groupby(["paradigm", "subject_id", class_col])
        .agg(
            n=("global_trial_index", "size"),
            rt_med=("rt", "median"),
            dist_med=("dist", "median"),
            rt_observed=("rt", "count"),
        )
        .reset_index()
    )
    out["rt_missing"] = out["n"] - out["rt_observed"]
    return out


def _compute_zero_anchored_ticks(
    data_vals: np.ndarray,
    margin_ratio: float = 0.07,
    candidates: tuple[int, ...] = (100, 200, 250, 500),
    target_ticks: int = 7,
) -> tuple[tuple[float, float], np.ndarray]:
    """统一零锚定美观刻度：零点严格包含，端部 6%-8% margin，动态步长选择。"""
    val_min = float(np.min(data_vals))
    val_max = float(np.max(data_vals))
    span = max(0.0, val_max) - min(0.0, val_min)
    margin = margin_ratio * span
    raw_min = min(0.0, val_min) - margin
    raw_max = max(0.0, val_max) + margin

    best_step = candidates[-1]
    best_diff = float("inf")
    for s in candidates:
        n = (np.ceil(raw_max / s) - np.floor(raw_min / s)) + 1
        diff = abs(n - target_ticks)
        if diff < best_diff:
            best_diff = diff
            best_step = s

    t_min = float(np.floor(raw_min / best_step) * best_step)
    t_max = float(np.ceil(raw_max / best_step) * best_step)
    ticks = np.arange(t_min, t_max + best_step / 2, best_step)
    return (t_min, t_max), ticks


def _compute_distance_ticks(
    data_vals: np.ndarray,
    margin_ratio: float = 0.07,
    candidates: tuple[int, ...] = (25, 50, 100, 150, 200),
    target_ticks: int = 7,
) -> tuple[tuple[float, float], np.ndarray]:
    """Distance 面板刻度：从 0 开始，均匀步长，端部 6%-8% margin。"""
    val_max = float(np.max(data_vals))
    raw_max = val_max * (1.0 + margin_ratio)

    best_step = candidates[-1]
    best_diff = float("inf")
    for s in candidates:
        n = np.ceil(raw_max / s) + 1
        diff = abs(n - target_ticks)
        if diff < best_diff:
            best_diff = diff
            best_step = s

    t_max = float(np.ceil(raw_max / best_step) * best_step)
    ticks = np.arange(0, t_max + best_step / 2, best_step)
    return (0.0, t_max), ticks


def plot_paradigm_rt_dist(
    df: pd.DataFrame,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """2-panel 范式 × 主分析分组：RT / escape distance。

    箱线建立在动物级中位数上（每动物一个数据点，避免 trial 级伪重复），
    逐动物点叠加；底部 Kruskal-Wallis（跨范式，动物级）按类报告。
    """
    from scipy import stats as sps

    vis = get_visualization()
    if figsize is None:
        figsize = tuple(vis.paradigm_rt_dist_figsize)
    subj = summarize_subjects_by_class(df, grouped=True)
    paradigms = sorted(subj["paradigm"].unique(), key=_paradigm_sort_key)
    classes = [c for c in effective_main_types()
               if c in _BURST_CLASSES and (subj["response_group"] == c).any()]
    n_p = len(paradigms)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.subplots_adjust(**vis.paradigm_rt_dist_layout.to_dict())
    rng = np.random.default_rng(42)
    panel_tags = ("a", "b")
    panel_notes: list[list[str]] = []
    for ax, (col, ylab), tag in zip(
        axes,
        (("rt_med", "Selected timing endpoint vs reference (ms)"), ("dist_med", "Escape-interval distance (mm)")),
        panel_tags,
    ):
        # 面板标签：set_title 左对齐到轴框上方，自动避让 y 轴标签（不再重叠）
        ax.set_title(tag, loc="left", fontsize=8.5, fontweight="bold", pad=11)
        data, positions, colors = [], [], []
        for i, p in enumerate(paradigms):
            p_classes = [
                c for c in classes
                if len(
                    subj.loc[
                        (subj["paradigm"] == p) & (subj["response_group"] == c), col
                    ].dropna()
                ) > 0
            ]
            if len(p_classes) == 2:
                p_offsets = [-0.10, 0.10]
            elif len(p_classes) == 3:
                p_offsets = [-0.16, 0.0, 0.16]
            elif len(p_classes) == 1:
                p_offsets = [0.0]
            else:
                p_offsets = np.linspace(-0.16, 0.16, len(p_classes))

            for c, off in zip(p_classes, p_offsets):
                vals = (
                    subj.loc[
                        (subj["paradigm"] == p) & (subj["response_group"] == c), col
                    ]
                    .dropna()
                    .values
                )
                positions.append(i + off)
                data.append(vals)
                colors.append(RESPONSE_COLORS[c])

        if data:
            bp = ax.boxplot(
                data, positions=positions, widths=0.14,
                patch_artist=True, showfliers=False,
                whiskerprops=dict(color="0.2", lw=0.9),
                capprops=dict(color="0.2", lw=0.9),
                medianprops=dict(color="black", lw=1.3),
            )
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(mcolors.to_rgba(c, 0.25))
                patch.set_edgecolor(c)
                patch.set_linewidth(0.8)
            for pos, vals, c in zip(positions, data, colors):
                ax.scatter(
                    pos + rng.uniform(-0.04, 0.04, len(vals)), vals,
                    s=16, c=c, edgecolors="white", linewidths=0.4,
                    alpha=0.65, zorder=6,
                )
        ax.set_xticks(range(n_p))
        n_per = subj.groupby("paradigm")["subject_id"].nunique()
        observed_per = subj.loc[subj[col].notna()].groupby("paradigm")["subject_id"].nunique()
        ax.set_xticklabels(
            paradigms,
            fontsize=6.5, rotation=0, ha="center",
        )
        coverage = subj.groupby("paradigm")[['n', 'rt_observed', 'rt_missing']].sum()
        observed_total = int(coverage.rt_observed.sum())
        n_total = int(coverage.n.sum())
        subjects_total = int(n_per.sum())
        observed_subjects = int(observed_per.sum())
        per_paradigm = [
            f"{p}: {int(coverage.loc[p, 'rt_observed'])}/{int(coverage.loc[p, 'n'])}"
            for p in paradigms
        ]
        coverage_lines = [
            f"RT coverage: N={n_total}; observed={observed_total}; missing={n_total - observed_total}",
            f"subjects={observed_subjects}/{subjects_total}; observed / classifier trials by paradigm:",
        ] + [";  ".join(per_paradigm[i:i + 2]) for i in range(0, n_p, 2)]
        ax.text(
            ax.get_position().x0, float(vis.paradigm_rt_dist_coverage_y),
            "\n".join(coverage_lines),
            transform=fig.transFigure, fontsize=float(vis.paradigm_rt_dist_footer_fontsize),
            color="0.3", ha="left", va="top", linespacing=1.2,
        )
        ax.set_xlim(-0.5, n_p - 0.5)
        ax.set_ylabel(ylab, fontsize=8.0)
        ax.tick_params(axis="both", labelsize=6.5)
        ax.grid(axis="y", color="#E5E7EB", lw=0.5, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if col == "rt_med":
            ax.axhline(0, color="#7F8C8D", ls="--", lw=0.6, zorder=1)
            # symlog：负值区 log 压缩（bv 碰撞锚 RT 深至 -2300ms），正值区线性
            # 放大（风锚 RT 0~120ms 的组间差异可读）——单一线性轴会把正值压扁。
            ax.set_yscale("symlog", linthresh=100, linscale=1.2)
            ax.set_ylim(-3000, 160)
            ax.set_yticks([-2000, -1000, -500, -200, -100, -50, 0, 50, 100])
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
            ax.yaxis.set_minor_locator(mticker.NullLocator())
        elif col == "dist_med":
            all_vals = np.concatenate(data) if data else np.array([0.0])
            (t_min, t_max), ticks = _compute_distance_ticks(all_vals)
            ax.set_ylim(t_min, t_max)
            ax.set_yticks(ticks)
        # 跨范式检验：动物级中位数 Kruskal-Wallis（≥3 组且有数据才报）
        notes = []
        for c in classes:
            groups = [
                g[col].dropna().values
                for _p, g in subj[subj["response_group"] == c].groupby("paradigm")
            ]
            groups = [g for g in groups if len(g) > 0]
            if len(groups) >= 3 and any(len(g) >= 2 for g in groups):
                kw = sps.kruskal(*groups)
                p_str = "p<0.001" if kw.pvalue < 0.001 else f"p={kw.pvalue:.3f}"
                notes.append(f"{c}: H={kw.statistic:.1f}, {p_str}")
        panel_notes.append(notes)
    # 固定 figure footer 行，coverage 与 KW 不再争用负轴坐标。
    for ax, notes in zip(axes, panel_notes):
        if not notes:
            continue
        ax.text(
            ax.get_position().x0, float(vis.paradigm_rt_dist_stats_y),
            "KW (subject medians): " + "; ".join(notes),
            transform=fig.transFigure, fontsize=float(vis.paradigm_rt_dist_footer_fontsize),
            color="0.3", ha="left", va="top",
        )
    handles = [
        mpatches.Patch(
            facecolor=mcolors.to_rgba(RESPONSE_COLORS[c], 0.30),
            edgecolor=RESPONSE_COLORS[c], label=c,
        )
        for c in classes
    ]
    # 图例位于固定主图区域上方，footer 的独立行保留覆盖率与统计信息。
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
        ncol=len(classes), fontsize=7, frameon=False, columnspacing=1.5,
    )
    return fig


def plot_paradigm_dumbbell(
    df: pd.DataFrame,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """3-panel dumbbell: response rate / RT / distance across paradigms.

    输入 = full 模式的 trial 级聚合 df（需含 paradigm / subject_id 列）。
    """
    if figsize is None:
        vis = get_visualization()
        figsize = tuple(vis.get("paradigm_dumbbell_figsize", vis.get("paradigm_reaction_distance_figsize", (7.5, 3.2))))
    subj = summarize_subjects(df)
    paradigms = sorted(subj["paradigm"].unique(), key=_paradigm_sort_key)
    n_p = len(paradigms)
    colors = {p: NPG_PALETTE[i % len(NPG_PALETTE)] for i, p in enumerate(paradigms)}

    panels = (
        ("response_group_rate", "Escape probability", "P(derived Escape group)"),
        ("rt_mean", "Class-selected response timing", "Selected timing endpoint (ms)"),
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
        n_per = subj.loc[subj["rt_trials"] > 0].groupby("paradigm")["subject_id"].nunique() if col == "rt_mean" else subj.groupby("paradigm")["subject_id"].nunique()
        observed_per = subj.loc[subj[col].notna()].groupby("paradigm")["subject_id"].nunique()
        ax.set_xticks(range(n_p))
        ax.set_xticklabels(
            [
                f"{p}\nsubjects={int(observed_per.get(p, 0))}/{int(n_per[p])}"
                for p in paradigms
            ],
            fontsize=6.0, rotation=45, ha="right",
        )
        if col == "rt_mean":
            coverage = subj.groupby("paradigm")[["rt_trials", "rt_observed", "rt_missing"]].sum()
            ax.text(
                0.0, -0.25,
                "RT coverage:\n" + "\n".join(
                    f"{p}: N={int(coverage.loc[p, 'rt_trials'])}; "
                    f"observed={int(coverage.loc[p, 'rt_observed'])}; "
                    f"missing={int(coverage.loc[p, 'rt_missing'])}; "
                    f"subjects={int(observed_per.get(p, 0))}/{int(n_per[p])}"
                    for p in paradigms
                ),
                transform=ax.transAxes, fontsize=5.5, color="0.3",
                ha="left", va="top", linespacing=1.3,
            )
        ax.set_xlim(-0.5, n_p - 0.5)
        ax.set_title(title, fontweight="bold", fontsize=8.5)
        ax.set_ylabel(ylab, fontsize=7.5)
        ax.tick_params(axis="y", labelsize=6.5)
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
