"""
Cercus Framework — Behavior Probability & Habituation Plots
===========================================================
"""

from __future__ import annotations

import logging

import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.constants import (
    BAR_LABEL_STYLE,
    COLOR_ESCAPE,
    COLOR_NO_RESPONSE,
    COLOR_NO_STILLNESS,
    COLOR_PREWALK,
    COLOR_WITH_STILLNESS,
    ESCAPE_START_THRESHOLD,
    ESCAPE_VMAX_THRESHOLD,
    PREWALK_WINDOW_MS,
)
from cercus.config import get_thresholds, get_visualization
from cercus.constants.response_types import RESPONSE_COLORS, RESPONSE_TYPES

log = logging.getLogger(__name__)

_NPG8 = [
    "#E64B35", "#4DBBD5", "#00A087", "#3C5488",
    "#F39B7F", "#8491B4", "#91D1C2", "#DC0000",
]


def plot_behavior_probability(df: pd.DataFrame) -> plt.Figure:
    """Bar chart of response-type proportions (RESPONSE_TYPES order)."""
    counts = df.groupby("global_trial_id")["response_type"].first().value_counts()
    total = counts.sum()

    categories = list(RESPONSE_TYPES)
    values = [
        counts.get(c, 0) / total if total > 0 else 0.0 for c in categories
    ]
    colors = [RESPONSE_COLORS[c] for c in categories]

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    bars = ax.bar(
        categories, values, color=colors, width=0.55, edgecolor="none", alpha=0.85
    )

    for bar, val in zip(bars, values):
        if val > 0.02:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.1%}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1.05)
    ax.set_title("Behavior Probability Distribution", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=1.0)
    return fig


def plot_habituation_curve(df: pd.DataFrame) -> plt.Figure:
    """Scatter + line of V_max per trial across the global trial sequence."""
    trial_agg = (
        df.groupby("global_trial_index")
        .agg(
            v_max=("v_max", "first"),
            response_type=("response_type", "first"),
        )
        .reset_index()
    )
    trial_agg = trial_agg.sort_values("global_trial_index")

    x = trial_agg["global_trial_index"].values
    y = trial_agg["v_max"].values

    point_colors = [
        RESPONSE_COLORS.get(rt, COLOR_NO_RESPONSE)
        for rt in trial_agg["response_type"].values
    ]

    fig, ax = plt.subplots(figsize=(8, 3.5))

    ax.plot(x, y, color="0.7", lw=0.8, alpha=0.6, zorder=1)
    ax.scatter(
        x, y, c=point_colors, s=28, edgecolors="white", linewidths=0.4, zorder=2
    )

    ax.axhline(
        y=ESCAPE_VMAX_THRESHOLD,
        color="k",
        linestyle="--",
        linewidth=0.75,
        alpha=0.7,
    )
    ax.text(
        x[-1] + 0.3,
        ESCAPE_VMAX_THRESHOLD,
        f"{ESCAPE_VMAX_THRESHOLD:.0f}",
        ha="left",
        va="center",
        fontsize=6,
        color="k",
        alpha=0.7,
    )

    ax.axhline(
        y=ESCAPE_START_THRESHOLD,
        color="0.5",
        linestyle="--",
        linewidth=0.5,
        alpha=0.5,
    )
    ax.text(
        x[-1] + 0.3,
        ESCAPE_START_THRESHOLD,
        f"{ESCAPE_START_THRESHOLD:.0f}",
        ha="left",
        va="center",
        fontsize=6,
        color="0.5",
        alpha=0.5,
    )

    ax.set_xlabel("Global Trial Index")
    ax.set_ylabel("V$_{max}$ (mm/s)")
    ax.set_title("Habituation Curve", fontweight="bold")
    ax.set_xlim(0.5, len(x) + 0.5)

    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=c,
            markersize=5,
            label=l,
        )
        for l, c in [(rt, RESPONSE_COLORS[rt]) for rt in RESPONSE_TYPES]
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=6)

    fig.tight_layout(pad=1.0)
    return fig


def plot_population_habituation(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (10, 4.5),
) -> plt.Figure:
    """Population fatigue curve: per-subject lines + mean ± SEM ribbon."""
    trial_df = (
        df.groupby(["subject_id", "global_trial_index"])
        .agg(
            v_max=("v_max", "first"),
            response_type=("response_type", "first"),
        )
        .reset_index()
    )
    trial_df = trial_df.sort_values(["subject_id", "global_trial_index"])

    subjects = sorted(trial_df["subject_id"].unique())
    all_indices = sorted(trial_df["global_trial_index"].unique())

    subject_pivots: list[pd.Series] = []
    for subj in subjects:
        sdf = trial_df[trial_df["subject_id"] == subj].set_index(
            "global_trial_index"
        )["v_max"]
        subject_pivots.append(sdf)

    fig, ax = plt.subplots(figsize=figsize)

    for i, (subj, sdf) in enumerate(zip(subjects, subject_pivots)):
        color = _NPG8[i % len(_NPG8)]
        ax.plot(sdf.index, sdf.values, color=color, lw=0.7, alpha=0.35, zorder=1)

    pivot_df = pd.DataFrame(
        {subj: sdf for subj, sdf in zip(subjects, subject_pivots)}
    )
    pivot_df = pivot_df.reindex(all_indices)
    mean = pivot_df.mean(axis=1)
    sem = pivot_df.sem(axis=1)

    ax.plot(
        mean.index,
        mean.values,
        color="black",
        lw=2.0,
        alpha=0.9,
        zorder=3,
        label="Mean ± SEM",
    )
    ax.fill_between(
        mean.index,
        (mean - sem).values,
        (mean + sem).values,
        color="black",
        alpha=0.12,
        zorder=2,
    )

    x_max = max(all_indices)
    ax.axhline(
        y=ESCAPE_VMAX_THRESHOLD,
        color="k",
        linestyle="--",
        linewidth=0.75,
        alpha=0.7,
    )
    ax.text(
        x_max + 0.3,
        ESCAPE_VMAX_THRESHOLD,
        f"{ESCAPE_VMAX_THRESHOLD:.0f}",
        ha="left",
        va="center",
        fontsize=6,
        color="k",
        alpha=0.7,
    )

    ax.axhline(
        y=ESCAPE_START_THRESHOLD,
        color="0.5",
        linestyle="--",
        linewidth=0.5,
        alpha=0.5,
    )
    ax.text(
        x_max + 0.3,
        ESCAPE_START_THRESHOLD,
        f"{ESCAPE_START_THRESHOLD:.0f}",
        ha="left",
        va="center",
        fontsize=6,
        color="0.5",
        alpha=0.5,
    )

    ax.set_xlabel("Global Trial Index")
    ax.set_ylabel("$V_{max}$ (mm/s)")
    ax.set_title("Population Habituation Curve", fontweight="bold")
    ax.set_xlim(0.5, x_max + 0.5)

    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=_NPG8[i % len(_NPG8)], lw=1.0, label=s)
        for i, s in enumerate(subjects)
    ]
    handles.append(Line2D([0], [0], color="black", lw=2.0, label="Mean ± SEM"))
    ax.legend(
        handles=handles,
        loc="upper right",
        frameon=False,
        fontsize=5,
        ncol=max(1, len(subjects) // 4 + 1),
    )

    n_subjects = len(subjects)
    n_trials = len(trial_df)
    ax.text(
        0.02,
        0.97,
        f"{n_subjects} subjects, {n_trials} trials",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.8"
        ),
    )

    fig.tight_layout(pad=1.0)
    return fig


def plot_population_behavior_probability(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (4.5, 3.5),
    bar_label_style: str | None = None,
) -> plt.Figure:
    """Bar chart with per-subject scatter points."""
    trial_level = (
        df.groupby(["subject_id", "global_trial_index"])["response_type"]
        .first()
        .reset_index()
    )

    counts = trial_level["response_type"].value_counts()
    total = counts.sum()
    categories = [rt for rt in RESPONSE_TYPES if rt != "PreEscape" or counts.get(rt, 0) > 0]
    values = [
        counts.get(c, 0) / total if total > 0 else 0.0 for c in categories
    ]
    colors = [RESPONSE_COLORS[c] for c in categories]

    subject_probs = (
        trial_level.groupby("subject_id")["response_type"]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
    )

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(
        categories, values, color=colors, width=0.55, edgecolor="none", alpha=0.85
    )

    rng = np.random.default_rng(42)
    scatter_size = 20
    scatter_alpha = 0.35
    jitter_width = 0.15

    for i, (cat, color) in enumerate(zip(categories, colors)):
        if cat in subject_probs.columns:
            subj_vals = subject_probs[cat].values
        else:
            subj_vals = np.zeros(len(subject_probs))

        jitter = rng.uniform(-jitter_width, jitter_width, size=len(subj_vals))
        ax.scatter(
            np.full(len(subj_vals), i) + jitter,
            subj_vals,
            s=scatter_size,
            c=color,
            alpha=scatter_alpha,
            edgecolors="white",
            linewidths=0.5,
            zorder=5,
        )

    _style = bar_label_style if bar_label_style is not None else BAR_LABEL_STYLE.value

    for bar, val in zip(bars, values):
        if val > 0.02:
            if _style == "axis":
                ax.plot(
                    [0, bar.get_x() + bar.get_width() / 2],
                    [val, val],
                    "k--",
                    lw=0.5,
                    alpha=0.4,
                    zorder=1,
                )
                ax.text(
                    0.02,
                    val,
                    f"{val:.1%}",
                    ha="left",
                    va="center",
                    fontsize=7,
                    color="black",
                    fontweight="bold",
                    zorder=10,
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    0.03,
                    f"{val:.1%}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="black",
                    fontweight="bold",
                    zorder=10,
                    path_effects=[
                        path_effects.withStroke(
                            linewidth=2.0, foreground="white"
                        )
                    ],
                )

    n_subjects = df["subject_id"].nunique()
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1.05)
    ax.set_title("Behavior Probability Distribution", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=1.0)
    fig.text(
        1.02,
        0.5,
        f"{total} trials\n({n_subjects} subjects)",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=7,
        color="0.4",
    )
    return fig


def plot_prewalk_stillness(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (4.5, 3.5),
    stillness_threshold: float = ESCAPE_START_THRESHOLD,
    bar_label_style: str | None = None,
) -> plt.Figure:
    """Plot observed low-speed state separately from resolvable stopping RT.

    Wind ``PreWalk`` trials use ``stillness_presence`` from the source-clock
    measurement. A low-speed state is not itself a resolvable stopping RT;
    missing acquisition data remains explicitly unobserved.
    """
    agg_spec = {
        "response_type": ("response_type", "first"),
        "interval_onset_ms": ("interval_onset_ms", "first"),
    }
    for col in ("stillness_presence", "stillness_reaction_time_ms"):
        if col in df:
            agg_spec[col] = (col, "first")
    trial_level = df.groupby(["subject_id", "global_trial_index"]).agg(**agg_spec).reset_index()
    prewalk_trials = trial_level[trial_level["response_type"] == "PreWalk"].copy()

    if prewalk_trials.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No PreWalk trials", ha="center", va="center", transform=ax.transAxes, fontsize=10, color="0.5")
        return fig

    # Legacy rows lack the field; classify them through the same production path.
    presence: dict[tuple, str] = {}
    prewalk_keys = set(zip(prewalk_trials["subject_id"], prewalk_trials["global_trial_index"]))
    for (subj, tidx), grp in df.groupby(["subject_id", "global_trial_index"]):
        if (subj, tidx) not in prewalk_keys:
            continue
        trial_type = str(grp["type"].iloc[0]).lower() if "type" in grp else ""
        if "wind" in trial_type:
            if "stillness_presence" in grp:
                state = grp["stillness_presence"].iloc[0]
            else:
                from pipeline.classifier import classify_trial
                state = classify_trial(grp)["stillness_presence"]
            presence[(subj, tidx)] = (
                str(state) if state in {"low_speed", "no_low_speed"} else "unobserved"
            )
            continue
        onset_ms = grp["interval_onset_ms"].iloc[0]
        if pd.isna(onset_ms):
            presence[(subj, tidx)] = "unobserved"
            continue
        window = grp.loc[
            (grp["t_rel"] >= onset_ms - PREWALK_WINDOW_MS)
            & (grp["t_rel"] < onset_ms),
            "speed",
        ].dropna()
        presence[(subj, tidx)] = (
            "low_speed" if (window < stillness_threshold).any()
            else "no_low_speed" if len(window) else "unobserved"
        )

    prewalk_trials["stillness_presence"] = [
        presence.get((row.subject_id, row.global_trial_index), "unobserved")
        for row in prewalk_trials.itertuples()
    ]
    categories = ["Low-speed state", "No low-speed state", "Unobserved"]
    states = ["low_speed", "no_low_speed", "unobserved"]
    counts = [int(prewalk_trials["stillness_presence"].eq(state).sum()) for state in states]
    n_total = len(prewalk_trials)
    values = [count / n_total for count in counts]
    colors = [COLOR_WITH_STILLNESS, COLOR_NO_STILLNESS, COLOR_NO_RESPONSE]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(categories, values, color=colors, width=0.55, edgecolor="none", alpha=0.85)
    subject_probs = (
        prewalk_trials.groupby("subject_id")["stillness_presence"]
        .value_counts(normalize=True).unstack(fill_value=0.0)
        .reindex(columns=states, fill_value=0.0)
    )
    rng = np.random.default_rng(42)
    for i, (state, color) in enumerate(zip(states, colors)):
        subj_vals = subject_probs[state].values
        ax.scatter(
            i + rng.uniform(-0.15, 0.15, len(subj_vals)), subj_vals,
            s=20, c=color, alpha=0.35, edgecolors="white", linewidths=0.5, zorder=5,
        )
    _style = bar_label_style if bar_label_style is not None else BAR_LABEL_STYLE.value
    for bar, value, count in zip(bars, values, counts):
        if value <= 0.02:
            continue
        label = f"{value:.1%}\n({count})"
        if _style == "axis":
            ax.text(0.02, value, label, ha="left", va="center", fontsize=7, color="black", fontweight="bold", zorder=10)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, 0.03, label, ha="center", va="bottom", fontsize=7, color="black", fontweight="bold", zorder=10, path_effects=[path_effects.withStroke(linewidth=2.0, foreground="white")])

    rt_values = prewalk_trials.get("stillness_reaction_time_ms", pd.Series(dtype=float))
    n_rt = int(np.isfinite(pd.to_numeric(rt_values, errors="coerce")).sum())
    n_subjects = prewalk_trials["subject_id"].nunique()
    ax.set_ylabel("Proportion of PreWalk trials")
    ax.set_ylim(0, 1.05)
    ax.set_title("PreWalk: Low-speed state and stopping RT", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=1.0)
    fig.text(1.02, 0.5, f"{n_total} PreWalk trials\n({n_subjects} subjects)\nRT observed: {n_rt}/{n_total}", transform=ax.transAxes, ha="left", va="center", fontsize=7, color="0.4")
    return fig


def plot_reaction_distance_panel(
    df: pd.DataFrame,
    figsize: tuple[float, float] | None = None,
    *,
    pause_cohort: str = "strict_moving",
) -> plt.Figure | None:
    """Compare escape latency, distance, and paired stopping/escape endpoints.

    Causal T1/T2 defaults to the strict moving-history proxy across wind
    response classes. ``all_local`` retains intermittent transitions as an
    explicitly diagnostic view. Legacy tables retain their PreWalk fallback.
    """
    if pause_cohort not in {"strict_moving", "all_local"}:
        raise ValueError("pause_cohort must be 'strict_moving' or 'all_local'")
    rt_col = (
        "escape_reaction_time_ms"
        if "escape_reaction_time_ms" in df.columns
        else "reaction_time_ms"
    )
    if rt_col not in df.columns or "distance_mm" not in df.columns:
        log.warning("%s/distance_mm missing — skipping panel.", rt_col)
        return None

    classes = [rt for rt in RESPONSE_TYPES if rt != "NoResponse"]
    agg_spec = {
        "response_type": ("response_type", "first"),
        "rt": (rt_col, "first"),
        "dist": ("distance_mm", "first"),
    }
    causal_pair = {"pause_stopping_time_ms", "pause_to_escape_time_ms"}.issubset(df.columns)
    t1_col = "pause_stopping_time_ms" if causal_pair else "stillness_reaction_time_ms"
    t2_col = "pause_to_escape_time_ms" if causal_pair else "stop_to_escape_interval_ms"
    for key, col in (("t1", t1_col), ("t2", t2_col),
                     ("pause_rt", "pause_reaction_time_ms"), ("pause_status", "pause_status"),
                     ("baseline", "pause_baseline_status")):
        if col in df.columns:
            agg_spec[key] = (col, "first")
    if "type" in df.columns:
        agg_spec["type"] = ("type", "first")
    trial = (
        df.groupby(["subject_id", "global_trial_id"])
        .agg(**agg_spec)
        .reset_index()
    )
    for col in ("t1", "t2"):
        if col not in trial.columns:
            trial[col] = np.nan
    if "pause_rt" in trial and "pause_status" in trial and "type" in trial:
        paused = (
            trial["response_type"].eq("PreWalk")
            & trial["type"].astype(str).str.contains("wind", case=False, na=False)
            & trial["pause_status"].eq("observed")
        )
        # Do not replace an unresolved causal endpoint with the centered one.
        trial.loc[paused, "rt"] = trial.loc[paused, "pause_rt"]
    # Subject summaries use complete trial pairs; faint lines retain each
    # actual trial pair rather than join two independent subject medians.
    subj_med = trial.groupby(["subject_id", "response_type"])[["rt", "dist"]].median()

    fig, axes = plt.subplots(
        1, 3, figsize=figsize or tuple(get_visualization().reaction_distance_figsize),
    )
    rng = np.random.default_rng(42)
    for ax, col, title, xlabel in (
        (axes[0], "rt", "Stimulus-to-escape latency", "Escape onset vs stimulus (ms)"),
        (axes[1], "dist", "Escape distance", "Distance (mm)"),
    ):
        pairs = [
            (rt,
             subj_med.xs(rt, level="response_type")[col].dropna().values
             if rt in subj_med.index.get_level_values("response_type")
             else np.array([]))
            for rt in classes
        ]
        pairs = [(rt, d) for rt, d in pairs if len(d) > 0]
        kept = [rt for rt, _ in pairs]
        data = [d for _, d in pairs]
        if not data:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color="0.5")
            ax.set_title(title, fontweight="bold", fontsize=8)
            continue
        positions = list(range(len(data)))
        bp = ax.boxplot(
            data, positions=positions, patch_artist=True, widths=0.55,
            medianprops=dict(color="black", lw=1.2),
            whiskerprops=dict(color="0.2", lw=0.9),
            capprops=dict(color="0.2", lw=0.9),
            flierprops=dict(markersize=2, alpha=0.4),
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(kept)
        # 顶刊盒须样式：盒边=类别色实线，盒面=同色半透明（黑边+整体 alpha 会渲染成灰边）
        for patch, rt in zip(bp["boxes"], kept):
            c = RESPONSE_COLORS[rt]
            patch.set_facecolor(mcolors.to_rgba(c, 0.30))
            patch.set_edgecolor(c)
            patch.set_linewidth(1.0)
        # per-subject 中位数散点（伪重复对策：显示独立个体的变异）
        for i, (rt, vals) in enumerate(zip(kept, data)):
            ax.scatter(
                np.full(len(vals), i) + rng.uniform(-0.22, 0.22, len(vals)),
                vals, s=9, c=RESPONSE_COLORS[rt], edgecolors="black",
                linewidths=0.3, alpha=0.85, zorder=6,
            )
        ax.set_ylabel(xlabel, fontsize=7)
        ax.set_title(title, fontweight="bold", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(-0.5, len(kept) - 0.5)
        if col == "dist":
            ax.set_ylim(0, get_visualization().reaction_distance_max_mm)
        if col == "rt":
            ax.axhline(0, color="0.5", ls="--", lw=0.7)
            # RT 特殊处理：扩展正值区域并细化刻度（reaction time 通常 -200~+100 ms）
            y_min, y_max = ax.get_ylim()
            if y_max < 50:  # 若 auto-scale 砍掉了正值区域
                y_max = max(50, y_max)
            # 负值范围保留 auto，正值细化刻度间隔（每 25ms 一刻度）
            ax.set_ylim(y_min, y_max)
            ticks = list(ax.get_yticks())
            # 在 [0, y_max] 区间插入细刻度
            pos_ticks = np.arange(0, y_max + 1, 25)
            ticks = sorted(set(ticks) | set(pos_ticks))
            ax.set_yticks([t for t in ticks if y_min <= t <= y_max])
            ax.grid(axis="y", color="#E5E7EB", lw=0.5, alpha=0.6)

    rt_coverage = [
        f"{response}: {int(np.isfinite(group['rt']).sum())}/{len(group)}"
        for response, group in trial.groupby("response_type", sort=False)
        if response in classes
    ]
    axes[0].text(0.5, -0.23, "RT observed — " + "; ".join(rt_coverage),
                 transform=axes[0].transAxes, ha="center", va="top",
                 fontsize=6, color="0.35")

    # ── T1/T2 cohort is independent of the legacy response class ──
    t_ax = axes[2]
    strict = causal_pair and pause_cohort == "strict_moving"
    pair_title = (
        "Strict moving: T1 / T2" if strict else
        "Local transitions: T1 / T2" if causal_pair else
        "Wind PreWalk: T1 / T2 (legacy)"
    )
    t_ax.set_title(pair_title, fontweight="bold", fontsize=8)
    t_ax.set_ylabel("Latency (ms)", fontsize=7)
    t_ax.spines["top"].set_visible(False)
    t_ax.spines["right"].set_visible(False)
    t_ax.tick_params(labelsize=6)
    has_type = "type" in trial.columns
    if has_type:
        type_text = trial["type"].astype(str).str.lower()
        wind = type_text.str.contains("wind", na=False)
        source = wind if causal_pair else wind & trial["response_type"].eq("PreWalk")
        moving = trial.get("baseline", pd.Series(index=trial.index, dtype=str)).eq("continuous_moving")
        selected = source & moving if strict else source
        paired = trial.loc[selected].copy()
        coverage = []
        for name, mask in (("Pure", ~type_text.str.contains("looming", na=False)),
                           ("Multimodal", type_text.str.contains("looming", na=False))):
            group = trial.loc[source & mask]
            if group.empty:
                continue
            eligible = trial.loc[selected & mask]
            complete = np.isfinite(eligible["t1"]) & np.isfinite(eligible["t2"])
            if strict:
                coverage.append(f"{name}: moving {len(eligible)}/{len(group)}; paired {int(complete.sum())}/{len(eligible)}")
            else:
                cohort_n = int((source & mask & moving).sum())
                coverage.append(f"{name}: paired {int(complete.sum())}/{len(group)}; moving-history {cohort_n}")
        paired = paired[np.isfinite(paired["t1"]) & np.isfinite(paired["t2"])]
    else:
        paired = trial.iloc[0:0].copy()
        coverage = []
    if not paired.empty:
        type_text = paired["type"].astype(str).str.lower()
        paired["cohort"] = np.where(
            type_text.str.contains("looming", na=False),
            "Multimodal wind", "Pure wind",
        )
        subject_pairs = (
            paired.groupby(["subject_id", "cohort"])[["t1", "t2"]]
            .median().reset_index()
        )
        cohort_order = ["Pure wind", "Multimodal wind"]
        cohort_color = {"Pure wind": COLOR_PREWALK, "Multimodal wind": COLOR_ESCAPE}
        positions = {"Pure wind": (0, 1), "Multimodal wind": (3, 4)}
        box_values = []
        box_labels = []
        box_colors = []
        box_positions = []
        for cohort in cohort_order:
            values = subject_pairs.loc[subject_pairs["cohort"].eq(cohort)]
            if values.empty:
                continue
            x1, x2 = positions[cohort]
            for x, col, label in ((x1, "t1", "T1"), (x2, "t2", "T2")):
                vals = values[col].dropna().to_numpy(float)
                if len(vals):
                    box_values.append(vals)
                    box_positions.append(x)
                    box_labels.append(f"{cohort.replace(' wind', '')}\n{label}")
                    box_colors.append(cohort_color[cohort])
            for row in paired.loc[paired["cohort"].eq(cohort)].itertuples(index=False):
                t_ax.plot(
                    [x1, x2], [row.t1, row.t2],
                    marker="o", markersize=2.5, markeredgewidth=0.3,
                    markeredgecolor="white", color=cohort_color[cohort],
                    alpha=0.3, lw=0.6, zorder=2,
                )
        if box_values:
            bp = t_ax.boxplot(
                box_values, positions=box_positions, widths=0.55,
                patch_artist=True, medianprops=dict(color="black", lw=1.1),
                whiskerprops=dict(color="0.2", lw=0.8),
                capprops=dict(color="0.2", lw=0.8),
                flierprops=dict(markersize=2, alpha=0.4),
            )
            for patch, color in zip(bp["boxes"], box_colors):
                patch.set_facecolor(mcolors.to_rgba(color, 0.30))
                patch.set_edgecolor(color)
                patch.set_linewidth(1.0)
            for x, vals, color in zip(box_positions, box_values, box_colors):
                t_ax.scatter(
                    np.full(len(vals), x) + rng.uniform(-0.16, 0.16, len(vals)),
                    vals, s=9, c=color, edgecolors="black", linewidths=0.3,
                    alpha=0.85, zorder=5,
                )
            t_ax.set_xticks(box_positions)
            t_ax.set_xticklabels(box_labels)
            t_ax.set_xlim(min(box_positions) - 0.5, max(box_positions) + 0.5)
            t_ax.grid(axis="y", color="#E5E7EB", lw=0.5, alpha=0.6)
            n_subjects = paired["subject_id"].nunique()
            n_trials = len(paired)
            t_ax.text(0.5, -0.21, f"paired: {n_trials} trials / {n_subjects} subjects",
                      transform=t_ax.transAxes, ha="center", va="top", fontsize=6,
                      color="0.35")
            t_ax.text(0.5, -0.28 - 0.07 * max(1, len(coverage)),
                      "Boxes/dots: subject medians; lines: trial pairs",
                      transform=t_ax.transAxes, ha="center", va="top", fontsize=6,
                      color="0.35")
        else:
            t_ax.text(0.5, 0.5, "no paired strict-moving endpoints" if strict else "no paired endpoints", transform=t_ax.transAxes,
                      ha="center", va="center", fontsize=8, color="0.5")
    else:
        t_ax.text(0.5, 0.5, "no paired strict-moving endpoints" if strict else "no paired wind PreWalk endpoints", transform=t_ax.transAxes,
                  ha="center", va="center", fontsize=8, color="0.5")
    t_ax.axhline(0, color="0.5", ls="--", lw=0.7)
    if coverage:
        t_ax.text(0.5, -0.27, "\n".join(coverage), transform=t_ax.transAxes,
                  ha="center", va="top", fontsize=6, color="0.35")

    # Escape vs PreEscape 机制分离检验（subject 级中位数，避免 trial 级伪重复）
    from scipy.stats import mannwhitneyu
    foot = []
    for col, lbl in (("rt", "RT"), ("dist", "dist")):
        esc = (
            subj_med.xs("Escape", level="response_type")[col].dropna().values
            if "Escape" in subj_med.index.get_level_values("response_type")
            else np.array([])
        )
        pre = (
            subj_med.xs("PreEscape", level="response_type")[col].dropna().values
            if "PreEscape" in subj_med.index.get_level_values("response_type")
            else np.array([])
        )
        if len(esc) > 1 and len(pre) > 1:
            _, p_val = mannwhitneyu(esc, pre)
            foot.append(f"{lbl}: n={len(esc)}+{len(pre)} subj, p = {p_val:.2g}")
    if foot:
        fig.text(
            0.5, 0.01,
            "Escape vs PreEscape (subject medians, Mann-Whitney) — "
            + " | ".join(foot),
            ha="center", fontsize=6, color="0.3",
        )

    if causal_pair:
        cfg = get_thresholds()
        cohort_note = (
            f"T1/T2: prior {float(cfg.prewalk.window_ms) / 1000:g} s strictly "
            f">{float(cfg.baseline.quiet_mm_s):g} mm/s, all wind classes (operational moving proxy)."
            if strict else "T1/T2: all local transitions, including intermittent movers (diagnostic, not paper cohort)."
        )
        note = (
            cohort_note + "\n"
            "Observed Wind PreWalk pauses use causal RTm-like = T1 + T2; other RTs retain centered-speed endpoints.\n"
            "PreEscape is a pre-wind lead. No fixed delay correction; not physiological RT."
        )
    else:
        note = (
            "Stimulus-to-escape endpoint; T1 = stimulus-to-stop, T2 = stop-to-escape (legacy centered escape endpoint)."
            if "escape_reaction_time_ms" in df
            else "Legacy reaction_time_ms shown; endpoint definitions depend on the source table."
        )
    fig.text(0.5, 0.035, note, ha="center", fontsize=6, color="0.3")
    fig.tight_layout(pad=1.0, rect=(0, 0.17, 1, 1))
    return fig
