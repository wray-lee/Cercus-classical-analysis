"""
Cercus Framework — Behavior Probability & Habituation Plots
===========================================================
"""

from __future__ import annotations

import logging

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

log = logging.getLogger(__name__)

_NPG8 = [
    "#E64B35", "#4DBBD5", "#00A087", "#3C5488",
    "#F39B7F", "#8491B4", "#91D1C2", "#DC0000",
]


def plot_behavior_probability(df: pd.DataFrame) -> plt.Figure:
    """Bar chart of Escape / PreWalk / NoResponse proportions."""
    counts = df.groupby("global_trial_id")["response_type"].first().value_counts()
    total = counts.sum()

    categories = ["Escape", "PreWalk", "NoResponse"]
    values = [
        counts.get(c, 0) / total if total > 0 else 0.0 for c in categories
    ]
    colors = [COLOR_ESCAPE, COLOR_PREWALK, COLOR_NO_RESPONSE]

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

    color_map = {
        "Escape": COLOR_ESCAPE,
        "PreWalk": COLOR_PREWALK,
        "NoResponse": COLOR_NO_RESPONSE,
    }
    point_colors = [
        color_map.get(rt, COLOR_NO_RESPONSE)
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
        for l, c in [
            ("Escape", COLOR_ESCAPE),
            ("PreWalk", COLOR_PREWALK),
            ("NoResponse", COLOR_NO_RESPONSE),
        ]
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
    categories = ["Escape", "PreWalk", "NoResponse"]
    values = [
        counts.get(c, 0) / total if total > 0 else 0.0 for c in categories
    ]
    colors = [COLOR_ESCAPE, COLOR_PREWALK, COLOR_NO_RESPONSE]

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
    """Proportion of PreWalk trials with stillness before escape onset."""
    trial_level = (
        df.groupby(["subject_id", "global_trial_index"])
        .agg(
            response_type=("response_type", "first"),
            interval_onset_ms=("interval_onset_ms", "first"),
        )
        .reset_index()
    )
    prewalk_trials = trial_level[
        trial_level["response_type"] == "PreWalk"
    ].copy()

    if prewalk_trials.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No PreWalk trials",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            color="0.5",
        )
        return fig

    has_stillness: dict[tuple, bool] = {}
    prewalk_keys = set(
        zip(prewalk_trials["subject_id"], prewalk_trials["global_trial_index"])
    )

    for (subj, tidx), grp in df.groupby(
        ["subject_id", "global_trial_index"]
    ):
        if (subj, tidx) not in prewalk_keys:
            continue
        onset_ms = grp["interval_onset_ms"].iloc[0]
        if pd.isna(onset_ms):
            continue
        window_mask = (
            grp["t_rel"] >= onset_ms - PREWALK_WINDOW_MS
        ) & (grp["t_rel"] < onset_ms)
        window_speed = grp.loc[window_mask, "speed"].dropna()
        has_stillness[(subj, tidx)] = bool(
            (window_speed < stillness_threshold).any()
        )

    prewalk_trials["has_stillness"] = prewalk_trials.apply(
        lambda r: has_stillness.get(
            (r["subject_id"], r["global_trial_index"]), False
        ),
        axis=1,
    )

    n_with = prewalk_trials["has_stillness"].sum()
    n_without = len(prewalk_trials) - n_with
    n_total = len(prewalk_trials)
    prop_with = n_with / n_total if n_total > 0 else 0.0

    subject_props = (
        prewalk_trials.groupby("subject_id")["has_stillness"]
        .mean()
        .reset_index()
        .rename(columns={"has_stillness": "prop_stillness"})
    )

    fig, ax = plt.subplots(figsize=figsize)
    categories = ["With Stillness", "Without Stillness"]
    values = [prop_with, 1.0 - prop_with]
    bar_colors = [COLOR_WITH_STILLNESS, COLOR_NO_STILLNESS]

    bars = ax.bar(
        categories, values, color=bar_colors, width=0.55, edgecolor="none", alpha=0.85
    )

    rng = np.random.default_rng(42)
    jitter_width = 0.15
    for i, col in enumerate(["prop_stillness"]):
        subj_vals = subject_props[col].values
        jitter = rng.uniform(-jitter_width, jitter_width, size=len(subj_vals))
        ax.scatter(
            np.full(len(subj_vals), i) + jitter,
            subj_vals,
            s=20,
            c=bar_colors[i],
            alpha=0.35,
            edgecolors="white",
            linewidths=0.5,
            zorder=5,
        )
    subj_without = 1.0 - subject_props["prop_stillness"].values
    jitter2 = rng.uniform(-jitter_width, jitter_width, size=len(subj_without))
    ax.scatter(
        np.full(len(subj_without), 1) + jitter2,
        subj_without,
        s=20,
        c=bar_colors[1],
        alpha=0.35,
        edgecolors="white",
        linewidths=0.5,
        zorder=5,
    )

    _style = bar_label_style if bar_label_style is not None else BAR_LABEL_STYLE.value

    for bar, val, n in zip(bars, values, [n_with, n_without]):
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
                    f"{val:.1%} ({n})",
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
                    f"{val:.1%}\n({n})",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="black",
                    fontweight="bold",
                    zorder=10,
                    path_effects=[
                        path_effects.withStroke(
                            linewidth=2.0, foreground="white"
                        )
                    ],
                )

    n_subjects = prewalk_trials["subject_id"].nunique()
    ax.set_ylabel("Proportion of PreWalk Trials")
    ax.set_ylim(0, 1.05)
    ax.set_title("PreWalk: Stillness Before Escape Onset", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=1.0)
    fig.text(
        1.02,
        0.5,
        f"{n_total} PreWalk trials\n({n_subjects} subjects)",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=7,
        color="0.4",
    )
    return fig
