"""
Cercus Framework — Heatmap Plots
=================================
Density heatmaps and trial-stacked heatmaps for kinetics visualization.
"""

from __future__ import annotations

import logging

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.constants import (
    COLOR_ESCAPE,
    COLOR_NO_RESPONSE,
    COLOR_PREWALK,
    ESCAPE_START_THRESHOLD,
    ESCAPE_VMAX_THRESHOLD,
    HEATMAP_VMAX,
)

log = logging.getLogger(__name__)


def plot_spaghetti_kinetics_heatmap(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (9, 3.5),
    t_window: tuple[float, float] = (-400.0, 500.0),
    speed_bins: int = 50,
    y_col: str = "speed",
    y_label: str = "Speed (mm/s)",
) -> plt.Figure:
    """Density heatmap of spaghetti kinetics, aligned to escape onset."""
    df = df.copy()
    has_onset = df["interval_onset_ms"].notna()
    df["t_aligned"] = np.where(
        has_onset,
        df["t_rel"] - df["interval_onset_ms"],
        df["t_rel"],
    )

    dt = 5.0
    t_common = np.arange(t_window[0], t_window[1] + dt, dt)

    response_types = ["Escape", "PreWalk", "NoResponse"]
    response_colors = {
        "Escape": COLOR_ESCAPE,
        "PreWalk": COLOR_PREWALK,
        "NoResponse": COLOR_NO_RESPONSE,
    }

    n_panels = len(response_types)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, n_panels, hspace=0.1, wspace=0.15, figure=fig)

    ax_panels: list[plt.Axes] = []
    for j in range(n_panels):
        sharey = ax_panels[0] if ax_panels else None
        ax = fig.add_subplot(gs[0, j], sharey=sharey)
        ax_panels.append(ax)

    speed_min, speed_max = 0.0, 600.0
    speed_edges = np.linspace(speed_min, speed_max, speed_bins + 1)

    vmax_global = 0.0

    density_by_panel: dict[int, np.ndarray] = {}
    mean_by_panel: dict[int, np.ndarray] = {}
    sem_by_panel: dict[int, np.ndarray] = {}
    valid_by_panel: dict[int, np.ndarray] = {}
    stillness_by_panel: dict[int, float] = {}

    for j, response_type in enumerate(response_types):
        subset = df[df["response_type"] == response_type]
        if subset.empty:
            continue

        aligned_speeds: list[np.ndarray] = []
        for (_subj, _tid), grp in subset.groupby(
            ["subject_id", "global_trial_id"]
        ):
            grp_sorted = grp.sort_values("t_aligned")
            t_trial = grp_sorted["t_aligned"].values
            s_trial = grp_sorted[y_col].values
            if (
                len(t_trial) < 2
                or t_trial.min() > t_window[1]
                or t_trial.max() < t_window[0]
            ):
                continue
            s_interp = np.interp(
                t_common, t_trial, s_trial, left=np.nan, right=np.nan
            )
            aligned_speeds.append(s_interp)

        if not aligned_speeds:
            continue

        speed_matrix = np.vstack(aligned_speeds)

        density = np.zeros((len(t_common), speed_bins))
        for i_t in range(len(t_common)):
            speeds_at_t = speed_matrix[:, i_t]
            speeds_at_t = speeds_at_t[~np.isnan(speeds_at_t)]
            if len(speeds_at_t) > 0:
                counts, _ = np.histogram(speeds_at_t, bins=speed_edges)
                density[i_t, :] = counts / (counts.sum() + 1e-12)

        mean = np.nanmean(speed_matrix, axis=0)
        sem = np.nanstd(speed_matrix, axis=0) / np.sqrt(
            np.sum(~np.isnan(speed_matrix), axis=0)
        )
        sem[np.isnan(sem)] = 0
        valid = ~np.isnan(mean)

        pre_mask = (t_common >= -400) & (t_common <= 0)
        pre_speeds = speed_matrix[:, pre_mask]
        pre_speeds = pre_speeds[~np.isnan(pre_speeds)]
        stillness = (
            (pre_speeds < ESCAPE_START_THRESHOLD).mean()
            if len(pre_speeds) > 0
            else 0.0
        )

        density_by_panel[j] = density
        mean_by_panel[j] = mean
        sem_by_panel[j] = sem
        valid_by_panel[j] = valid
        stillness_by_panel[j] = stillness
        vmax_global = max(
            vmax_global, np.percentile(density[density > 0], 99.5)
        )

    ims: list = []
    for j, response_type in enumerate(response_types):
        ax = ax_panels[j]
        color = response_colors[response_type]

        if j not in density_by_panel:
            ax.set_title(response_type, fontweight="bold")
            ax.set_xlim(t_window)
            ax.set_ylim(speed_min, speed_max)
            continue

        density = density_by_panel[j]
        mean = mean_by_panel[j]
        sem = sem_by_panel[j]
        valid = valid_by_panel[j]
        stillness = stillness_by_panel[j]

        dt = t_common[1] - t_common[0] if len(t_common) > 1 else 5.0
        t_edges = np.concatenate(
            [
                [t_common[0] - dt / 2],
                (t_common[:-1] + t_common[1:]) / 2,
                [t_common[-1] + dt / 2],
            ]
        )

        norm = mcolors.PowerNorm(gamma=0.45, vmin=0, vmax=vmax_global)
        im = ax.pcolormesh(
            t_edges,
            speed_edges,
            density.T,
            cmap="magma",
            norm=norm,
            shading="auto",
            rasterized=True,
            edgecolors="none",
        )
        ims.append(im)

        ax.plot(
            t_common[valid],
            mean[valid],
            color="black",
            lw=3.5,
            path_effects=[
                path_effects.withStroke(linewidth=3.5, foreground="black")
            ],
            zorder=5,
        )
        ax.plot(
            t_common[valid],
            mean[valid],
            color="white",
            lw=2.0,
            path_effects=[
                path_effects.withStroke(linewidth=3.5, foreground="black")
            ],
            zorder=6,
        )

        ax.plot(
            t_common[valid],
            mean[valid] + sem[valid],
            color="white",
            lw=1.0,
            ls=":",
            alpha=0.8,
            zorder=4,
        )
        ax.plot(
            t_common[valid],
            mean[valid] - sem[valid],
            color="white",
            lw=1.0,
            ls=":",
            alpha=0.8,
            zorder=4,
        )

        ax.axvline(0, color="white", ls="--", lw=0.8, alpha=0.6, zorder=7)

        note = " (aligned to TTC)" if response_type == "NoResponse" else ""
        ax.set_title(f"{response_type}{note}", fontweight="bold")

        # ax.text(
        #     0.02,
        #     0.98,
        #     f"stillness {stillness:.0%}",
        #     transform=ax.transAxes,
        #     ha="left",
        #     va="top",
        #     fontsize=7,
        #     color="white",
        #     fontweight="bold",
        #     path_effects=[
        #         path_effects.withStroke(linewidth=2.0, foreground="black")
        #     ],
        #     zorder=10,
        # )

        ax.set_xlabel("Time from escape onset (ms)")
        if j == 0:
            ax.set_ylabel(y_label)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)

        ax.set_xlim(t_window)
        ax.set_ylim(speed_min, speed_max)
        ax.grid(False)

        if y_col == "speed":
            ax.axhline(
                y=ESCAPE_VMAX_THRESHOLD,
                color="k",
                ls="--",
                lw=0.75,
                alpha=0.7,
            )
            ax.text(
                ax.get_xlim()[1] * 0.98,
                ESCAPE_VMAX_THRESHOLD + 1.0,
                f"Vmax ({ESCAPE_VMAX_THRESHOLD:.0f})",
                ha="right",
                va="bottom",
                fontsize=6,
                color="k",
                alpha=0.7,
                path_effects=[
                    path_effects.withStroke(linewidth=1.5, foreground="white")
                ],
            )
            ax.axhline(
                y=ESCAPE_START_THRESHOLD,
                color="0.5",
                ls="--",
                lw=0.5,
                alpha=0.5,
            )
            ax.text(
                ax.get_xlim()[1] * 0.98,
                ESCAPE_START_THRESHOLD + 1.0,
                f"Start ({ESCAPE_START_THRESHOLD:.0f})",
                ha="right",
                va="bottom",
                fontsize=6,
                color="0.5",
                alpha=0.5,
                path_effects=[
                    path_effects.withStroke(linewidth=1.5, foreground="white")
                ],
            )
        else:
            ax.axhline(y=0, color="0.5", ls="--", lw=0.5, alpha=0.4)

    if ims:
        cbar = fig.colorbar(
            ims[0], ax=ax_panels, fraction=0.02, pad=0.02
        )
        cbar.set_label("P(speed | time)", fontsize=7)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["0", "1"])
        cbar.ax.tick_params(labelsize=6)

    fig.tight_layout(pad=1.0)
    return fig


def plot_trial_stacked_heatmap(
    df: pd.DataFrame,
    align: str = "ttc",
    t_window: tuple[float, float] = (-1.0, 2.0),
    t_bin_s: float = 0.01,
    vmax: float = HEATMAP_VMAX,
    figsize: tuple[float, float] = (12, 4.5),
    conditions: list[str] | None = None,
    max_trials_per_panel: int = 200,
    comparison_mode: bool = False,
) -> plt.Figure:
    """Trial-stacked heatmap: each row is one trial, sorted by onset latency.

    Parameters
    ----------
    comparison_mode : bool
        If True, facets by modality (columns) and response_type (rows),
        aligns to TTC=0, uses PowerNorm(gamma=0.4), shows stillness% per panel.
    """
    if comparison_mode:
        return _plot_comparison_heatmap(df, align, t_window, t_bin_s, vmax, figsize, max_trials_per_panel)

    # ── Original: single-condition panels ──
    df = df.copy()
    # Align time axis: 'ttc' uses t_rel directly (t_rel=0 is TTC),
    # 'onset' subtracts escape onset so t=0 is the first >10 mm/s frame.
    if align == "onset":
        df["_align_t_s"] = np.where(
            df["interval_onset_ms"].notna(),
            (df["t_rel"] - df["interval_onset_ms"]) / 1000.0,
            df["t_rel"] / 1000.0,
        )
    else:
        df["_align_t_s"] = df["t_rel"] / 1000.0

    n_types = df["type"].nunique() if "type" in df.columns else 0
    if conditions is None:
        if n_types >= 4:
            conditions = sorted(df["type"].dropna().unique())
        else:
            conditions = ["Escape", "PreWalk", "NoResponse"]

    n_panels = len(conditions)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        1, n_panels, hspace=0.15, wspace=0.25, figure=fig
    )

    t_common = np.arange(t_window[0], t_window[1], t_bin_s)
    ims: list = []

    for j, cond in enumerate(conditions):
        ax = fig.add_subplot(gs[0, j])

        if n_types >= 4:
            subset = df[df["type"] == cond].copy()
        else:
            subset = df[df["response_type"] == cond].copy()

        if subset.empty:
            ax.set_title(cond, fontweight="bold")
            continue

        trial_rows: list[tuple[float, np.ndarray]] = []
        for (_subj, _tid), grp in subset.groupby(
            ["subject_id", "global_trial_id"]
        ):
            grp_sorted = grp.sort_values("_align_t_s")
            t_trial = grp_sorted["_align_t_s"].values
            s_trial = grp_sorted["speed"].values
            if (
                len(t_trial) < 2
                or t_trial.min() > t_window[1]
                or t_trial.max() < t_window[0]
            ):
                continue
            s_interp = np.interp(
                t_common, t_trial, s_trial, left=0, right=0
            )
            above = s_interp > ESCAPE_START_THRESHOLD
            latency = (
                float(t_common[above].min()) if above.any() else float("inf")
            )
            trial_rows.append((latency, s_interp))

        if not trial_rows:
            ax.set_title(cond, fontweight="bold")
            continue

        trial_rows.sort(key=lambda x: x[0] if np.isfinite(x[0]) else 1e9)
        if len(trial_rows) > max_trials_per_panel:
            step = len(trial_rows) / max_trials_per_panel
            indices = np.arange(0, len(trial_rows), step).astype(int)
            trial_rows = [trial_rows[i] for i in indices]

        M = np.vstack([row[1] for row in trial_rows])

        ax.imshow(
            M,
            aspect="auto",
            origin="lower",
            extent=[t_common[0], t_common[-1], 0, len(M)],
            cmap="inferno",
            vmin=0,
            vmax=vmax,
            interpolation="nearest",
        )

        ax.axvline(0, color="white", ls="--", lw=1.2, alpha=0.7)

        ax.set_title(cond, fontweight="bold", fontsize=9)
        if align == "ttc":
            ax.set_xlabel("TTC (s)")
        else:
            ax.set_xlabel("Time from escape onset (s)")
        if j == 0:
            ax.set_ylabel("Trial")
            ax.set_yticks([])
        else:
            plt.setp(ax.get_yticklabels(), visible=False)

        ax.text(
            0.02,
            0.98,
            f"n={len(M)}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            color="white",
            fontweight="bold",
            path_effects=[
                path_effects.withStroke(linewidth=2.0, foreground="black")
            ],
        )

    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
    cmap = plt.get_cmap("inferno")
    norm = mcolors.Normalize(vmin=0, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Translational velocity (mm/s)", fontsize=8)

    fig.tight_layout(pad=1.0)
    return fig


def _plot_comparison_heatmap(
    df: pd.DataFrame,
    align: str = "ttc",
    t_window: tuple[float, float] = (-1.0, 2.0),
    t_bin_s: float = 0.01,
    vmax: float = HEATMAP_VMAX,
    figsize: tuple[float, float] = (12, 4.5),
    max_trials_per_panel: int = 200,
) -> plt.Figure:
    """Cross-modal comparison heatmap faceted by modality and response_type."""
    df = df.copy()
    has_response = "response_type" in df.columns
    mod_col = "modality" if "modality" in df.columns else (
        "type" if "type" in df.columns else "response_type"
    )

    # Determine rows and columns
    modalities = sorted(df[mod_col].dropna().unique())
    if has_response:
        response_types = sorted(df["response_type"].dropna().unique())
    else:
        response_types = ["Escape"]

    n_rows = len(response_types)
    n_cols = len(modalities)
    fig = plt.figure(figsize=(figsize[0] * 0.8, figsize[1] * 0.8))
    gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.25, wspace=0.15, figure=fig)

    t_common = np.arange(t_window[0], t_window[1], t_bin_s)
    norm = mcolors.PowerNorm(gamma=0.4, vmin=0, vmax=vmax)
    ims = []

    for ri, rt in enumerate(response_types):
        for ci, mod in enumerate(modalities):
            ax = fig.add_subplot(gs[ri, ci])

            # Filter
            subset = df[df[mod_col] == mod].copy()
            if has_response:
                subset = subset[subset["response_type"] == rt]

            if subset.empty:
                ax.set_title(f"{mod} {rt}" if has_response else mod, fontweight="bold")
                continue

            # Align: 'ttc' uses t_rel directly, 'onset' subtracts escape onset
            if align == "onset":
                subset["_align_t_s"] = np.where(
                    subset["interval_onset_ms"].notna(),
                    (subset["t_rel"] - subset["interval_onset_ms"]) / 1000.0,
                    subset["t_rel"] / 1000.0,
                )
            else:
                subset["_align_t_s"] = subset["t_rel"] / 1000.0

            trial_rows: list[tuple[float, np.ndarray]] = []
            for (_subj, _tid), grp in subset.groupby(
                ["subject_id", "global_trial_id"]
            ):
                grp_sorted = grp.sort_values("_align_t_s")
                t_trial = grp_sorted["_align_t_s"].values
                s_trial = grp_sorted["speed"].values
                if len(t_trial) < 2 or t_trial.min() > t_window[1] or t_trial.max() < t_window[0]:
                    continue
                s_interp = np.interp(t_common, t_trial, s_trial, left=0, right=0)
                above = s_interp > ESCAPE_START_THRESHOLD
                latency = float(t_common[above].min()) if above.any() else float("inf")
                trial_rows.append((latency, s_interp))

            if not trial_rows:
                ax.set_title(f"{mod} {rt}" if has_response else mod, fontweight="bold")
                continue

            trial_rows.sort(key=lambda x: x[0] if np.isfinite(x[0]) else 1e9)
            if len(trial_rows) > max_trials_per_panel:
                step = len(trial_rows) / max_trials_per_panel
                indices = np.arange(0, len(trial_rows), step).astype(int)
                trial_rows = [trial_rows[i] for i in indices]

            M = np.vstack([row[1] for row in trial_rows])

            im = ax.imshow(M, aspect="auto", origin="lower",
                           extent=[t_common[0], t_common[-1], 0, len(M)],
                           cmap="inferno", norm=norm, interpolation="nearest")
            ims.append(im)

            ax.axvline(0, color="white", ls="--", lw=1.2, alpha=0.7)

            # Stillness%
            pre_mask = (t_common >= -1.0) & (t_common <= 0)
            pre_speeds = M[:, pre_mask].ravel()
            stillness = (pre_speeds < ESCAPE_START_THRESHOLD).mean() if len(pre_speeds) > 0 else 0.0

            ax.set_title(f"{mod} {rt}", fontweight="bold", fontsize=8)

            if ri == n_rows - 1:
                ax.set_xlabel("TTC (s)")
            if ci == 0:
                ax.set_ylabel(f"Trial ({rt})" if has_response else "Trial")
                ax.set_yticks([])
            else:
                plt.setp(ax.get_yticklabels(), visible=False)

            ax.text(0.02, 0.98, f"n={len(M)}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=6.5, color="white", fontweight="bold",
                    path_effects=[path_effects.withStroke(linewidth=2.0, foreground="black")])
            ax.text(0.98, 0.98, f"{stillness:.0%} stillness", transform=ax.transAxes,
                    ha="right", va="top", fontsize=6, color="white",
                    path_effects=[path_effects.withStroke(linewidth=2.0, foreground="black")])

    if ims:
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap("inferno"), norm=norm)
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Translational velocity (mm/s)", fontsize=8)

    fig.tight_layout(pad=1.0)
    return fig