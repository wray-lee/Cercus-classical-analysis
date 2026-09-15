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
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from pipeline.constants import (
    COLOR_ESCAPE,
    COLOR_NO_RESPONSE,
    COLOR_PREWALK,
    ESCAPE_START_THRESHOLD,
    ESCAPE_VMAX_THRESHOLD,
    PREWALK_WINDOW_MS,
)
from cercus.config import get_geometry

log = logging.getLogger(__name__)


def plot_spaghetti_kinetics_heatmap(
    df: pd.DataFrame,
    figsize: tuple[float, float] | None = None,
    t_window: tuple[float, float] = (-400.0, 500.0),
    speed_bins: int = 50,
    y_col: str = "speed",
    y_label: str = "Speed (mm/s)",
    dt: float = 5.0,
    orientation: str = "horizontal",
) -> plt.Figure:
    """Density heatmap of spaghetti kinetics, aligned to escape onset."""
    df = df.copy()
    has_onset = df["interval_onset_ms"].notna()
    df["t_aligned"] = np.where(
        has_onset,
        df["t_rel"] - df["interval_onset_ms"],
        df["t_rel"],
    )

    t_common = np.arange(t_window[0], t_window[1] + dt, dt)

    response_types = ["Escape", "PreWalk", "NoResponse"]
    response_colors = {
        "Escape": COLOR_ESCAPE,
        "PreWalk": COLOR_PREWALK,
        "NoResponse": COLOR_NO_RESPONSE,
    }

    n_panels = len(response_types)
    if figsize is None:
        figsize = (8.0, 7.5) if orientation == "vertical" else (9, 3.5)
    fig = plt.figure(figsize=figsize)
    if orientation == "vertical":
        gs = gridspec.GridSpec(n_panels, 1, hspace=0.35, wspace=0.1, figure=fig)
    else:
        gs = gridspec.GridSpec(1, n_panels, hspace=0.1, wspace=0.15, figure=fig)

    ax_panels: list[plt.Axes] = []
    for j in range(n_panels):
        sharey = ax_panels[0] if ax_panels else None
        ax = fig.add_subplot(gs[j, 0] if orientation == "vertical" else gs[0, j], sharey=sharey)
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

        pre_mask = (t_common >= t_window[0]) & (t_common <= 0)
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

        dt_step = t_common[1] - t_common[0] if len(t_common) > 1 else dt
        t_edges = np.concatenate(
            [
                [t_common[0] - dt_step / 2],
                (t_common[:-1] + t_common[1:]) / 2,
                [t_common[-1] + dt_step / 2],
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

        if orientation == "vertical":
            ax.set_ylabel(y_label)
            if j == n_panels - 1:
                ax.set_xlabel("Time from escape onset (ms)")
            else:
                ax.set_xlabel("")
        else:
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


def _flag_trial(
    cond: str,
    align: str,
    n_types: int,
    t_trial: np.ndarray,
    s_trial: np.ndarray,
    grp_sorted: pd.DataFrame,
) -> dict | None:
    """Per-trial PreWalk-window metadata for onset-aligned heatmaps.

    Returns the classifier's *actual* PreWalk window in onset-aligned seconds
    plus a ``prewind`` flag: escape onset earlier than wind onset, i.e. the
    burst was triggered by vision alone before the wind arrived.  Only then is
    the wind-anchored classifier window shifted right of the white onset line —
    the case that reads as a misclassification without this annotation.
    """
    if align != "onset" or cond not in ("PreWalk", "Escape"):
        return None
    if n_types >= 4 and "wind" not in str(grp_sorted["type"].iloc[0]).lower():
        return None
    onset = grp_sorted["interval_onset_ms"].iloc[0]
    if np.isnan(onset):
        return None
    wind = (
        grp_sorted["target_ttc_ms"].iloc[0]
        if "target_ttc_ms" in grp_sorted.columns
        else np.nan
    )
    if pd.notna(wind):
        lo = (wind - PREWALK_WINDOW_MS - onset) / 1000.0
        hi = (wind - 50.0 - onset) / 1000.0
        prewind = bool(onset < wind)
    else:
        lo, hi, prewind = -PREWALK_WINDOW_MS / 1000.0, -50.0 / 1000.0, False
    m = (t_trial >= lo) & (t_trial < hi)
    mean_in = float(np.nanmean(s_trial[m])) if m.any() else 0.0
    return {"lo": lo, "hi": hi, "prewind": prewind, "mean_in": mean_in}


def plot_trial_stacked_heatmap(
    df: pd.DataFrame,
    align: str = "ttc",
    t_window: tuple[float, float] | None = None,
    t_bin_s: float | None = None,
    vmax: float | None = None,
    figsize: tuple[float, float] | None = None,
    conditions: list[str] | None = None,
    max_trials_per_panel: int = 200,
    orientation: str = "horizontal",
) -> plt.Figure:
    """Trial-stacked heatmap using config-driven windows and color scaling."""
    heatmap_cfg = get_geometry().heatmap
    if t_window is None:
        configured = (
            heatmap_cfg.t_window_onset
            if align == "onset"
            else heatmap_cfg.t_window_ttc
        )
        t_window = (float(configured[0]), float(configured[1]))
    if t_bin_s is None:
        t_bin_s = float(heatmap_cfg.t_bin_s)
    if vmax is None:
        vmax = float(heatmap_cfg.vmax)
    gamma = float(heatmap_cfg.gamma)

    df = df.copy()
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
    if figsize is None:
        figsize = (8.0, 7.5) if orientation == "vertical" else (12, 4.5)
    fig = plt.figure(figsize=figsize)
    if orientation == "vertical":
        gs = gridspec.GridSpec(n_panels, 1, hspace=0.35, wspace=0.1, figure=fig)
    else:
        gs = gridspec.GridSpec(1, n_panels, hspace=0.15, wspace=0.25, figure=fig)

    t_common = np.arange(t_window[0], t_window[1], t_bin_s)
    norm = mcolors.PowerNorm(gamma=gamma, vmin=0, vmax=vmax)
    ims: list = []

    for j, cond in enumerate(conditions):
        ax = fig.add_subplot(gs[j, 0] if orientation == "vertical" else gs[0, j])

        if n_types >= 4:
            subset = df[df["type"] == cond].copy()
        else:
            subset = df[df["response_type"] == cond].copy()

        if subset.empty:
            ax.set_title(cond, fontweight="bold")
            continue

        trial_rows: list[tuple[float, np.ndarray, dict | None]] = []
        for (_subj, _tid), grp in subset.groupby(
            ["subject_id", "global_trial_id"]
        ):
            grp_sorted = grp.sort_values("_align_t_s")
            t_trial = grp_sorted["_align_t_s"].values
            s_trial = grp_sorted["speed"].values
            # Filter NaNs before interp to avoid propagation into latency/sorting
            valid_mask = ~np.isnan(s_trial) & ~np.isnan(t_trial)
            t_trial = t_trial[valid_mask]
            s_trial = s_trial[valid_mask]
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
            trial_rows.append((latency, s_interp, _flag_trial(
                cond, align, n_types, t_trial, s_trial, grp_sorted,
            )))

        if not trial_rows:
            ax.set_title(cond, fontweight="bold")
            continue

        if align == "onset" and cond == "PreWalk":
            # pre-wind escapes grouped at the bright edge, then by speed inside
            # the *actual* (wind-anchored) classifier window
            trial_rows.sort(
                key=lambda x: (
                    x[2]["prewind"] if x[2] else False,
                    x[2]["mean_in"] if x[2] else 0.0,
                ),
                reverse=True,
            )
        else:
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
            norm=norm,
            interpolation="nearest",
        )

        ax.axvline(0, color="white", ls="--", lw=1.2, alpha=0.7)
        # Per-trial: draw the classifier's ACTUAL PreWalk window (wind-anchored,
        # converted to onset-aligned seconds) as a box on that trial's row.
        # Gold + star = pre-wind escape (burst started before the stimulus).
        for i, (_lat, _row, flag) in enumerate(trial_rows):
            if flag is None:
                continue
            y = i + 0.5
            if flag["prewind"]:
                ax.add_patch(
                    Rectangle(
                        (flag["lo"], i), flag["hi"] - flag["lo"], 1.0,
                        facecolor="gold", alpha=0.22, edgecolor="gold",
                        lw=0.8, zorder=6,
                    )
                )
                ax.plot(
                    t_common[0] + 0.02, y, marker="*", ms=4, color="gold",
                    zorder=7, clip_on=False,
                )
            else:
                ax.plot(
                    [flag["lo"], flag["hi"]], [y, y], color="white",
                    lw=0.9, alpha=0.55, solid_capstyle="butt", zorder=6,
                )
        if align == "onset" and any(r[2] for r in trial_rows):
            ax.text(
                0.02, 0.02, "gold box/★ = pre-wind escape · white tick = classifier window",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=5,
                color="white", alpha=0.85,
                path_effects=[path_effects.withStroke(linewidth=1.5, foreground="black")],
            )

        ax.set_title(cond, fontweight="bold", fontsize=9)
        if orientation == "vertical":
            ax.set_ylabel("Trial")
            ax.set_yticks([])
            if j == n_panels - 1:
                ax.set_xlabel("TTC (s)" if align == "ttc" else "Time from escape onset (s)")
            else:
                ax.set_xlabel("")
        else:
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

    if orientation == "vertical":
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.90, 0.2, 0.02, 0.6])
    else:
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
    cmap = plt.get_cmap("inferno")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Translational velocity (mm/s)", fontsize=8)

    fig.tight_layout(pad=1.0)
    return fig