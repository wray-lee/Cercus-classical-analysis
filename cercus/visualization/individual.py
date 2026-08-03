"""
Cercus Framework — Individual-Robustness Supplementary Figures
============================================================
Publication-grade (Nature/Science) figures accompanying the individual-level
robustness checks in ``cercus/analysis/individual.py``. Style mirrors
``cercus/visualization/polar.py`` (NPG palette, gray caption text, white
strokes for legibility).
"""

from __future__ import annotations

import logging

import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.constants import COLOR_ESCAPE, COLOR_PREWALK, NPG_PALETTE
from cercus.visualization._circstats import circ_mean_rad, rayleigh_p

log = logging.getLogger(__name__)

_SECOND_ORDER_MIN_TRIALS = 3   # mirrors cercus/config/defaults/analysis.yaml
_WALLRAFF_MIN_TRIALS = 5       # mirrors cercus/config/defaults/analysis.yaml


def plot_trial_counts(
    counts: pd.Series,
    wallraff: dict | None = None,
    figsize: tuple[float, float] = (6.0, 4.2),
) -> plt.Figure:
    """Histogram of escape trials per animal.

    x = trials per animal, y = number of animals with that many escape trials.
    Reference lines mark the inclusion thresholds used downstream: n≥3 for the
    second-order Rayleigh test, n≥5 for the k-sample Wallraff dispersion test.
    """
    counts = counts.astype(int)
    max_c = int(counts.max())
    n_animals = len(counts)
    median_trials = float(counts.median())

    fig, ax = plt.subplots(figsize=figsize)

    if max_c > 0:
        bins = np.arange(0.5, max_c + 1.5, 1.0)
        ax.hist(
            counts,
            bins=bins,
            color=mcolors.to_rgba(COLOR_ESCAPE, 0.55),
            edgecolor=COLOR_ESCAPE,
            linewidth=1.2,
            rwidth=0.85,
            zorder=2,
        )
        # annotate count above each bar
        vals, _ = np.histogram(counts, bins=bins)
        centers = (bins[:-1] + bins[1:]) / 2
        for x, v in zip(centers, vals):
            if v > 0:
                ax.text(
                    x, v + 0.05, f"{int(v)}", ha="center", va="bottom",
                    fontsize=8, color="0.3",
                )

    ax.axvline(
        _SECOND_ORDER_MIN_TRIALS, color=NPG_PALETTE[1], linestyle="--",
        linewidth=1.0, zorder=1,
    )
    ax.text(
        _SECOND_ORDER_MIN_TRIALS + 0.1, ax.get_ylim()[1] * 0.92,
        f"n≥{_SECOND_ORDER_MIN_TRIALS} (second-order Rayleigh)",
        fontsize=7.5, color=NPG_PALETTE[1], ha="left", va="top",
    )
    ax.axvline(
        _WALLRAFF_MIN_TRIALS, color=NPG_PALETTE[3], linestyle="--",
        linewidth=1.0, zorder=1,
    )
    ax.text(
        _WALLRAFF_MIN_TRIALS + 0.1, ax.get_ylim()[1] * 0.72,
        f"n≥{_WALLRAFF_MIN_TRIALS} (k-sample Wallraff)",
        fontsize=7.5, color=NPG_PALETTE[3], ha="left", va="top",
    )

    ax.set_xlabel("Escape trials per animal")
    ax.set_ylabel("Number of animals")
    ax.set_title("Escape-trial distribution across animals", fontweight="bold")
    ax.set_xticks(np.arange(0, max_c + 2, max(1, (max_c + 2) // 6)))

    caption = (
        f"N animals = {n_animals}, total escape trials = {int(counts.sum())}, "
        f"median = {median_trials:.0f} trials/animal"
    )
    ax.text(
        0.0, -0.22, caption, transform=ax.transAxes, ha="left",
        fontsize=7.5, color="0.35",
    )

    # Wallraff-by-animal dispersion test (individual-level, k-sample across animals)
    if wallraff is not None and not np.isnan(wallraff.get("p", np.nan)):
        sig = "ns" if wallraff["p"] >= 0.05 else "sig."
        n_groups = int(wallraff.get("n_groups", len(counts)))
        min_trials = int(wallraff.get("min_trials", 5))
        excluded = wallraff.get("excluded", 0)
        w_text = (
            f"k-sample Wallraff (n≥{min_trials}, k={n_groups}): H={wallraff['H']:.2f}, "
            f"p={wallraff['p']:.3g} ({sig}) | Animals with <{min_trials} trials excluded"
        )
        ax.text(
            0.0, -0.32, w_text, transform=ax.transAxes, ha="left",
            fontsize=7.0, color="0.45",
        )

    ax.yaxis.grid(True, color="#E5E7EB", linestyle="-", linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=1.2)
    return fig


def plot_second_order(
    animal_df: pd.DataFrame,
    figsize: tuple[float, float] = (4.8, 4.8),
) -> plt.Figure:
    """Polar scatter of per-animal mean escape directions + second-order mean.

    Each marker is one animal placed at (mu_k, R_k): its own circular mean
    direction and within-animal resultant length. The arrow marks the
    second-order (across-animal) mean direction; its length is the second-order
    resultant R2. This is the anti-pseudo-replication view — every animal
    contributes exactly one unit of evidence.
    """
    mu_deg = np.asarray(animal_df["mu_deg"].values, dtype=float)
    R_k = np.asarray(animal_df["R"].values, dtype=float)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetagrids(
        [0, 90, 180, 270],
        ["0°", "90°\nIpsi", "±180°", "−90°\nContra"],
        fontsize=9, color="0.2",
    )
    ax.tick_params(axis="x", pad=6)
    ax.grid(True, color="0.88", linewidth=0.7, linestyle="-")
    ax.set_axisbelow(True)

    if len(mu_deg) > 0:
        ax.scatter(
            np.radians(mu_deg),
            R_k,
            s=48,
            color=mcolors.to_rgba(COLOR_ESCAPE, 0.75),
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )
        # thin spokes from origin to each animal (visual anchor for direction)
        for m, r in zip(np.radians(mu_deg), R_k):
            ax.plot([m, m], [0, r], color=mcolors.to_rgba(COLOR_ESCAPE, 0.25),
                    lw=0.7, zorder=2)

    # Second-order mean + Rayleigh test (n = number of animals)
    if len(mu_deg) >= 2:
        mu2 = circ_mean_rad(np.radians(mu_deg))
        R2 = float(np.abs(np.mean(np.exp(1j * np.radians(mu_deg)))))
        p2 = rayleigh_p(R2, len(mu_deg))
        arrow = ax.annotate(
            "",
            xy=(mu2, R2),
            xytext=(mu2, 0.02),
            arrowprops=dict(
                arrowstyle="-|>", color="k", lw=1.6,
                mutation_scale=11, shrinkA=0, shrinkB=0,
                connectionstyle="arc3,rad=0",
            ),
            zorder=5,
        )
        arrow.arrow_patch.set_path_effects(
            [path_effects.withStroke(linewidth=3.0, foreground="white")]
        )
    else:
        mu2, R2, p2 = np.nan, np.nan, np.nan

    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=3))
    ax.tick_params(axis="y", labelsize=7, colors="0.45", pad=1)
    for lbl in ax.get_yticklabels():
        lbl.set_path_effects(
            [path_effects.withStroke(linewidth=2.0, foreground="white")]
        )

    ax.set_title(
        "Per-animal mean escape directions\n(one dot = one animal)",
        fontweight="bold", pad=22,
    )

    p_str = f"p = {p2:.2e}" if not np.isnan(p2) else "p = n/a"
    sig = "" if np.isnan(p2) else (" (sig.)" if p2 < 0.05 else " (n.s.)")
    caption = (
        f"Second-order mean μ = {np.degrees(mu2):.0f}° (R = {R2:.2f}), "
        f"N = {len(mu_deg)} animals, Rayleigh {p_str}{sig}"
    )
    ax.text(
        0.5, -0.18, caption, transform=ax.transAxes,
        ha="center", fontsize=7.5, color="0.35",
    )
    return fig


def plot_loo(
    loo_df: pd.DataFrame,
    pooled_mu_deg: float,
    figsize: tuple[float, float] = (7.0, 3.8),
) -> plt.Figure:
    """Leave-one-animal-out sensitivity of the pooled escape direction.

    x = animal removed, y = pooled circular mean over the remaining trials.
    The solid line is the full pooled mean; the dashed band marks ±5°. If every
    point stays inside the band, the pooled −74° preference is not carried by
    any single animal.
    """
    animal_ids = loo_df["animal_id"].astype(str).values
    mu_deg = np.asarray(loo_df["mu_deg"].values, dtype=float)
    delta = np.asarray(loo_df["delta_deg"].values, dtype=float)
    n_animals = len(loo_df)

    fig, ax = plt.subplots(figsize=figsize)

    ax.axhline(
        pooled_mu_deg, color="0.15", linewidth=1.4, zorder=1,
        label=f"full pooled μ = {pooled_mu_deg:.1f}°",
    )
    ax.axhline(pooled_mu_deg + 5.0, color="0.6", linestyle="--", linewidth=0.9, zorder=1)
    ax.axhline(
        pooled_mu_deg - 5.0, color="0.6", linestyle="--", linewidth=0.9, zorder=1,
        label="±5° band",
    )

    colors = [
        COLOR_ESCAPE if abs(d) <= 5.0 else COLOR_PREWALK for d in delta
    ]
    ax.scatter(
        range(n_animals), mu_deg, s=46, c=colors,
        edgecolor="white", linewidth=0.8, zorder=3,
    )
    for x, (aid, m) in enumerate(zip(animal_ids, mu_deg)):
        ax.annotate(
            f"{m:.0f}°", (x, m), textcoords="offset points",
            xytext=(0, 6), ha="center", fontsize=7, color="0.35",
        )

    ax.set_xticks(range(n_animals))
    ax.set_xticklabels(animal_ids, rotation=55, ha="right", fontsize=8)
    ax.set_xlim(-0.6, n_animals - 0.4)
    ax.set_xlabel("Animal removed")
    ax.set_ylabel("Pooled escape direction after removal (°)")
    ax.set_title("Leave-one-animal-out sensitivity", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="best")

    n_inside = int((np.abs(delta) <= 5.0).sum())
    caption = (
        f"max |Δμ| = {np.abs(delta).max():.1f}°; "
        f"{n_inside}/{n_animals} removals stay within ±5° of the full pooled mean"
    )
    ax.text(
        0.0, -0.28, caption, transform=ax.transAxes, ha="left",
        fontsize=7.5, color="0.35",
    )

    ax.yaxis.grid(True, color="#E5E7EB", linestyle="-", linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=1.2)
    return fig
