"""
Cercus Framework — Vmax Distribution Plots
===========================================
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import gaussian_kde, norm
from sklearn.mixture import GaussianMixture

from pipeline.constants import (
    ESCAPE_START_THRESHOLD,
    ESCAPE_VMAX_THRESHOLD,
)

log = logging.getLogger(__name__)


def _get_trial_vmax(
    df: pd.DataFrame,
    response_filter: list[str] | None = None,
) -> np.ndarray:
    """Extract per-trial V_max values, optionally filtered by response_type."""
    work = df if response_filter is None else df[df["response_type"].isin(response_filter)]
    return (
        work.groupby(["subject_id", "global_trial_id"])["v_max"]
        .first()
        .dropna()
        .values
    )


def _setup_vmax_axes(
    trial_vmax: np.ndarray,
    x_upper_override: float | None = None,
    figsize: tuple[float, float] = (5.5, 4.0),
) -> tuple[plt.Figure, plt.Axes, float]:
    """Create figure + axes with histogram + KDE."""
    fig, ax = plt.subplots(figsize=figsize)

    x_upper = (
        x_upper_override
        if x_upper_override
        else max(np.percentile(trial_vmax, 99) * 1.15, 200)
    )

    bins = np.linspace(0, x_upper, 51)
    ax.hist(
        trial_vmax,
        bins=bins,
        density=True,
        color="#D0D0D0",
        edgecolor="white",
        linewidth=0.4,
        alpha=0.85,
        label="Histogram",
        zorder=2,
    )

    kde = gaussian_kde(trial_vmax, bw_method="scott")
    x_kde = np.linspace(0, x_upper, 500)
    ax.plot(x_kde, kde(x_kde), color="black", lw=1.2, alpha=0.9, label="KDE", zorder=3)

    return fig, ax, x_upper


def _draw_gmm_thresholds(
    ax: plt.Axes,
    x_upper: float,
    gmm_start_threshold: float | None = None,
    gmm_escape_threshold: float | None = None,
    iqr_gmm_threshold: float | None = None,
) -> None:
    """Draw adaptive GMM threshold lines with staggered labels."""
    label_y = [0.80, 0.70, 0.60]
    idx = 0

    if gmm_start_threshold is not None:
        ax.axvline(
            gmm_start_threshold,
            color="#E69F00",
            ls="--",
            lw=1.0,
            alpha=0.9,
            zorder=4,
        )
        ax.text(
            gmm_start_threshold + x_upper * 0.01,
            ax.get_ylim()[1] * label_y[idx],
            f"GMM start\n{gmm_start_threshold:.0f}",
            fontsize=6,
            color="#E69F00",
            va="top",
        )
        idx += 1

    if gmm_escape_threshold is not None:
        ax.axvline(
            gmm_escape_threshold,
            color="#DC0000",
            ls="--",
            lw=1.0,
            alpha=0.9,
            zorder=4,
        )
        ax.text(
            gmm_escape_threshold + x_upper * 0.01,
            ax.get_ylim()[1] * label_y[idx],
            f"GMM escape\n{gmm_escape_threshold:.0f}",
            fontsize=6,
            color="#DC0000",
            va="top",
        )
        idx += 1

    if iqr_gmm_threshold is not None:
        ax.axvline(
            iqr_gmm_threshold,
            color="#3C5488",
            ls="--",
            lw=1.0,
            alpha=0.9,
            zorder=4,
        )
        ax.text(
            iqr_gmm_threshold + x_upper * 0.01,
            ax.get_ylim()[1] * label_y[idx],
            f"IQR-GMM\n{iqr_gmm_threshold:.0f}",
            fontsize=6,
            color="#3C5488",
            va="top",
        )


def plot_vmax_distribution(
    df: pd.DataFrame, figsize: tuple[float, float] = (5.0, 3.5)
) -> plt.Figure:
    """Histogram + KDE of per-trial V_max for Escape trials."""
    df_resp = df[df["response_type"] == "Escape"].copy()
    if df_resp.empty:
        log.warning("No Escape trials for V_max distribution plot.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5, 0.5, "No Escape trials",
            ha="center", va="center", transform=ax.transAxes,
        )
        return fig

    trial_vmax = (
        df_resp.groupby("global_trial_id")["v_max"].first().dropna().values
    )

    if len(trial_vmax) < 2:
        log.warning("All V_max values are NaN — skipping distribution plot.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5, 0.5, "No valid V_max",
            ha="center", va="center", transform=ax.transAxes,
        )
        return fig

    fig, ax = plt.subplots(figsize=figsize)

    bins = np.linspace(0, 400, 51)
    ax.hist(
        trial_vmax,
        bins=bins,
        density=True,
        color="#D0D0D0",
        edgecolor="white",
        linewidth=0.4,
        alpha=0.85,
        label="Histogram",
        zorder=2,
    )

    kde = gaussian_kde(trial_vmax, bw_method="scott")
    x_kde = np.linspace(0, 400, 500)
    ax.plot(x_kde, kde(x_kde), color="black", lw=1.2, alpha=0.9, label="KDE", zorder=3)

    ax.axvline(
        ESCAPE_VMAX_THRESHOLD, color="black", ls="--", lw=0.75, alpha=0.7, zorder=4
    )
    ax.text(
        ESCAPE_VMAX_THRESHOLD + 3,
        ax.get_ylim()[1] * 0.92,
        f"Vmax\n{ESCAPE_VMAX_THRESHOLD:.0f}",
        fontsize=6,
        color="black",
        va="top",
    )

    ax.axvline(
        ESCAPE_START_THRESHOLD, color="0.5", ls="--", lw=0.75, alpha=0.7, zorder=4
    )
    ax.text(
        ESCAPE_START_THRESHOLD + 3,
        ax.get_ylim()[1] * 0.92,
        f"Start\n{ESCAPE_START_THRESHOLD:.0f}",
        fontsize=6,
        color="0.5",
        va="top",
    )

    ax.set_xlim(0, 400)
    ax.set_xlabel("$V_{max}$ (mm/s)")
    ax.set_ylabel("Probability Density")
    ax.set_title(
        "$V_{max}$ Distribution — Threshold Diagnostic", fontweight="bold"
    )
    ax.legend(loc="upper right", frameon=False, fontsize=6)

    fig.tight_layout(pad=1.0)
    return fig


def plot_population_vmax_gmm(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (5.5, 4.0),
    gmm_start_threshold: float | None = None,
    gmm_escape_threshold: float | None = None,
    iqr_gmm_threshold: float | None = None,
    draw_fixed_thresholds: bool = True,
) -> plt.Figure:
    """V_max distribution of all trials with GMM threshold markers."""
    trial_vmax = _get_trial_vmax(df)

    if len(trial_vmax) < 2:
        log.warning("Insufficient V_max values for GMM plot — skipping.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5, 0.5, "No valid V$_{max}$",
            ha="center", va="center", transform=ax.transAxes, fontsize=12,
        )
        return fig

    all_t = [
        t
        for t in [gmm_start_threshold, gmm_escape_threshold, iqr_gmm_threshold]
        if t
    ]
    x_upper = max(
        np.percentile(trial_vmax, 99) * 1.15,
        max(all_t) * 1.1 if all_t else 0,
        200,
    )

    fig, ax, x_upper = _setup_vmax_axes(trial_vmax, x_upper, figsize)

    if draw_fixed_thresholds:
        ax.axvline(
            ESCAPE_VMAX_THRESHOLD, color="black", ls="--", lw=0.75, alpha=0.7, zorder=4
        )
        ax.text(
            ESCAPE_VMAX_THRESHOLD + x_upper * 0.01,
            ax.get_ylim()[1] * 0.92,
            f"Vmax\n{ESCAPE_VMAX_THRESHOLD:.0f}",
            fontsize=6,
            color="black",
            va="top",
        )

        ax.axvline(
            ESCAPE_START_THRESHOLD, color="0.5", ls="--", lw=0.75, alpha=0.7, zorder=4
        )
        ax.text(
            ESCAPE_START_THRESHOLD + x_upper * 0.01,
            ax.get_ylim()[1] * 0.92,
            f"Start\n{ESCAPE_START_THRESHOLD:.0f}",
            fontsize=6,
            color="0.5",
            va="top",
        )

    _draw_gmm_thresholds(
        ax, x_upper, gmm_start_threshold, gmm_escape_threshold, iqr_gmm_threshold
    )

    ax.set_xlim(0, x_upper)
    ax.set_xlabel("$V_{max}$ (mm/s)")
    ax.set_ylabel("Probability Density")
    ax.set_title(
        "$V_{max}$ Distribution — All Trials (Threshold Determination)",
        fontweight="bold",
    )
    ax.legend(loc="upper right", frameon=False, fontsize=6)

    n_subjects = df["subject_id"].nunique()
    ax.text(
        0.97,
        0.70,
        f"n = {len(trial_vmax)} trials\n({n_subjects} subjects)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.8"
        ),
    )

    fig.tight_layout(pad=1.0)
    return fig


def plot_population_vmax_response(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (5.5, 4.0),
    auto_threshold: float | None = None,
    draw_fixed_thresholds: bool = True,
) -> plt.Figure:
    """V_max distribution of Escape + PreWalk trials only."""
    trial_vmax = _get_trial_vmax(df, response_filter=["Escape", "PreWalk"])

    if len(trial_vmax) < 2:
        log.warning("Insufficient Escape+PreWalk V_max values — skipping.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5, 0.5, "No valid V$_{max}$",
            ha="center", va="center", transform=ax.transAxes, fontsize=12,
        )
        return fig

    x_upper = max(
        np.percentile(trial_vmax, 99) * 1.15,
        auto_threshold * 1.1 if auto_threshold else 0,
        200,
    )

    fig, ax, x_upper = _setup_vmax_axes(trial_vmax, x_upper, figsize)

    if draw_fixed_thresholds:
        ax.axvline(
            ESCAPE_VMAX_THRESHOLD, color="black", ls="--", lw=0.75, alpha=0.7, zorder=4
        )
        ax.text(
            ESCAPE_VMAX_THRESHOLD + x_upper * 0.01,
            ax.get_ylim()[1] * 0.92,
            f"Vmax\n{ESCAPE_VMAX_THRESHOLD:.0f}",
            fontsize=6,
            color="black",
            va="top",
        )

    if auto_threshold is not None:
        ax.axvline(
            auto_threshold, color="#DC0000", ls="--", lw=1.0, alpha=0.9, zorder=4
        )
        ax.text(
            auto_threshold + x_upper * 0.01,
            ax.get_ylim()[1] * 0.80,
            f"Threshold\n{auto_threshold:.0f}",
            fontsize=6,
            color="#DC0000",
            va="top",
        )

    n_esc = len(_get_trial_vmax(df, response_filter=["Escape"]))
    n_pw = len(_get_trial_vmax(df, response_filter=["PreWalk"]))
    n_subjects = df["subject_id"].nunique()
    ax.text(
        0.97,
        0.70,
        f"Escape: {n_esc}  PreWalk: {n_pw}\n({n_subjects} subjects)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.8"
        ),
    )

    ax.set_xlim(0, x_upper)
    ax.set_xlabel("$V_{max}$ (mm/s)")
    ax.set_ylabel("Probability Density")
    ax.set_title(
        "$V_{max}$ Distribution — Escape + PreWalk (Effective Responses)",
        fontweight="bold",
    )
    ax.legend(loc="upper right", frameon=False, fontsize=6)

    fig.tight_layout(pad=1.0)
    return fig


def plot_population_vmax_moving_gmm(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (5.5, 4.0),
    min_speed_floor: float = 10.0,
    gmm_escape_threshold: float | None = None,
    draw_fixed_thresholds: bool = True,
) -> plt.Figure:
    """V_max distribution of active trials (v_max >= min_speed_floor) with 2-component GMM curves.

    Fits a 2-component log-GMM on moving trials (excluding stationary noise < 10 mm/s)
    to separate Walking/Spontaneous Movement from true Escape bursts (~98 mm/s).
    """
    trial_vmax = _get_trial_vmax(df)
    moving_vmax = trial_vmax[trial_vmax >= min_speed_floor]

    if len(moving_vmax) < 2:
        log.warning("Insufficient moving V_max values (>= %.1f mm/s) — skipping.", min_speed_floor)
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5, 0.5, "No valid moving V$_{max}$",
            ha="center", va="center", transform=ax.transAxes, fontsize=12,
        )
        return fig

    fig, ax = plt.subplots(figsize=figsize)

    x_upper = max(
        np.percentile(moving_vmax, 99) * 1.15,
        gmm_escape_threshold * 1.15 if gmm_escape_threshold else 0,
        250,
    )

    bins = np.linspace(min_speed_floor, x_upper, 41)
    ax.hist(
        moving_vmax,
        bins=bins,
        density=True,
        color="#D0D0D0",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.8,
        label=f"Active Trials (≥{min_speed_floor:.0f} mm/s)",
        zorder=2,
    )

    kde = gaussian_kde(moving_vmax, bw_method="scott")
    x_eval = np.linspace(min_speed_floor, x_upper, 500)
    ax.plot(x_eval, kde(x_eval), color="black", lw=1.2, alpha=0.85, label="Empirical KDE", zorder=3)

    # ── Fit 2-component Log-GMM curves if enough samples ──
    fitted_th = gmm_escape_threshold
    if len(moving_vmax) >= 15:
        try:
            log_v = np.log(moving_vmax)
            gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42).fit(log_v.reshape(-1, 1))
            order = np.argsort(gmm.means_.ravel())
            means = gmm.means_.ravel()[order]
            covs = gmm.covariances_.ravel()[order]
            weights = gmm.weights_.ravel()[order]

            # Physical domain log-normal densities: pdf(v) = w * N(log(v)|mu, var) / v
            def _comp_pdf(v: np.ndarray, mu: float, var: float, w: float) -> np.ndarray:
                lv = np.log(v)
                return w * norm.pdf(lv, mu, np.sqrt(var)) / v

            c0_pdf = _comp_pdf(x_eval, means[0], covs[0], weights[0])
            c1_pdf = _comp_pdf(x_eval, means[1], covs[1], weights[1])
            gmm_tot = c0_pdf + c1_pdf

            ax.plot(
                x_eval, c0_pdf, color="#00468B", lw=1.3, ls="--",
                label=f"Walking (μ={np.exp(means[0]):.1f})", zorder=4,
            )
            ax.plot(
                x_eval, c1_pdf, color="#ED0000", lw=1.3, ls="--",
                label=f"Escape (μ={np.exp(means[1]):.1f})", zorder=4,
            )
            ax.plot(
                x_eval, gmm_tot, color="#42B540", lw=1.3,
                label="GMM Total", zorder=4,
            )

            if fitted_th is None:
                def _pair_diff(lx: float) -> float:
                    return (weights[0] * norm.pdf(lx, means[0], np.sqrt(covs[0]))
                            - weights[1] * norm.pdf(lx, means[1], np.sqrt(covs[1])))
                log_th = brentq(_pair_diff, means[0], means[1])
                fitted_th = float(np.exp(log_th))
        except Exception as exc:
            log.warning("GMM density curve plotting failed: %s", exc)

    # ── Threshold markers ──
    if fitted_th is not None:
        ax.axvline(fitted_th, color="#ED0000", ls="-", lw=1.5, alpha=0.9, zorder=5)
        ax.text(
            fitted_th + x_upper * 0.015,
            ax.get_ylim()[1] * 0.85,
            f"GMM Escape Vmax\n{fitted_th:.1f} mm/s",
            fontsize=7,
            color="#ED0000",
            fontweight="bold",
            va="top",
        )

    if draw_fixed_thresholds:
        ax.axvline(ESCAPE_START_THRESHOLD, color="0.5", ls=":", lw=1.0, alpha=0.8, zorder=4)
        ax.text(
            ESCAPE_START_THRESHOLD + x_upper * 0.01,
            ax.get_ylim()[1] * 0.95,
            f"Noise Floor\n{ESCAPE_START_THRESHOLD:.0f} mm/s",
            fontsize=6,
            color="0.4",
            va="top",
        )

    ax.set_xlim(0, x_upper)
    ax.set_xlabel("$V_{max}$ (mm/s)")
    ax.set_ylabel("Probability Density")
    ax.set_title(
        "$V_{max}$ Distribution — Moving Trials (GMM Threshold Model)",
        fontweight="bold",
    )
    ax.legend(loc="upper right", frameon=False, fontsize=6)

    n_subjects = df["subject_id"].nunique()
    ax.text(
        0.97,
        0.52,
        f"n = {len(moving_vmax)} active / {len(trial_vmax)} total\n({n_subjects} subjects)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.8"
        ),
    )

    fig.tight_layout(pad=1.0)
    return fig
