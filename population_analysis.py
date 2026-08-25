"""
Cercus Framework — Cross-Subject Population Batch Processor
============================================================
Scans an input directory, processes every subject through the full
preprocessing → classification pipeline, and exports a single unified
summary CSV with per-trial metrics for downstream statistical analysis.

Generates population-level visualization (fatigue curve + V_max distribution)
saved alongside the output CSV.

Usage:
    python population_analysis.py --input-dir path/to/data/ --output-dir results/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.signal import argrelmin, argrelmax
from scipy.stats import gaussian_kde, norm
from sklearn.mixture import GaussianMixture

from pipeline.classifier import label_trials
from pipeline.io import export_summary_metrics, load_and_concat_sessions, scan_and_pair_sessions
from pipeline.kinematics import preprocess
from cercus.visualization import (
    plot_escape_angle_distribution,
    plot_population_behavior_probability,
    plot_population_habituation,
    plot_population_polar_histogram,
    plot_population_pre_movement_prewalk,
    plot_population_spaghetti_kinetics,
    plot_population_speed_kinetics,
    plot_population_vmax_gmm,
    plot_population_vmax_moving_gmm,
    plot_population_vmax_response,
    plot_prewalk_stillness,
    plot_spaghetti_kinetics_heatmap,
    plot_trial_stacked_heatmap,
)

from cercus.config import get_config, get_geometry
from cercus.constants.thresholds import ESCAPE_VMAX_THRESHOLD

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

_cfg = get_config()
_USE_ADAPTIVE_THRESHOLD = bool(_cfg.analysis.vmax_thresholding.use_adaptive)  # True = auto KDE/GMM; False = use config threshold
_FALLBACK_VMAX_THRESHOLD = float(_cfg.analysis.vmax_thresholding.fallback_threshold)  # mm/s, used when all auto methods fail
_KDE_SEARCH_UPPER = float(_cfg.analysis.vmax_thresholding.kde_search_upper)           # mm/s, upper bound for KDE valley search
_ENABLE_IQR_GMM = bool(_cfg.analysis.vmax_thresholding.enable_iqr_gmm)               # set True to also compute IQR-GMM as a candidate
_DRAW_FIXED_THRESHOLDS = bool(_cfg.visualization.draw_fixed_thresholds)              # set False to hide start/vmax reference lines on vmax plots


# ══════════════════════════════════════════════════════════════════════
# CLI Parser
# ══════════════════════════════════════════════════════════════════════


def _safe_savefig(fig, path, **kwargs):
    """Save figure, removing stale file first (Windows OSError 22 workaround)."""
    path = Path(path)
    if path.exists():
        path.unlink()
    fig.savefig(path, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cercus Population Batch Processing — cross-subject metrics export",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input-dir", required=True,
        help="Root directory containing per-subject session CSVs.",
    )
    p.add_argument(
        "--output", required=True,
        help="Directory for all output files (CSV + figures). Created if it doesn't exist.",
    )
    p.add_argument(
        "--individual-checks", action="store_true",
        help="Also run per-animal robustness checks (second-order Rayleigh, "
             "leave-one-animal-out, animal-level bootstrap) and save supplementary "
             "figures (figures/suppl_*.png) plus individual_summary.csv.",
    )
    p.add_argument(
        "--individual-checks-filter-low-n", action="store_true",
        help="When --individual-checks is set, exclude animals with <3 (second-order) "
             "or <5 (Wallraff) response trials (default: include all with ≥1).",
    )
    p.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel workers for subject processing (default: all CPUs).",
    )
    return p


# ══════════════════════════════════════════════════════════════════════
# Auto-Threshold: KDE Valley Detection (Priority 1)
# ══════════════════════════════════════════════════════════════════════


def _compute_kde_valley_threshold(vmax_values: np.ndarray) -> float | None:
    """Locate the valley between the low-speed cluster and the escape-burst
    cluster by finding the local minimum of a KDE density curve.

    Steps:
        1. Fit ``scipy.stats.gaussian_kde`` with a slightly narrow bandwidth
           to preserve local features (``bw_method=0.3``).
        2. Evaluate the density on ``[0, _KDE_SEARCH_UPPER]``.
        3. Use ``argrelmax`` to find peaks and ``argrelmin`` to find troughs.
        4. Return the deepest trough that sits *between* the first two peaks
           (the "quiet" peak and the "burst" peak).

    Returns
    -------
    float | None
        The valley threshold in mm/s, or ``None`` if no valid valley is found.
    """
    vmax_clean = vmax_values[~np.isnan(vmax_values)]
    if len(vmax_clean) < 10:
        log.warning("KDE valley: only %d valid samples — skipping.", len(vmax_clean))
        return None

    try:
        kde = gaussian_kde(vmax_clean, bw_method=0.3)
        x = np.linspace(0, _KDE_SEARCH_UPPER, 1000)
        density = kde(x)

        # ── Find peaks and valleys ──
        peak_idx = argrelmax(density, order=5)[0]
        valley_idx = argrelmin(density, order=5)[0]

        if len(peak_idx) < 2:
            log.warning("KDE valley: fewer than 2 peaks detected (%d) — skipping.", len(peak_idx))
            return None

        # The first two peaks represent the two main modes
        first_peak_x = x[peak_idx[0]]
        second_peak_x = x[peak_idx[1]]
        log.info("KDE valley: peaks at %.1f and %.1f mm/s", first_peak_x, second_peak_x)

        # Filter valleys that lie between the two peaks
        between = valley_idx[(x[valley_idx] > first_peak_x) & (x[valley_idx] < second_peak_x)]

        if len(between) == 0:
            log.warning("KDE valley: no trough found between peaks — skipping.")
            return None

        # Pick the deepest trough (lowest density) as the threshold
        deepest = between[np.argmin(density[between])]
        threshold = float(x[deepest])
        log.info("KDE valley threshold: %.1f mm/s (density=%.6f)", threshold, density[deepest])
        return threshold

    except Exception as exc:
        log.warning("KDE valley detection failed (%s) — skipping.", exc)
        return None


# ══════════════════════════════════════════════════════════════════════
# Auto-Threshold: Log-Space GMM (Priority 2)
# ══════════════════════════════════════════════════════════════════════


def _compute_log_gmm_threshold(
    vmax_values: np.ndarray,
    min_speed_floor: float = 10.0,
) -> tuple[float, float] | None:
    """Fit a **2-component** GMM in log-space on moving trials (v_max >= min_speed_floor)
    to separate Walking (spontaneous movement) from Escape bursts.

    Filtering out the stationary noise floor (< 10 mm/s) avoids spending GMM components
    on near-zero noise, yielding the clean empirical escape threshold (~96-98 mm/s).

    Returns
    -------
    tuple[float, float] | None
        ``(start_threshold, escape_threshold)`` in mm/s, or ``None`` if
        fitting fails or data is insufficient (< 15 valid moving samples).
    """
    vmax_clean = vmax_values[~np.isnan(vmax_values) & (vmax_values >= min_speed_floor)]
    if len(vmax_clean) < 15:
        log.warning("Log-GMM: only %d valid active samples (>= %.1f mm/s) — skipping.", len(vmax_clean), min_speed_floor)
        return None

    try:
        log_vmax = np.log(vmax_clean)
        X = log_vmax.reshape(-1, 1)

        gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
        gmm.fit(X)

        # Sort components by mean in log-space: low (walking) / high (escape)
        order = np.argsort(gmm.means_.ravel())
        means = gmm.means_.ravel()[order]
        vars_ = gmm.covariances_.ravel()[order]
        weights = gmm.weights_.ravel()[order]

        def _pair_diff(log_x: float) -> float:
            return (weights[0] * norm.pdf(log_x, means[0], np.sqrt(vars_[0]))
                    - weights[1] * norm.pdf(log_x, means[1], np.sqrt(vars_[1])))

        log_escape = brentq(_pair_diff, means[0], means[1])
        escape_threshold = float(np.exp(log_escape))
        start_threshold = float(min_speed_floor)

        log.info("Log-GMM 2-component (moving >= %.1f mm/s): start=%.1f, escape=%.1f mm/s  "
                 "(mu_walk=%.1f, mu_esc=%.1f mm/s)",
                 min_speed_floor, start_threshold, escape_threshold,
                 np.exp(means[0]), np.exp(means[1]))
        return start_threshold, escape_threshold

    except Exception as exc:
        log.warning("Log-GMM fitting failed (%s) — skipping.", exc)
        return None


# ══════════════════════════════════════════════════════════════════════
# Auto-Threshold: IQR-Adaptive Truncation GMM (Priority 3)
# ══════════════════════════════════════════════════════════════════════


def _compute_iqr_gmm_threshold(vmax_values: np.ndarray) -> float | None:
    """Fit a 2-component GMM on V_max values truncated by the IQR rule.

    Uses ``Q3 + 1.5 * IQR`` as an adaptive upper bound derived from the data
    itself, replacing the previous hard-coded 300 mm/s ceiling.  The GMM then
    operates in the original physical domain on the truncated data.

    Returns
    -------
    float | None
        The threshold in mm/s, or ``None`` if fitting fails or data is
        insufficient (< 10 valid samples after truncation).
    """
    vmax_clean = vmax_values[~np.isnan(vmax_values)]
    if len(vmax_clean) < 10:
        log.warning("IQR-GMM: only %d valid samples — skipping.", len(vmax_clean))
        return None

    try:
        q1 = np.percentile(vmax_clean, 25)
        q3 = np.percentile(vmax_clean, 75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr

        vmax_truncated = vmax_clean[vmax_clean < upper_bound]

        if len(vmax_truncated) < 10:
            log.warning("IQR-GMM: only %d samples below IQR upper bound (%.1f) — skipping.",
                        len(vmax_truncated), upper_bound)
            return None

        X = vmax_truncated.reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
        gmm.fit(X)

        # Sort components by mean
        order = np.argsort(gmm.means_.ravel())
        mu0, mu1 = gmm.means_.ravel()[order]
        var0, var1 = gmm.covariances_.ravel()[order]
        w0, w1 = gmm.weights_.ravel()[order]

        def _diff(x: float) -> float:
            return w0 * norm.pdf(x, mu0, np.sqrt(var0)) - w1 * norm.pdf(x, mu1, np.sqrt(var1))

        threshold = brentq(_diff, mu0, mu1)
        log.info("IQR-GMM threshold: %.1f mm/s  (Q1=%.1f, Q3=%.1f, IQR=%.1f, upper=%.1f, mu0=%.1f, mu1=%.1f)",
                 threshold, q1, q3, iqr, upper_bound, mu0, mu1)
        return float(threshold)

    except Exception as exc:
        log.warning("IQR-GMM fitting failed (%s) — skipping.", exc)
        return None


# ══════════════════════════════════════════════════════════════════════
# Parallel Processing Helper
# ══════════════════════════════════════════════════════════════════════


def _process_subject(subject_name: str, sessions: list[tuple[Path, Path]]) -> pd.DataFrame:
    """Process one subject: load → preprocess → classify. Returns DataFrame with subject_id."""
    try:
        all_meta, all_windows, all_anchors, all_kin, _ = load_and_concat_sessions(sessions)
        df = preprocess(all_meta, all_windows, all_anchors, all_kin)
        df["global_trial_index"] = df["global_trial_id"]
        df = label_trials(df)
        df["subject_id"] = subject_name
        return df
    except Exception as exc:
        log.error("Failed to process %s: %s", subject_name, exc)
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"--input-dir does not exist: {input_dir}")

    # ── Discover subjects ──
    subjects = scan_and_pair_sessions(input_dir)
    if not subjects:
        log.error("No valid (events, kinematics) pairs found in %s", input_dir)
        return

    # ── Per-subject parallel processing ──
    n_workers = args.workers if args.workers else cpu_count()
    log.info("Processing %d subjects with %d workers...", len(subjects), n_workers)

    if n_workers == 1:
        # Single-threaded fallback
        population_parts = [_process_subject(name, sess) for name, sess in subjects.items()]
    else:
        # Parallel processing
        with Pool(processes=n_workers) as pool:
            population_parts = pool.starmap(_process_subject, subjects.items())

    # Filter out empty DataFrames
    population_parts = [df for df in population_parts if not df.empty]

    if not population_parts:
        log.error("No data processed.")
        return

    # ── Global concatenation ──
    all_data = pd.concat(population_parts, ignore_index=True)

    # ──────────────────────────────────────────────────────────────────
    # Adaptive threshold & data tagging
    # ──────────────────────────────────────────────────────────────────

    # Build trial-level view (needed for threshold and escape rate)
    trial_level = (
        all_data.groupby(["subject_id", "global_trial_index"])
        .agg(v_max=("v_max", "first"), response_type=("response_type", "first"))
        .reset_index()
    )
    all_vmax = trial_level["v_max"].dropna().values  # ALL trials — 3-component GMM handles separation

    # ── Compute all candidate thresholds ──
    log.info("Computing adaptive thresholds on %d trials (all types)...", len(all_vmax))

    kde_threshold = _compute_kde_valley_threshold(all_vmax)
    log_gmm_result = _compute_log_gmm_threshold(all_vmax)
    iqr_gmm_threshold = _compute_iqr_gmm_threshold(all_vmax) if _ENABLE_IQR_GMM else None

    # Unpack Log-GMM result: (start_threshold, escape_threshold) or None
    gmm_start_threshold: float | None = None
    gmm_escape_threshold: float | None = None
    if log_gmm_result is not None:
        gmm_start_threshold, gmm_escape_threshold = log_gmm_result

    log.info("Candidate thresholds:  KDE=%.1f  Log-GMM start=%s escape=%s  IQR-GMM=%s",
             kde_threshold if kde_threshold else -1,
             f"{gmm_start_threshold:.1f}" if gmm_start_threshold else "None",
             f"{gmm_escape_threshold:.1f}" if gmm_escape_threshold else "None",
             f"{iqr_gmm_threshold:.1f}" if iqr_gmm_threshold else "None")

    # ── Priority cascade for tagging: KDE → Log-GMM → (IQR-GMM) → hardcoded ──
    if _USE_ADAPTIVE_THRESHOLD:
        if kde_threshold is not None:
            auto_vmax_threshold = kde_threshold
            method = "KDE valley"
        elif gmm_escape_threshold is not None:
            auto_vmax_threshold = gmm_escape_threshold
            method = "Log-GMM (escape)"
        elif _ENABLE_IQR_GMM and iqr_gmm_threshold is not None:
            auto_vmax_threshold = iqr_gmm_threshold
            method = "IQR-GMM"
        else:
            auto_vmax_threshold = _FALLBACK_VMAX_THRESHOLD
            method = "hardcoded fallback"
        log.info("Adaptive threshold selected: %.1f mm/s  [method=%s]", auto_vmax_threshold, method)
    else:
        auto_vmax_threshold = ESCAPE_VMAX_THRESHOLD
        method = "config (fixed)"
        log.info("Using fixed threshold from config: %.1f mm/s  [adaptive disabled]", auto_vmax_threshold)

    all_data["is_valid_escape"] = all_data["v_max"] >= auto_vmax_threshold

    # ── Per-subject escape and prewalk response rates ──
    trial_level = (
        all_data.groupby(["subject_id", "global_trial_index"])
        .agg(response_type=("response_type", "first"),
             is_valid_escape=("is_valid_escape", "first"))
        .reset_index()
    )

    subject_rates = (
        trial_level.groupby("subject_id")
        .agg(
            total_trials=("global_trial_index", "count"),
            valid_escape_trials=("is_valid_escape", "sum"),
            escape_trials=("response_type", lambda s: (s == "Escape").sum()),
            prewalk_trials=("response_type", lambda s: (s == "PreWalk").sum()),
            no_response_trials=("response_type", lambda s: (s == "NoResponse").sum()),
        )
        .assign(
            escape_rate=lambda d: d["valid_escape_trials"] / d["total_trials"],
            prewalk_rate=lambda d: d["prewalk_trials"] / d["total_trials"],
            no_response_rate=lambda d: d["no_response_trials"] / d["total_trials"],
            prewalk_fraction=lambda d: np.where(
                (d["escape_trials"] + d["prewalk_trials"]) > 0,
                d["prewalk_trials"] / (d["escape_trials"] + d["prewalk_trials"]),
                0.0
            ),
        )
        .reset_index()
    )

    log.info("Per-subject response rates (auto-threshold=%.1f mm/s):", auto_vmax_threshold)
    for _, row in subject_rates.iterrows():
        log.info("  %s: Escape=%.1f%% (%d/%d), PreWalk=%.1f%% (%d/%d), NoResp=%.1f%% (%d/%d)",
                 row["subject_id"],
                 row["escape_rate"] * 100, int(row["escape_trials"]), int(row["total_trials"]),
                 row["prewalk_rate"] * 100, int(row["prewalk_trials"]), int(row["total_trials"]),
                 row["no_response_rate"] * 100, int(row["no_response_trials"]), int(row["total_trials"]))

    # ──────────────────────────────────────────────────────────────────
    # CSV export  (summary + subject-level rates)
    # ──────────────────────────────────────────────────────────────────

    csv_path = output_dir / "population_summary.csv"
    export_summary_metrics(all_data, csv_path, groupby=["subject_id", "global_trial_index"])

    rates_csv_path = output_dir / "subject_escape_rates.csv"
    subject_rates.to_csv(rates_csv_path, index=False, float_format="%.4f")

    n_subjects = all_data["subject_id"].nunique()
    n_trials = trial_level.shape[0]
    n_escape = (trial_level["response_type"] == "Escape").sum()
    n_prewalk = (trial_level["response_type"] == "PreWalk").sum()
    n_no_resp = (trial_level["response_type"] == "NoResponse").sum()

    log.info("Population export complete:")
    log.info("  Subjects: %d", n_subjects)
    log.info("  Total trials: %d", n_trials)
    log.info("  Escape: %d | PreWalk: %d | NoResponse: %d", n_escape, n_prewalk, n_no_resp)
    log.info("  CSV: %s", csv_path)
    log.info("  Escape rates: %s", rates_csv_path)

    # ── Population visualization ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Create subdirectories
    pop_dir = output_dir / "population"
    heatmap_dir = pop_dir / "heatmap"
    pop_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    fig1 = plot_population_habituation(all_data)
    _safe_savefig(fig1, pop_dir / "habituation.svg", dpi=300, bbox_inches="tight")
    plt.close(fig1)

    # ── [Legacy Vmax distribution plots commented out per user instruction] ──
    # fig2 = plot_population_vmax_gmm(
    #     all_data,
    #     gmm_start_threshold=gmm_start_threshold,
    #     gmm_escape_threshold=gmm_escape_threshold,
    #     iqr_gmm_threshold=iqr_gmm_threshold,
    #     draw_fixed_thresholds=_DRAW_FIXED_THRESHOLDS,
    # )
    # _safe_savefig(fig2, pop_dir / "vmax_gmm.svg", dpi=300, bbox_inches="tight")
    # plt.close(fig2)
    #
    # fig3 = plot_population_vmax_response(
    #     all_data,
    #     auto_threshold=auto_vmax_threshold,
    #     draw_fixed_thresholds=_DRAW_FIXED_THRESHOLDS,
    # )
    # _safe_savefig(fig3, pop_dir / "vmax_response.svg", dpi=300, bbox_inches="tight")
    # plt.close(fig3)

    # ── New Population Vmax GMM Figure (Active/Moving Trials) ──
    fig_vmax_moving = plot_population_vmax_moving_gmm(
        all_data,
        gmm_escape_threshold=gmm_escape_threshold,
        draw_fixed_thresholds=_DRAW_FIXED_THRESHOLDS,
    )
    _safe_savefig(fig_vmax_moving, pop_dir / "vmax_moving_gmm.svg", dpi=300, bbox_inches="tight")
    plt.close(fig_vmax_moving)

    fig4 = plot_population_behavior_probability(all_data)
    _safe_savefig(fig4, pop_dir / "behavior_prob.svg", dpi=300, bbox_inches="tight")
    plt.close(fig4)

    fig4b = plot_prewalk_stillness(all_data)
    _safe_savefig(fig4b, pop_dir / "prewalk_stillness.svg", dpi=300, bbox_inches="tight")
    plt.close(fig4b)

    # ── Population speed kinetics by response type (Escape / PreWalk / NoResponse) ──
    fig_speed = plot_population_speed_kinetics(all_data)
    _safe_savefig(fig_speed, pop_dir / "speed_kinetics.svg", dpi=300, bbox_inches="tight")
    plt.close(fig_speed)

    # ── Population spaghetti kinetics by response type ──
    fig_spaghetti = plot_population_spaghetti_kinetics(all_data)
    _safe_savefig(fig_spaghetti, pop_dir / "spaghetti_kinetics.svg", dpi=300, bbox_inches="tight")
    plt.close(fig_spaghetti)

    # ── Heatmap figures go to heatmap/ and heatmap/full_trial/ subdirectories ──
    heatmap_dir = pop_dir / "heatmap"
    full_trial_dir = heatmap_dir / "full_trial"
    pop_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    full_trial_dir.mkdir(parents=True, exist_ok=True)

    # 1. Standard / zoomed heatmaps (in heatmap/)
    fig_heatmap = plot_spaghetti_kinetics_heatmap(all_data)
    _safe_savefig(fig_heatmap, heatmap_dir / "spaghetti_density_heatmap.svg", dpi=300, bbox_inches="tight")
    plt.close(fig_heatmap)

    fig_trial_ttc = plot_trial_stacked_heatmap(all_data, align="ttc")
    _safe_savefig(fig_trial_ttc, heatmap_dir / "trial_stacked_heatmap_ttc.svg", dpi=300, bbox_inches="tight")
    plt.close(fig_trial_ttc)

    fig_trial_onset = plot_trial_stacked_heatmap(all_data, align="onset")
    _safe_savefig(fig_trial_onset, heatmap_dir / "trial_stacked_heatmap_onset.svg", dpi=300, bbox_inches="tight")
    plt.close(fig_trial_onset)

    # 2. Full-trial high-resolution heatmaps (in heatmap/full_trial/)
    geom_cfg = get_geometry()
    hm_cfg = getattr(geom_cfg, "heatmap", None)
    ft_cfg = getattr(hm_cfg, "full_trial", None) if hm_cfg else None

    ft_t_window_ttc = tuple(float(x) for x in getattr(ft_cfg, "t_window_ttc", [-3.5, 1.5])) if ft_cfg else (-3.5, 1.5)
    ft_t_window_onset = tuple(float(x) for x in getattr(ft_cfg, "t_window_onset", [-3.5, 1.5])) if ft_cfg else (-3.5, 1.5)
    ft_t_window_density = tuple(float(x) for x in getattr(ft_cfg, "t_window_density", [-3500.0, 1500.0])) if ft_cfg else (-3500.0, 1500.0)
    ft_t_bin_s = float(getattr(ft_cfg, "t_bin_s", 0.005)) if ft_cfg else 0.005
    ft_dt_ms = float(getattr(ft_cfg, "dt_ms", 2.0)) if ft_cfg else 2.0

    fig_ft_heatmap = plot_spaghetti_kinetics_heatmap(
        all_data,
        t_window=ft_t_window_density,
        dt=ft_dt_ms,
        orientation="vertical",
        figsize=(8.0, 7.5),
    )
    _safe_savefig(fig_ft_heatmap, full_trial_dir / "spaghetti_density_heatmap.svg", dpi=300, bbox_inches="tight")
    plt.close(fig_ft_heatmap)

    fig_ft_trial_ttc = plot_trial_stacked_heatmap(
        all_data,
        align="ttc",
        t_window=ft_t_window_ttc,
        t_bin_s=ft_t_bin_s,
        orientation="vertical",
        figsize=(8.0, 7.5),
    )
    _safe_savefig(fig_ft_trial_ttc, full_trial_dir / "trial_stacked_heatmap_ttc.svg", dpi=300, bbox_inches="tight")
    plt.close(fig_ft_trial_ttc)

    fig_ft_trial_onset = plot_trial_stacked_heatmap(
        all_data,
        align="onset",
        t_window=ft_t_window_onset,
        t_bin_s=ft_t_bin_s,
        orientation="vertical",
        figsize=(8.0, 7.5),
    )
    _safe_savefig(fig_ft_trial_onset, full_trial_dir / "trial_stacked_heatmap_onset.svg", dpi=300, bbox_inches="tight")
    plt.close(fig_ft_trial_onset)

    fig5 = plot_escape_angle_distribution(all_data)
    _safe_savefig(fig5, pop_dir / "escape_angle_distribution.svg", dpi=300, bbox_inches="tight")
    plt.close(fig5)

    fig6 = plot_population_polar_histogram(all_data)
    _safe_savefig(fig6, pop_dir / "polar_direction_histogram.svg", dpi=300, bbox_inches="tight")
    plt.close(fig6)

    fig7 = plot_population_pre_movement_prewalk(all_data)
    _safe_savefig(fig7, pop_dir / "pre_movement_prewalk.svg", dpi=300, bbox_inches="tight")
    plt.close(fig7)

    # ── Individual-level robustness checks (pseudo-replication guard) ──
    if args.individual_checks:
        from cercus.analysis.individual import run_individual_checks

        log.info("Running per-animal robustness checks (--individual-checks)...")
        filter_low_n = getattr(args, "individual_checks_filter_low_n", False)
        run_individual_checks(all_data, output_dir, filter_low_n=filter_low_n)

    log.info("Figures saved to %s/: habituation.svg, vmax_moving_gmm.svg, behavior_prob.svg, prewalk_stillness.svg, speed_kinetics.svg, spaghetti_kinetics.svg, escape_angle_distribution.svg, polar_direction_histogram.svg, pre_movement_prewalk.svg", pop_dir.name)
    log.info("Heatmaps saved to %s/heatmap/ and %s/heatmap/full_trial/: spaghetti_density_heatmap.svg, trial_stacked_heatmap_ttc.svg, trial_stacked_heatmap_onset.svg", pop_dir.name, pop_dir.name)
    log.info("All output in: %s", output_dir)


if __name__ == "__main__":
    main()
