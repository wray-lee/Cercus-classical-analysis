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

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.signal import argrelmin, argrelmax
from scipy.stats import gaussian_kde, norm
from sklearn.mixture import GaussianMixture

from pipeline.classifier import label_trials
from pipeline.io import export_summary_metrics, load_and_concat_sessions, scan_and_pair_sessions
from pipeline.kinematics import preprocess
from pipeline.visualization import (
    plot_escape_angle_distribution,
    plot_population_behavior_probability,
    plot_population_habituation,
    plot_population_polar_histogram,
    plot_population_spaghetti_kinetics,
    plot_population_speed_kinetics,
    plot_population_vmax_gmm,
    plot_population_vmax_response,
    plot_prewalk_stillness,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

_FALLBACK_VMAX_THRESHOLD = 120.0  # mm/s, used when all auto methods fail
_KDE_SEARCH_UPPER = 400.0         # mm/s, upper bound for KDE valley search
_ENABLE_IQR_GMM = False           # set True to also compute IQR-GMM as a candidate
_DRAW_FIXED_THRESHOLDS = True     # set False to hide start/vmax reference lines on vmax plots


# ══════════════════════════════════════════════════════════════════════
# CLI Parser
# ══════════════════════════════════════════════════════════════════════


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
        "--output-dir", required=True,
        help="Directory for all output files (CSV + figures). Created if it doesn't exist.",
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
) -> tuple[float, float] | None:
    """Fit a **3-component** GMM in log-space and return two physical-domain
    thresholds: ``(start_threshold, escape_threshold)``.

    * ``start_threshold`` — boundary between the no-response cluster and the
      weak-movement cluster.
    * ``escape_threshold`` — boundary between weak movement and escape bursts.

    Log-transform compresses the heavy right tail of the V_max distribution,
    reducing the leverage of extreme outliers without discarding data.

    Returns
    -------
    tuple[float, float] | None
        ``(start_threshold, escape_threshold)`` in mm/s, or ``None`` if
        fitting fails or data is insufficient (< 15 valid samples).
    """
    vmax_clean = vmax_values[~np.isnan(vmax_values) & (vmax_values > 0)]
    if len(vmax_clean) < 15:
        log.warning("Log-GMM: only %d valid positive samples — skipping.", len(vmax_clean))
        return None

    try:
        log_vmax = np.log(vmax_clean)
        X = log_vmax.reshape(-1, 1)

        gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42)
        gmm.fit(X)

        # Sort components by mean in log-space: low / mid / high
        order = np.argsort(gmm.means_.ravel())
        means = gmm.means_.ravel()[order]
        vars_ = gmm.covariances_.ravel()[order]
        weights = gmm.weights_.ravel()[order]

        # Find intersections between consecutive pairs
        def _pair_diff(log_x: float, i: int, j: int) -> float:
            return (weights[i] * norm.pdf(log_x, means[i], np.sqrt(vars_[i]))
                    - weights[j] * norm.pdf(log_x, means[j], np.sqrt(vars_[j])))

        from scipy.optimize import brentq as _brentq

        log_start = _brentq(lambda x: _pair_diff(x, 0, 1), means[0], means[1])
        log_escape = _brentq(lambda x: _pair_diff(x, 1, 2), means[1], means[2])

        start_threshold = float(np.exp(log_start))
        escape_threshold = float(np.exp(log_escape))

        log.info("Log-GMM 3-component: start=%.1f, escape=%.1f mm/s  "
                 "(log-mu=[%.2f, %.2f, %.2f])",
                 start_threshold, escape_threshold, *means)
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
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"--input-dir does not exist: {input_dir}")

    # ── Discover subjects ──
    subjects = scan_and_pair_sessions(input_dir)
    if not subjects:
        log.error("No valid (events, kinematics) pairs found in %s", input_dir)
        return

    # ── Per-subject processing ──
    population_parts: list[pd.DataFrame] = []

    for subject_name, sessions in subjects.items():
        log.info("Processing subject: %s", subject_name)

        # 1. Load & timestamp alignment
        all_meta, all_windows, all_anchors, all_kin, _ = load_and_concat_sessions(sessions)
        df = preprocess(all_meta, all_windows, all_anchors, all_kin)
        df["global_trial_index"] = df["global_trial_id"]

        # 2. Ternary state routing
        df = label_trials(df)

        # 3. Inject subject_id for cross-animal key isolation
        df["subject_id"] = subject_name
        population_parts.append(df)

        n_trials = df["global_trial_index"].nunique()
        log.info("  %s: %d trials classified", subject_name, n_trials)

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

    all_data["is_valid_escape"] = all_data["v_max"] >= auto_vmax_threshold

    # ── Per-subject escape rate ──
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
        )
        .assign(escape_rate=lambda d: d["valid_escape_trials"] / d["total_trials"])
        .reset_index()
    )

    log.info("Per-subject escape rates (threshold=%.1f mm/s):", auto_vmax_threshold)
    for _, row in subject_rates.iterrows():
        log.info("  %s: %.1f%% (%d/%d)",
                 row["subject_id"], row["escape_rate"] * 100,
                 row["valid_escape_trials"], row["total_trials"])

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

    fig1 = plot_population_habituation(all_data)
    fig1.savefig(output_dir / "habituation.svg", dpi=300, bbox_inches="tight")
    plt.close(fig1)

    fig2 = plot_population_vmax_gmm(
        all_data,
        gmm_start_threshold=gmm_start_threshold,
        gmm_escape_threshold=gmm_escape_threshold,
        iqr_gmm_threshold=iqr_gmm_threshold,
        draw_fixed_thresholds=_DRAW_FIXED_THRESHOLDS,
    )
    fig2.savefig(output_dir / "vmax_gmm.svg", dpi=300, bbox_inches="tight")
    plt.close(fig2)

    fig3 = plot_population_vmax_response(
        all_data,
        auto_threshold=auto_vmax_threshold,
        draw_fixed_thresholds=_DRAW_FIXED_THRESHOLDS,
    )
    fig3.savefig(output_dir / "vmax_response.svg", dpi=300, bbox_inches="tight")
    plt.close(fig3)

    fig4 = plot_population_behavior_probability(all_data)
    fig4.savefig(output_dir / "behavior_prob.svg", dpi=300, bbox_inches="tight")
    plt.close(fig4)

    fig4b = plot_prewalk_stillness(all_data)
    fig4b.savefig(output_dir / "prewalk_stillness.svg", dpi=300, bbox_inches="tight")
    plt.close(fig4b)

    # ── Population speed kinetics by response type (Escape / PreWalk / NoResponse) ──
    fig_speed = plot_population_speed_kinetics(all_data)
    fig_speed.savefig(output_dir / "speed_kinetics.svg", dpi=300, bbox_inches="tight")
    plt.close(fig_speed)

    # ── Population spaghetti kinetics by response type ──
    fig_spaghetti = plot_population_spaghetti_kinetics(all_data)
    fig_spaghetti.savefig(output_dir / "spaghetti_kinetics.svg", dpi=300, bbox_inches="tight")
    plt.close(fig_spaghetti)

    fig5 = plot_escape_angle_distribution(all_data)
    fig5.savefig(output_dir / "escape_angle_distribution.svg", dpi=300, bbox_inches="tight")
    plt.close(fig5)

    fig6 = plot_population_polar_histogram(all_data)
    fig6.savefig(output_dir / "polar_direction_histogram.svg", dpi=300, bbox_inches="tight")
    plt.close(fig6)

    log.info("Figures saved: habituation.svg, vmax_gmm.svg, vmax_response.svg, behavior_prob.svg, prewalk_stillness.svg, speed_kinetics.svg, spaghetti_kinetics.svg, escape_angle_distribution.svg, polar_direction_histogram.svg")
    log.info("All output in: %s", output_dir)


if __name__ == "__main__":
    main()
