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
    plot_reaction_distance_panel,
    plot_spaghetti_kinetics_heatmap,
    plot_trial_stacked_heatmap,
)

from cercus.analysis.vmax_threshold import select_vmax_threshold
from cercus.config import get_config, get_geometry

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

_cfg = get_config()
_DRAW_FIXED_THRESHOLDS = bool(_cfg.visualization.draw_fixed_thresholds)              # set False to hide start/vmax reference lines on vmax plots
_INDIVIDUAL_CHECKS = True          # ponytail: hardcoded, flip here if needed
_INDIVIDUAL_CHECKS_FILTER_LOW_N = False


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
        "--workers", type=int, default=None,
        help="Number of parallel workers for subject processing (default: all CPUs).",
    )
    return p


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


def _render_and_save(job_tuple) -> str:
    """Wrapper for parallel figure rendering. job_tuple = (plot_func, output_path, args, kwargs)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_func, output_path, args, kwargs = job_tuple
    try:
        fig = plot_func(*args, **kwargs)
        if fig is None:  # figure not applicable to this dataset (missing cols)
            return ""
        _safe_savefig(fig, output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return str(output_path)
    except Exception as exc:
        log.error("Failed to render %s: %s", output_path.name, exc)
        return ""


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

    # ── Adaptive threshold selection — cascade lives in cercus.analysis.vmax_threshold ──
    log.info("Computing adaptive thresholds on %d trials (all types)...", len(all_vmax))
    auto_vmax_threshold, method, _thr_info = select_vmax_threshold(all_vmax)
    gmm_escape_threshold = _thr_info["gmm_escape_threshold"]


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
            preescape_trials=("response_type", lambda s: (s == "PreEscape").sum()),
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
    n_preescape = (trial_level["response_type"] == "PreEscape").sum()
    n_prewalk = (trial_level["response_type"] == "PreWalk").sum()
    n_no_resp = (trial_level["response_type"] == "NoResponse").sum()

    log.info("Population export complete:")
    log.info("  Subjects: %d", n_subjects)
    log.info("  Total trials: %d", n_trials)
    log.info("  Escape: %d | PreEscape: %d | PreWalk: %d | NoResponse: %d",
             n_escape, n_preescape, n_prewalk, n_no_resp)
    log.info("  CSV: %s", csv_path)
    log.info("  Escape rates: %s", rates_csv_path)

    # ── Population visualization ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Create subdirectories
    pop_dir = output_dir / "population"
    heatmap_dir = pop_dir / "heatmap"
    full_trial_dir = heatmap_dir / "full_trial"
    pop_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    full_trial_dir.mkdir(parents=True, exist_ok=True)

    # ── Prepare full-trial config ──
    geom_cfg = get_geometry()
    hm_cfg = getattr(geom_cfg, "heatmap", None)
    ft_cfg = getattr(hm_cfg, "full_trial", None) if hm_cfg else None
    ft_t_window_ttc = tuple(float(x) for x in getattr(ft_cfg, "t_window_ttc", [-3.5, 1.5])) if ft_cfg else (-3.5, 1.5)
    ft_t_window_onset = tuple(float(x) for x in getattr(ft_cfg, "t_window_onset", [-3.5, 1.5])) if ft_cfg else (-3.5, 1.5)
    ft_t_window_density = tuple(float(x) for x in getattr(ft_cfg, "t_window_density", [-3500.0, 1500.0])) if ft_cfg else (-3500.0, 1500.0)
    ft_t_bin_s = float(getattr(ft_cfg, "t_bin_s", 0.005)) if ft_cfg else 0.005
    ft_dt_ms = float(getattr(ft_cfg, "dt_ms", 2.0)) if ft_cfg else 2.0

    # ── Build plot job list: (func, output_path, args, kwargs) ──
    plot_jobs = [
        (plot_population_habituation, pop_dir / "habituation.svg", (all_data,), {}),
        (plot_population_vmax_moving_gmm, pop_dir / "vmax_moving_gmm.svg", (all_data,),
         {"gmm_escape_threshold": gmm_escape_threshold, "draw_fixed_thresholds": _DRAW_FIXED_THRESHOLDS}),
        (plot_population_behavior_probability, pop_dir / "behavior_prob.svg", (all_data,), {}),
        (plot_reaction_distance_panel, pop_dir / "reaction_distance_panel.svg", (all_data,), {}),
        (plot_prewalk_stillness, pop_dir / "prewalk_stillness.svg", (all_data,), {}),
        (plot_population_speed_kinetics, pop_dir / "speed_kinetics.svg", (all_data,), {}),
        (plot_population_spaghetti_kinetics, pop_dir / "spaghetti_kinetics.svg", (all_data,), {}),
        (plot_spaghetti_kinetics_heatmap, heatmap_dir / "spaghetti_density_heatmap.svg", (all_data,), {}),
        (plot_trial_stacked_heatmap, heatmap_dir / "trial_stacked_heatmap_ttc.svg", (all_data,), {"align": "ttc"}),
        (plot_trial_stacked_heatmap, heatmap_dir / "trial_stacked_heatmap_onset.svg", (all_data,), {"align": "onset"}),
        (plot_spaghetti_kinetics_heatmap, full_trial_dir / "spaghetti_density_heatmap.svg", (all_data,),
         {"t_window": ft_t_window_density, "dt": ft_dt_ms, "orientation": "vertical", "figsize": (8.0, 7.5)}),
        (plot_trial_stacked_heatmap, full_trial_dir / "trial_stacked_heatmap_ttc.svg", (all_data,),
         {"align": "ttc", "t_window": ft_t_window_ttc, "t_bin_s": ft_t_bin_s, "orientation": "vertical", "figsize": (8.0, 7.5)}),
        (plot_trial_stacked_heatmap, full_trial_dir / "trial_stacked_heatmap_onset.svg", (all_data,),
         {"align": "onset", "t_window": ft_t_window_onset, "t_bin_s": ft_t_bin_s, "orientation": "vertical", "figsize": (8.0, 7.5)}),
        (plot_escape_angle_distribution, pop_dir / "escape_angle_distribution.svg", (all_data,), {}),
        (plot_population_polar_histogram, pop_dir / "polar_direction_histogram.svg", (all_data,), {}),
        (plot_population_pre_movement_prewalk, pop_dir / "pre_movement_prewalk.svg", (all_data,), {}),
    ]

    log.info("Rendering %d figures with %d workers...", len(plot_jobs), n_workers)

    if n_workers == 1:
        # Sequential fallback
        for job in plot_jobs:
            _render_and_save(job)
    else:
        # Parallel rendering
        with Pool(processes=n_workers) as pool:
            pool.map(_render_and_save, plot_jobs)

    # ── Individual-level robustness checks (pseudo-replication guard) ──
    if _INDIVIDUAL_CHECKS:
        from cercus.analysis.individual import run_individual_checks

        log.info("Running per-animal robustness checks...")
        run_individual_checks(all_data, output_dir, filter_low_n=_INDIVIDUAL_CHECKS_FILTER_LOW_N)

    log.info("Figures saved to %s/: habituation.svg, vmax_moving_gmm.svg, behavior_prob.svg, prewalk_stillness.svg, speed_kinetics.svg, spaghetti_kinetics.svg, escape_angle_distribution.svg, polar_direction_histogram.svg, pre_movement_prewalk.svg", pop_dir.name)
    log.info("Heatmaps saved to %s/heatmap/ and %s/heatmap/full_trial/: spaghetti_density_heatmap.svg, trial_stacked_heatmap_ttc.svg, trial_stacked_heatmap_onset.svg", pop_dir.name, pop_dir.name)
    log.info("All output in: %s", output_dir)


if __name__ == "__main__":
    main()
