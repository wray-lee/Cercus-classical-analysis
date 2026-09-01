"""
Cercus Framework — Fixed Unified Trajectory Overlay (All Paradigms)
===================================================================
Scans all subjects, runs the full preprocessing and classification pipeline,
and produces a single high-density figure with every trial drawn on one
unified physical-coordinate grid.

This variant uses **local coordinate reconstruction** to eliminate the radial
distortion caused by global cumulative heading drift in the pre-computed
x/y columns.

Usage:
    python plot_all_trajectories_fixed.py --input-dir path/to/data/ --save fixed_trajectories.svg
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from multiprocessing import Pool

import matplotlib.pyplot as plt
import pandas as pd

from pipeline.classifier import label_trials
from pipeline.constants import _apply_publication_style
from pipeline.io import load_and_concat_sessions, scan_and_pair_sessions
from pipeline.kinematics import preprocess
from pipeline.visualization import plot_global_trajectory_overlay_fixed

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

_apply_publication_style()


def _process_subject(args: tuple[str, list]) -> pd.DataFrame:
    """Process one subject. For multiprocessing."""
    subject_name, sessions = args
    all_meta, all_windows, all_anchors, all_kin, _ = load_and_concat_sessions(sessions)
    df = preprocess(all_meta, all_windows, all_anchors, all_kin)
    df["global_trial_index"] = df["global_trial_id"]
    df = label_trials(df)
    df["subject_id"] = subject_name
    return df


# ══════════════════════════════════════════════════════════════════════
# CLI Parser
# ══════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cercus Fixed Unified Trajectory Overlay — all paradigms, local coordinates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input-dir", required=True,
        help="Root directory containing per-subject session CSVs.",
    )
    p.add_argument(
        "--output", required=True,
        help="Path to save the output figure (e.g. fixed_trajectories.svg).",
    )
    p.add_argument(
        "--escape-only", action="store_true",
        help="Strictly filter and plot ONLY trials classified as 'Escape'.",
    )
    return p


# ══════════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    input_dir = Path(args.input_dir)
    save_path = Path(args.output)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"--input-dir does not exist: {input_dir}")

    # ── Discover subjects ──
    subjects = scan_and_pair_sessions(input_dir)
    if not subjects:
        log.error("No valid (events, kinematics) pairs found in %s", input_dir)
        return

    # ── Parallel per-subject processing ──
    log.info("Processing %d subjects in parallel...", len(subjects))
    args_list = list(subjects.items())
    with Pool() as pool:
        population_parts = pool.map(_process_subject, args_list)

    if not population_parts:
        log.error("No data processed.")
        return

    # ── Global concatenation ──
    all_data = pd.concat(population_parts, ignore_index=True)
    if args.escape_only:
        all_data = all_data[all_data["response_type"] == "Escape"].copy()
        log.info("Escape-only filter activated. Noise isolated.")

    n_subjects = all_data["subject_id"].nunique()
    n_trials = all_data.groupby(["subject_id", "global_trial_index"]).ngroups
    log.info("Population assembled: %d subjects, %d trials", n_subjects, n_trials)

    # ── Generate fixed unified trajectory overlay ──
    log.info("Generating fixed unified trajectory overlay...")
    fig = plot_global_trajectory_overlay_fixed(all_data)

    # ── Save ──
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    log.info("Figure saved to %s", save_path)


if __name__ == "__main__":
    main()
