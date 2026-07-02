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
        "--save", required=True,
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
    save_path = Path(args.save)

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
    if args.escape_only:
        all_data = all_data[all_data["response_type"] == "Escape"].copy()
        log.info("Escape-only filter activated. Noise isolated.")

    n_subjects = all_data["subject_id"].nunique()
    n_trials = all_data.groupby(["subject_id", "global_trial_index"]).ngroups
    log.info("Population assembled: %d subjects, %d valid trials", n_subjects, n_trials)

    # ── Generate fixed unified trajectory overlay ──
    log.info("Generating fixed unified trajectory overlay...")
    fig = plot_global_trajectory_overlay_fixed(all_data)

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
