"""
Cercus Framework — Cross-Subject Population Batch Processor
============================================================
Scans an input directory, processes every subject through the full
preprocessing → classification pipeline, and exports a single unified
summary CSV with per-trial metrics for downstream statistical analysis.

Zero-rendering: no matplotlib, no visualization imports.

Usage:
    python population_analysis.py --input-dir path/to/data/ --output-csv population_summary.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from pipeline.classifier import label_trials
from pipeline.io import export_summary_metrics, load_and_concat_sessions, scan_and_pair_sessions
from pipeline.kinematics import preprocess

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


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
        "--output-csv", required=True,
        help="Path for the global summary CSV (e.g. population_summary.csv).",
    )
    return p


# ══════════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    input_dir = Path(args.input_dir)
    output_path = Path(args.output_csv)

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
        n_esc = (df["response_type"] == "Escape").sum() // max(1, len(df) // max(1, n_trials))
        log.info("  %s: %d trials classified", subject_name, n_trials)

    if not population_parts:
        log.error("No data processed.")
        return

    # ── Global concatenation & export ──
    all_data = pd.concat(population_parts, ignore_index=True)
    export_summary_metrics(all_data, output_path, groupby=["subject_id", "global_trial_index"])

    n_subjects = all_data["subject_id"].nunique()
    n_trials = all_data.groupby(["subject_id", "global_trial_index"]).ngroups
    n_escape = all_data.groupby(["subject_id", "global_trial_index"])["response_type"].first().value_counts().get("Escape", 0)
    n_prewalk = all_data.groupby(["subject_id", "global_trial_index"])["response_type"].first().value_counts().get("PreWalk", 0)
    n_no_resp = all_data.groupby(["subject_id", "global_trial_index"])["response_type"].first().value_counts().get("NoResponse", 0)

    log.info("Population export complete:")
    log.info("  Subjects: %d", n_subjects)
    log.info("  Total trials: %d", n_trials)
    log.info("  Escape: %d | PreWalk: %d | NoResponse: %d", n_escape, n_prewalk, n_no_resp)
    log.info("  Output: %s", output_path)


if __name__ == "__main__":
    main()
