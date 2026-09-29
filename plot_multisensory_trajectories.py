"""Cercus Framework — Multisensory Trajectory Comparison
=====================================================
Loads whichever of baseline-visual, baseline-wind, and multisensory datasets
are provided and plots their trajectories on a full grid, with only the
corresponding mirrored half shaded and labeled.

Usage:
    python plot_multisensory_trajectories.py --bv path/bv --bw path/bw --ms path/ms --output fig.svg
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
from cercus.visualization import plot_multisensory_trajectory_comparison

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

_apply_publication_style()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cercus Multisensory Trajectory Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--bv", help="Baseline-visual data directory")
    p.add_argument("--bw", help="Baseline-wind data directory")
    p.add_argument("--ms", help="Multisensory data directory")
    p.add_argument("--output", required=True, help="Path to save the output figure")
    p.add_argument("--escape-only", action="store_true", help="Plot only Escape trials")
    return p


def _process_one_subject(args: tuple[str, list, bool]) -> pd.DataFrame:
    """Process one subject's sessions. For multiprocessing."""
    name, sessions, escape_only = args
    all_meta, all_windows, all_anchors, all_kin, _ = load_and_concat_sessions(sessions)
    df = preprocess(all_meta, all_windows, all_anchors, all_kin)
    df["global_trial_index"] = df["global_trial_id"]
    df = label_trials(df)
    df["subject_id"] = name
    if escape_only:
        df = df[df["response_type"] == "Escape"].copy()
    return df


def _load_dir(data_dir: Path, escape_only: bool = False) -> pd.DataFrame:
    """Scan, preprocess, classify all subjects under *data_dir* (parallel per subject)."""
    subjects = scan_and_pair_sessions(data_dir)
    if not subjects:
        log.warning("No valid session pairs in %s", data_dir)
        return pd.DataFrame()

    # Parallel per-subject processing
    args_list = [(name, sessions, escape_only) for name, sessions in subjects.items()]
    with Pool() as pool:
        parts = pool.map(_process_one_subject, args_list)

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True)
    log.info("  %s: %d subjects, %d trials", data_dir.name, len(subjects), out["global_trial_index"].nunique())
    return out


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    supplied = {
        label: Path(raw)
        for label, raw in (("bv", args.bv), ("bw", args.bw), ("ms", args.ms))
        if raw is not None
    }
    if not supplied:
        parser.error("Provide at least one of --bv, --bw, or --ms")

    for label, data_dir in supplied.items():
        if not data_dir.is_dir():
            parser.error(f"--{label} is not a directory: {data_dir}")

    # Sequential outer load (inner per-subject pool handles parallelism).
    log.info("Loading %d datasets...", len(supplied))
    datasets = {
        label: _load_dir(data_dir, escape_only=args.escape_only)
        for label, data_dir in supplied.items()
    }

    empty = pd.DataFrame()
    fig = plot_multisensory_trajectory_comparison(
        datasets.get("bv", empty), datasets.get("bw", empty), datasets.get("ms", empty),
        present=tuple(supplied),
    )
    save_path = Path(args.output)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    log.info("Figure saved to %s", save_path)


if __name__ == "__main__":
    main()
