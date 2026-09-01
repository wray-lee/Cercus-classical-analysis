"""Cercus Framework — Multisensory Trajectory Comparison
=====================================================
Loads three data directories (baseline-visual, baseline-wind, multisensory),
runs the full pipeline on each, and produces one figure with baselines
mirrored to −x and multisensory to +x.

Usage:
    python plot_multisensory_trajectories.py --bv path/bv --bw path/bw --ms path/ms --output fig.svg
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from functools import partial
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
    p.add_argument("--bv", required=True, help="Baseline-visual data directory")
    p.add_argument("--bw", required=True, help="Baseline-wind data directory")
    p.add_argument("--ms", required=True, help="Multisensory data directory")
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
    args = build_parser().parse_args(argv)
    bv_dir, bw_dir, ms_dir = Path(args.bv), Path(args.bw), Path(args.ms)
    save_path = Path(args.output)
    escape_only = args.escape_only

    for label, d in [("bv", bv_dir), ("bw", bw_dir), ("ms", ms_dir)]:
        if not d.is_dir():
            raise FileNotFoundError(f"--{label} does not exist: {d}")

    # Sequential outer load (inner per-subject pool handles parallelism)
    log.info("Loading 3 datasets...")
    df_bv = _load_dir(bv_dir, escape_only=escape_only)
    df_bw = _load_dir(bw_dir, escape_only=escape_only)
    df_ms = _load_dir(ms_dir, escape_only=escape_only)

    fig = plot_multisensory_trajectory_comparison(df_bv, df_bw, df_ms)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    log.info("Figure saved to %s", save_path)


if __name__ == "__main__":
    main()
