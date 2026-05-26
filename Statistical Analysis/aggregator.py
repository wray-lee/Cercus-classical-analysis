"""
Aggregator — Multi-Level Data Assembly Across Sessions & Subjects
=================================================================
Walks a root directory tree to discover session-level outputs from
feature_extractor.py, assembles them into population-level datasets
for downstream statistical analysis.

Expected directory structure:
    root/
        subject_01/
            session_a/
                session_summary.csv
                session_timeseries.parquet
            session_b/
                ...
        subject_02/
            ...

Outputs:
    population_master.csv        — one row per trial, merged across all sessions
    population_timeseries.parquet — merged high-frequency time series

Usage:
    python aggregator.py --root path/to/data_root --output results/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def _discover_sessions(root: Path) -> list[dict]:
    """
    Walk the directory tree to find all session_summary.csv files.
    Infers subject_id from the parent directory name.

    Returns list of dicts: {subject_id, session_dir, summary_path, timeseries_path}
    """
    sessions = []
    for summary_path in sorted(root.rglob("session_summary.csv")):
        session_dir = summary_path.parent
        ts_path = session_dir / "session_timeseries.parquet"

        # Infer subject_id: one level up from session directory
        subject_id = session_dir.parent.name if session_dir.parent != root else "unknown"

        sessions.append({
            "subject_id": subject_id,
            "session_dir": str(session_dir),
            "summary_path": str(summary_path),
            "timeseries_path": str(ts_path) if ts_path.exists() else None,
        })

    log.info("Discovered %d sessions across %d subjects",
             len(sessions), len({s["subject_id"] for s in sessions}))
    return sessions


def _assemble_population_summary(sessions: list[dict]) -> pd.DataFrame:
    """
    Merge all session_summary.csv files into a single population DataFrame.
    Adds subject_id column if not already present.
    """
    parts = []
    for sess in sessions:
        df = pd.read_csv(sess["summary_path"])
        if "subject_id" not in df.columns:
            df["subject_id"] = sess["subject_id"]
        parts.append(df)

    if not parts:
        log.warning("No session_summary.csv files found.")
        return pd.DataFrame()

    master = pd.concat(parts, ignore_index=True)
    log.info("Population master: %d trials from %d subjects",
             len(master), master["subject_id"].nunique())
    return master


def _assemble_population_timeseries(sessions: list[dict]) -> pd.DataFrame:
    """
    Merge all session_timeseries.parquet files into a single high-frequency dataset.
    Adds subject_id column if not already present.
    """
    parts = []
    for sess in sessions:
        ts_path = sess["timeseries_path"]
        if ts_path is None:
            continue
        try:
            df = pd.read_parquet(ts_path)
        except Exception as exc:
            log.warning("Failed to read %s: %s", ts_path, exc)
            continue
        if "subject_id" not in df.columns:
            df["subject_id"] = sess["subject_id"]
        parts.append(df)

    if not parts:
        log.warning("No session_timeseries.parquet files found.")
        return pd.DataFrame()

    merged = pd.concat(parts, ignore_index=True)
    log.info("Population timeseries: %d frames from %d subjects",
             len(merged), merged["subject_id"].nunique())
    return merged


def aggregate(
    root: str | Path,
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full aggregation pipeline.

    Parameters
    ----------
    root : root directory to search for session outputs
    output_dir : directory to write population_master.csv and population_timeseries.parquet

    Returns
    -------
    master : DataFrame — one row per trial with scalar features
    timeseries : DataFrame — merged high-frequency time series
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Root directory not found: {root}")

    sessions = _discover_sessions(root)
    if not sessions:
        raise RuntimeError(f"No session_summary.csv files found under {root}")

    master = _assemble_population_summary(sessions)
    timeseries = _assemble_population_timeseries(sessions)

    # Save outputs
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        master_path = out / "population_master.csv"
        master.to_csv(master_path, index=False)
        log.info("Saved: %s", master_path)

        ts_path = out / "population_timeseries.parquet"
        timeseries.to_parquet(ts_path, index=False)
        log.info("Saved: %s", ts_path)

    return master, timeseries


# ══════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ══════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Aggregator — assemble population-level datasets from session outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--root", required=True, help="Root directory containing session outputs")
    p.add_argument("--output", default=".", help="Output directory (default: current dir)")
    return p


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    master, timeseries = aggregate(root=args.root, output_dir=args.output)
    return master, timeseries


if __name__ == "__main__":
    main()
