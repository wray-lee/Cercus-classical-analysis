"""
Cercus Framework — CLI Entry Point
===================================
Cross-session behavioral analysis with ternary classification and
publication-grade figure generation.

Usage:
    python main.py --input-dir path/to/data/ --save figures/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.classifier import label_trials
from pipeline.constants import (
    _apply_publication_style,
    POST_STIM_BUFFER_MS,
)
from pipeline.io import export_summary_metrics, load_and_concat_sessions, scan_and_pair_sessions
from pipeline.kinematics import preprocess
from pipeline.visualization import (
    plot_behavior_probability,
    plot_habituation_curve,
    plot_single_trial_kinetics,
    plot_spaghetti_kinetics,
    plot_speed_kinetics,
    plot_trajectory_overlay,
    plot_vmax_distribution,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

_apply_publication_style()


# ══════════════════════════════════════════════════════════════════════
# CLI Parser
# ══════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cercus framework — cross-session behavioral analysis with ternary classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input-dir", required=True,
        help="Directory containing *_session_*_events.csv and *_session_*_kinematics.csv files.",
    )
    p.add_argument("--control-type", default="baseline_visual_test", help="Trial type for control condition.")
    p.add_argument("--stim-type", default="looming_wind", help="Trial type for stimulus condition.")
    p.add_argument("--save", default=None, help="Directory to save PNG figures. Omit to show interactively.")
    return p


# ══════════════════════════════════════════════════════════════════════
# Per-Response-Type Figure Generation
# ══════════════════════════════════════════════════════════════════════


def _generate_response_figures(
    df_slice: pd.DataFrame,
    output_dir: Path,
    control_type: str,
    stim_type: str,
    label: str,
) -> None:
    """Generate trajectory, speed kinetics, and spaghetti figures for a response-type slice."""
    if df_slice.empty:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    fig_traj = plot_trajectory_overlay(df_slice, control_type=control_type)
    fig_traj.savefig(output_dir / "trajectory_overlay.png", dpi=300, bbox_inches="tight")
    plt.close(fig_traj)

    fig_speed = plot_speed_kinetics(df_slice, control_type=control_type, stim_type=stim_type)
    fig_speed.savefig(output_dir / "speed_kinetics.png", dpi=300, bbox_inches="tight")
    plt.close(fig_speed)

    fig_spaghetti = plot_spaghetti_kinetics(df_slice, control_type=control_type, stim_type=stim_type)
    fig_spaghetti.savefig(output_dir / "spaghetti_kinetics.png", dpi=300, bbox_inches="tight")
    plt.close(fig_spaghetti)

    log.info("%s figures saved to %s", label, output_dir)


def _generate_individual_trial_figures(
    df_slice: pd.DataFrame,
    output_dir: Path,
    response_type: str,
) -> None:
    """Generate per-trial speed kinetics figures for a response-type slice."""
    if df_slice.empty:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for tid, grp in df_slice.groupby("global_trial_index"):
        trial_data = grp.sort_values("t_rel")
        row = grp.iloc[0]
        lat = float(row["latency_ms"])
        vmax = float(row["v_max"])

        fig_trial = plot_single_trial_kinetics(
            trial_data, lat, vmax, int(tid), response_type=response_type,
        )
        fig_trial.savefig(output_dir / f"trial_{int(tid)}_{response_type.lower()}.png",
                          dpi=300, bbox_inches="tight")
        plt.close(fig_trial)

    n_exported = df_slice["global_trial_index"].nunique()
    log.info("Exported %d individual %s trial figures to %s", n_exported, response_type, output_dir)


# ══════════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"--input-dir does not exist: {input_dir}")

    # ── Module 1: scan, pair, concatenate ──
    subjects = scan_and_pair_sessions(input_dir)
    if not subjects:
        log.error("No valid (events, kinematics) pairs found in %s", input_dir)
        return

    for subject_name, sessions in subjects.items():
        log.info("═══ Processing subject: %s ═══", subject_name)

        all_meta, all_windows, all_anchors, all_kin, trial_to_global = load_and_concat_sessions(sessions)

        # ── Preprocess ──
        df = preprocess(all_meta, all_windows, all_anchors, all_kin)
        df["global_trial_index"] = df["global_trial_id"]

        # ── Module 2: ternary classification ──
        df = label_trials(df)

        types_present = set(df["type"].dropna().unique())
        log.info("Trial types in data: %s", types_present)

        # ── Summary metrics CSV export ──
        if args.save:
            subject_dir = Path(args.save) / subject_name
            export_summary_metrics(df, subject_dir / f"{subject_name}_summary_metrics.csv")

        # ── Module 3: output routing ──
        df_escape = df[df["response_type"] == "Escape"].copy()
        df_prewalk = df[df["response_type"] == "PreWalk"].copy()
        df_no_response = df[df["response_type"] == "NoResponse"].copy()

        n_esc = df_escape["global_trial_id"].nunique()
        n_pw = df_prewalk["global_trial_id"].nunique()
        n_nr = df_no_response["global_trial_id"].nunique()
        log.info("Split: %d Escape, %d PreWalk, %d NoResponse", n_esc, n_pw, n_nr)

        # Peak diagnostic for discarded trials
        if not df_no_response.empty:
            diag_window = df_no_response[
                (df_no_response["t_rel"] > 0) & (df_no_response["t_rel"] <= POST_STIM_BUFFER_MS)
            ]
            if not diag_window.empty:
                peaks = diag_window.groupby("global_trial_id")["speed"].max()
                log.info("====== NoResponse peak diagnostic (0–%.0f ms) ======", POST_STIM_BUFFER_MS)
                for tid, pmax in peaks.items():
                    log.info("  Trial %s peak: %.1f mm/s", tid, pmax)
                log.info("  Mean peak: %.1f mm/s", peaks.mean())
                log.info("════════════════════════════════════════════════")

        if args.save:
            subject_dir = Path(args.save) / subject_name

            # ── Escape figures ──
            _generate_response_figures(
                df_escape, subject_dir / "response", args.control_type, args.stim_type, "Escape",
            )

            # ── PreWalk figures (NEW — full parity with Escape) ──
            _generate_response_figures(
                df_prewalk, subject_dir / "prewalk", args.control_type, args.stim_type, "PreWalk",
            )

            # ── NoResponse figures ──
            _generate_response_figures(
                df_no_response, subject_dir / "no_response", args.control_type, args.stim_type, "NoResponse",
            )

            # ── Behavior probability distribution ──
            fig_prob = plot_behavior_probability(df)
            fig_prob.savefig(subject_dir / "behavior_probability_distribution.png", dpi=300, bbox_inches="tight")
            plt.close(fig_prob)
            log.info("Behavior probability saved to %s", subject_dir)

            # ── Habituation curve ──
            fig_hab = plot_habituation_curve(df)
            fig_hab.savefig(subject_dir / "habituation_curve.png", dpi=300, bbox_inches="tight")
            plt.close(fig_hab)
            log.info("Habituation curve saved to %s", subject_dir)

            # ── Diagnostic: V_max distribution ──
            fig_vmax = plot_vmax_distribution(df)
            fig_vmax.savefig(subject_dir / "vmax_distribution_diagnostic.png", dpi=300, bbox_inches="tight")
            plt.close(fig_vmax)
            log.info("V_max distribution diagnostic saved to %s", subject_dir)

            # ── Individual Escape trial export ──
            _generate_individual_trial_figures(
                df_escape, subject_dir / "individual_escapes", "Escape",
            )

            # ── Individual PreWalk trial export (NEW) ──
            _generate_individual_trial_figures(
                df_prewalk, subject_dir / "individual_escapes_prewalk", "PreWalk",
            )

        else:
            plt.show()

    log.info("All subjects processed.")


if __name__ == "__main__":
    main()
