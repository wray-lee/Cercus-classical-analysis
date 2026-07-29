"""
Cercus Framework — CLI Application
===================================
Unified CLI using Typer. Commands: single, population, mcmc, trial-panels, trajectories.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer
from scipy.stats import circmean

app = typer.Typer(
    name="cercus",
    help="Cercus framework — Classical behavioral neuroscience analysis pipeline",
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# CLI Commands
# ══════════════════════════════════════════════════════════════════════


@app.command()
def single(
    input: Path = typer.Option(..., "--input", "-i", help="Directory containing session CSV files"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Directory to save SVG figures"),
    control_type: str = typer.Option("baseline_visual_test", "--control-type", help="Control condition type"),
    stim_type: str = typer.Option("looming_wind", "--stim-type", help="Stimulus condition type"),
) -> None:
    """Single-subject analysis pipeline (equivalent to main.py)."""
    from main import main as run_main
    argv = [f"--input-dir={input}", f"--output={output}"]
    if control_type:
        argv.append(f"--control-type={control_type}")
    if stim_type:
        argv.append(f"--stim-type={stim_type}")
    run_main(argv)


@app.command()
def population(
    input: Path = typer.Option(..., "--input", "-i", help="Directory containing session CSV files"),
    output: Path = typer.Option(..., "--output", "-o", help="Directory to save results"),
) -> None:
    """Population-level analysis (equivalent to population_analysis.py)."""
    from population_analysis import main as run_population
    run_population([f"--input-dir={input}", f"--output={output}"])


@app.command()
def mcmc(
    input: Path = typer.Option(..., "--input", "-i", help="Directory containing session CSV files"),
    output: Path = typer.Option("results", "--output", "-o", help="Directory to save output files"),
    binary_mode: str = typer.Option("escape_only", "--binary-mode", help="Binary mode: escape_only or escape_prewalk"),
    chains: int = typer.Option(8, "--chains", help="Number of MCMC chains"),
    draws: int = typer.Option(2000, "--draws", help="Number of posterior draws"),
    tune: int = typer.Option(3000, "--tune", help="Number of tuning steps"),
) -> None:
    """Run MCMC Bayesian psychophysics analysis."""
    from mcmc_analysis import main as run_mcmc
    argv = [
        f"--input-dir={input}", f"--output={output}",
        f"--binary-mode={binary_mode}",
        f"--n-chains={chains}", f"--n-draws={draws}", f"--n-tune={tune}",
    ]
    run_mcmc(argv)


@app.command("trial-panels")
def trial_panels(
    input: Path = typer.Option(..., "--input", "-i", help="Directory containing session CSV files"),
    output: Path = typer.Option(..., "--output", "-o", help="Directory to save figures"),
) -> None:
    """Generate trial panel figures (equivalent to plot_trial_panels.py)."""
    from plot_trial_panels import main as run_panels
    run_panels([f"--input-dir={input}", f"--output={output}"])


@app.command()
def trajectories(
    input: Path = typer.Option(..., "--input", "-i", help="Directory containing session CSV files"),
    output: Path = typer.Option(..., "--output", "-o", help="Path to save the output figure"),
) -> None:
    """Generate unified trajectory overlays (equivalent to plot_all_trajectories_fixed.py)."""
    from plot_all_trajectories_fixed import main as run_traj
    run_traj([f"--input-dir={input}", f"--output={output}"])


@app.command("compare")
def compare(
    input: Path = typer.Option(..., "--input", "-i", help="Path to CSV or directory with V/M/VM subdirs"),
    output: Path = typer.Option(..., "--output", "-o", help="Directory to save comparison results"),
    unified_preset: str = typer.Option("escape_angular_peak", "--unified-preset", help="dz_integration_range for unified preset"),
) -> None:
    """Cross-modal comparison: V vs M vs VM using unified angular_peak preset."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from cercus.config import config, reload_config
    from cercus.config.settings import TrajectoryConfig

    # ── Force comparison mode in config ──
    raw = config.to_dict()
    raw["comparison"] = {
        "enabled": True,
        "unified_preset": {
            "use_z_degree_to_draw": True,
            "use_escape_onset_only_xy": True,
            "use_rigid_rotation": True,
            "dz_integration_range": unified_preset,
            "use_escape_onset_heading": False,
            "use_angular_velocity_offset": False,
        },
    }
    # Write to a temp config and reload — or just monkeypatch the data dict
    # ponytail: simpler to patch the internal data dict directly
    config._data["comparison"] = raw["comparison"]
    reload_config()

    # ── Load data ──
    input_path = Path(input)
    if input_path.is_dir():
        # Check for V/M/VM subdirectories
        subdirs = [d for d in input_path.iterdir() if d.is_dir()]
        mod_subdirs = [d for d in subdirs if d.name.upper() in ("V", "M", "VM")]
        if mod_subdirs:
            parts = []
            for d in mod_subdirs:
                csv_files = list(d.glob("**/*.csv"))
                if not csv_files:
                    continue
                mod_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
                mod_df["modality"] = d.name.upper()
                parts.append(mod_df)
            if not parts:
                log.error("No CSV files found in V/M/VM subdirectories.")
                raise typer.Exit(1)
            df = pd.concat(parts, ignore_index=True)
        else:
            # Single directory with CSV files
            csv_files = list(input_path.glob("**/*.csv"))
            if not csv_files:
                log.error("No CSV files found in %s", input_path)
                raise typer.Exit(1)
            df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
            if "modality" not in df.columns:
                log.error("Input CSV must have a 'modality' column when not using V/M/VM subdirs.")
                raise typer.Exit(1)
    else:
        df = pd.read_csv(input_path)
        if "modality" not in df.columns:
            log.error("Single CSV input must have a 'modality' column.")
            raise typer.Exit(1)

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Comparison data: %d trials, modalities=%s",
             len(df), sorted(df["modality"].unique()))

    # ── Polar comparison ──
    from cercus.visualization.polar import plot_population_polar_histogram
    from cercus.visualization.heatmaps import plot_trial_stacked_heatmap

    fig_polar = plot_population_polar_histogram(df, comparison_mode=True)
    polar_path = output_dir / "polar_comparison_angular_peak.svg"
    fig_polar.savefig(polar_path, dpi=300, bbox_inches="tight")
    plt.close(fig_polar)
    log.info("Saved polar comparison: %s", polar_path)

    # ── Heatmap comparison ──
    fig_heatmap = plot_trial_stacked_heatmap(df, comparison_mode=True)
    fig_heatmap.savefig(output_dir / "heatmap_comparison_ttc.svg", dpi=300, bbox_inches="tight")
    plt.close(fig_heatmap)
    log.info("Saved heatmap comparison TTC")

    # Also onset-aligned
    fig_onset = plot_trial_stacked_heatmap(df, align="onset", t_window=(-0.5, 1.51), comparison_mode=True)
    fig_onset.savefig(output_dir / "heatmap_comparison_onset.svg", dpi=300, bbox_inches="tight")
    plt.close(fig_onset)
    log.info("Saved heatmap comparison onset")

    # ── Summary DataFrame ──
    from pipeline.constants import _get_unified_side, ESCAPE_START_THRESHOLD
    from cercus.visualization._circstats import rayleigh_p, wallraff_test, watson_williams_test
    from cercus.core.kinematics.trajectory_integration import body_to_traj, build_angular_peak_dz_mask

    group_cols = ["subject_id", "global_trial_id"] if "subject_id" in df.columns else ["global_trial_id"]
    mod_col = "modality"

    summary_rows = []
    for mod in sorted(df[mod_col].unique()):
        sub = df[df[mod_col] == mod]
        n_total = sub.groupby(group_cols).ngroups
        escape_sub = sub[sub["response_type"] == "Escape"] if "response_type" in sub.columns else sub
        n_escape = escape_sub.groupby(group_cols).ngroups if "response_type" in sub.columns else n_total
        p_escape = n_escape / n_total if n_total > 0 else 0.0

        # Mean latency
        latencies = []
        for _, grp in sub.groupby(group_cols):
            grp = grp.sort_values("t_rel")
            above = grp["speed"] > ESCAPE_START_THRESHOLD
            if above.any():
                lat = grp[above]["t_rel"].iloc[0] - grp["interval_onset_ms"].iloc[0] if pd.notna(grp["interval_onset_ms"].iloc[0]) else np.nan
                latencies.append(lat)
        mean_lat = np.nanmean(latencies) if latencies else np.nan

        # Escape angles
        angles = []
        for _, grp in escape_sub.groupby(group_cols):
            grp = grp.sort_values("t_rel")
            _onset = grp["interval_onset_ms"].iloc[0] if "interval_onset_ms" in grp.columns else np.nan
            _offset = grp["interval_offset_ms"].iloc[0] if "interval_offset_ms" in grp.columns else np.nan
            t_vals = grp["t_rel"].values
            if pd.notna(_onset) and pd.notna(_offset):
                oi = int(np.argmin(np.abs(t_vals - _onset)))
                oi2 = int(np.argmin(np.abs(t_vals - _offset)))
                if oi >= oi2:
                    oi2 = min(oi + 1, len(t_vals) - 1)
                av = grp["angular_velocity"].values if "angular_velocity" in grp.columns else None
                if av is not None:
                    mz = build_angular_peak_dz_mask(av, oi, oi2, len(t_vals))
                else:
                    mz = np.zeros(len(t_vals), dtype=bool)
                    mz[oi:oi2] = True
                mxy = np.zeros(len(t_vals), dtype=bool)
                mxy[oi:oi2] = True
            else:
                continue
            result = body_to_traj(grp, mxy, use_z=True, use_rigid_rotation=True, mask_z=mz)
            if result is None:
                continue
            tx, ty = result
            if tx is None or len(tx) < 2:
                continue
            ss = _get_unified_side(grp)
            if ss == "left":
                tx = -tx
            angles.append(float(np.arctan2(tx[-1], ty[-1])))

        angles = np.asarray(angles)
        mu = float(np.degrees(circmean(angles, high=np.pi, low=-np.pi))) if len(angles) > 0 else np.nan
        R = float(np.abs(np.exp(1j * angles).mean())) if len(angles) > 0 else np.nan
        p_ray = rayleigh_p(R, len(angles)) if len(angles) > 0 else np.nan

        summary_rows.append({
            "modality": mod, "n": n_total, "P_escape": p_escape,
            "mean_latency_ms": mean_lat, "mu_deg": mu, "R": R, "p_rayleigh": p_ray,
        })

    summary_df = pd.DataFrame(summary_rows)

    # MSI
    if all(m in summary_df["modality"].values for m in ["V", "M", "VM"]):
        RV = summary_df.loc[summary_df["modality"] == "V", "R"].values[0]
        RM = summary_df.loc[summary_df["modality"] == "M", "R"].values[0]
        RVM = summary_df.loc[summary_df["modality"] == "VM", "R"].values[0]
        summary_df.loc[summary_df["modality"] == "VM", "MSI_R"] = (RVM - max(RV, RM)) / max(RV, RM) * 100

        PV = summary_df.loc[summary_df["modality"] == "V", "P_escape"].values[0]
        PM = summary_df.loc[summary_df["modality"] == "M", "P_escape"].values[0]
        PVM = summary_df.loc[summary_df["modality"] == "VM", "P_escape"].values[0]
        summary_df.loc[summary_df["modality"] == "VM", "MSI_prob"] = (PVM - max(PV, PM)) / max(PV, PM) * 100

    csv_path = output_dir / "multisensory_summary.csv"
    summary_df.to_csv(csv_path, index=False, float_format="%.4f")
    log.info("Saved summary: %s", csv_path)
    log.info("Comparison complete — figures in: %s", output_dir)


if __name__ == "__main__":
    app()
