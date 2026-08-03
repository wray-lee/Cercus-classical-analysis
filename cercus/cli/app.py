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

import typer

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
    individual_checks: bool = typer.Option(
        False, "--individual-checks",
        help="Also run per-animal robustness checks (second-order Rayleigh, "
             "leave-one-out, bootstrap) and save suppl figures + individual_summary.csv",
    ),
) -> None:
    """Population-level analysis (equivalent to population_analysis.py)."""
    from population_analysis import main as run_population
    argv = [f"--input-dir={input}", f"--output={output}"]
    if individual_checks:
        argv.append("--individual-checks")
    run_population(argv)


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


if __name__ == "__main__":
    app()
