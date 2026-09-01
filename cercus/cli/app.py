"""
Cercus Framework — CLI Application
===================================
Unified CLI using Typer. Commands: single, population, mcmc, trial-panels, trajectories.
"""

from __future__ import annotations

import logging
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
    workers: Optional[int] = typer.Option(None, "--workers", help="Number of parallel workers (default: all CPUs)"),
) -> None:
    """Single-subject analysis pipeline (equivalent to main.py)."""
    from main import main as run_main
    argv = [f"--input-dir={input}", f"--output={output}"]
    if control_type:
        argv.append(f"--control-type={control_type}")
    if stim_type:
        argv.append(f"--stim-type={stim_type}")
    if workers is not None:
        argv.append(f"--workers={workers}")
    run_main(argv)


@app.command()
def population(
    input: Path = typer.Option(..., "--input", "-i", help="Directory containing session CSV files"),
    output: Path = typer.Option(..., "--output", "-o", help="Directory to save results"),
    workers: Optional[int] = typer.Option(None, "--workers", help="Number of parallel workers (default: all CPUs)"),
) -> None:
    """Population-level analysis (equivalent to population_analysis.py)."""
    from population_analysis import main as run_population
    argv = [f"--input-dir={input}", f"--output={output}"]
    if workers is not None:
        argv.append(f"--workers={workers}")
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
    workers: Optional[int] = typer.Option(None, "--workers", help="Number of parallel workers (default: all CPUs)"),
) -> None:
    """Generate trial panel figures (equivalent to plot_trial_panels.py)."""
    from plot_trial_panels import main as run_panels
    argv = [f"--input-dir={input}", f"--output={output}"]
    if workers is not None:
        argv.append(f"--workers={workers}")
    run_panels(argv)


@app.command()
def trajectories(
    input: Path = typer.Option(..., "--input", "-i", help="Directory containing session CSV files"),
    output: Path = typer.Option(..., "--output", "-o", help="Path to save the output figure"),
) -> None:
    """Generate unified trajectory overlays (equivalent to plot_all_trajectories_fixed.py)."""
    from plot_all_trajectories_fixed import main as run_traj
    run_traj([f"--input-dir={input}", f"--output={output}"])


@app.command("multisensory-traj")
def multisensory_traj(
    bv: Path = typer.Option(..., "--bv", help="Directory for baseline-visual data"),
    bw: Path = typer.Option(..., "--bw", help="Directory for baseline-wind data"),
    ms: Path = typer.Option(..., "--ms", help="Directory for multisensory data"),
    output: Path = typer.Option(..., "--output", "-o", help="Path to save the output figure"),
) -> None:
    """Multisensory trajectory comparison: baselines (−x) vs multisensory (+x)."""
    from plot_multisensory_trajectories import main as run_ms
    run_ms([f"--bv={bv}", f"--bw={bw}", f"--ms={ms}", f"--output={output}"])


@app.command()
def calibrate(
    input: Path = typer.Option(..., "--input", "-i", help="Directory containing session CSV folders"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Directory to save report and corrected CSVs (optional)"),
    estimate_only: bool = typer.Option(False, "--estimate-only", help="Only estimate delta without writing corrected CSVs"),
    plot: bool = typer.Option(True, "--plot/--no-plot", help="Generate calibration report plot if output is set"),
) -> None:
    """Calibrate ring-airflow stimulus angle offset (equivalent to tools/calibrate_offset.py)."""
    from tools.calibrate_offset import main as run_cali
    argv = [f"--input={input}"]
    if output:
        argv.append(f"--output={output}")
    if estimate_only:
        argv.append("--estimate-only")
    if not plot:
        argv.append("--no-plot")
    run_cali(argv)


if __name__ == "__main__":
    app()
