#!/usr/bin/env python3
"""
MCMC Bayesian Psychophysics Analysis for Multisensory Integration
=================================================================

This script analyzes cricket escape behavior data from the Cercus experimental framework.
It uses PyMC to fit psychometric functions and perform Bayesian inference on multisensory
integration (visual looming + wind stimulus).

Key analyses:
1. Psychometric function fitting: P(Escape) = sigmoid(k * (TTC - TTC50))
2. Posterior distribution estimation for PSE (TTC50) and slope (k)
3. Hypothesis testing: Compare TTC50 across stimulus conditions
4. Bayesian optimal integration: Variance reduction analysis

Author: Claude Code Multi-Agent Workflow
Date: 2026-06-30
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

# Threshold for escape detection (mm/s cumulative displacement)
DEFAULT_ESCAPE_THRESHOLD_MM = 5.0

# Time window for escape detection (seconds after stimulus onset)
ESCAPE_WINDOW_SEC = 2.0

# MCMC sampling parameters
N_CHAINS = 4
N_DRAWS = 2000
N_TUNE = 3000  # Increased from 1000 for better adaptation
TARGET_ACCEPT = 0.95  # Added for sigmoid model stability
RANDOM_SEED = 42

# Plotting style
plt.style.use("seaborn-v0_8-whitegrid")


def generate_color_palette(condition_names: List[str]) -> Dict[str, str]:
    """Generate color palette dynamically from condition names.

    Uses seaborn's husl palette for consistent, distinguishable colors.
    """
    n = len(condition_names)
    palette = sns.color_palette("husl", n)
    # Convert RGB tuples to hex strings
    return {cond: "#{:02x}{:02x}{:02x}".format(
        int(palette[i][0] * 255),
        int(palette[i][1] * 255),
        int(palette[i][2] * 255)
    ) for i, cond in enumerate(condition_names)}


# ============================================================================
# Data Loading and Preprocessing
# ============================================================================


def load_events(events_path: Path) -> pd.DataFrame:
    """Load and parse events CSV file.

    Extracts trial-level information including:
    - trial_type: baseline_visual, baseline_wind, looming_wind
    - target_ttc_ms: temporal offset between visual and wind stimulus
    - delay_sec: wind delay for pure wind trials
    - side: stimulus presentation side (L/R)
    """
    if not events_path.exists():
        raise FileNotFoundError(f"Events file not found: {events_path}")

    logger.info(f"Loading events from {events_path}")
    df = pd.read_csv(events_path, encoding="utf-8-sig")

    # Validate required columns
    required_cols = {"event_name", "details"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Parse JSON details column with logging for errors
    def _parse_details(x):
        if pd.isna(x) or x == "":
            return {}
        try:
            return json.loads(x)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to parse details JSON: {x!r} - {e}")
            return {}

    details_df = df["details"].apply(_parse_details).apply(pd.Series)
    df = pd.concat([df, details_df], axis=1)

    # Filter for trial_start events to get trial-level data
    trials = df[df["event_name"] == "trial_start"].copy()

    # Extract trial information using vectorized operations
    trials["trial_type"] = trials["type"].fillna("unknown") if "type" in trials.columns else "unknown"
    if "target_ttc_ms" in trials.columns:
        trials["target_ttc_ms"] = pd.to_numeric(trials["target_ttc_ms"], errors="coerce")
    else:
        trials["target_ttc_ms"] = np.nan

    if "delay_sec" in trials.columns:
        trials["delay_sec"] = pd.to_numeric(trials["delay_sec"], errors="coerce")
    else:
        trials["delay_sec"] = np.nan

    trials["side"] = trials["side"] if "side" in trials.columns else "unknown"

    # Create stimulus condition labels using vectorized operations
    trials["stim_condition"] = _classify_conditions_vectorized(trials)

    logger.info(f"Loaded {len(trials)} trials")
    logger.info(f"Stimulus conditions: {trials['stim_condition'].value_counts().to_dict()}")

    return trials


def load_multiple_sessions(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and combine multiple session files from a directory.

    The Cercus framework stores each session in separate files:
    - {subject}_session_{n}_events.csv
    - {subject}_session_{n}_kinematics.csv

    This function finds and combines all matching files.

    Returns
    -------
    tuple
        (trials_df, kinematics_df) combined from all sessions
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Find all event files
    event_files = sorted(data_dir.glob("*_events.csv"))
    if not event_files:
        raise FileNotFoundError(f"No event files found in {data_dir}")

    logger.info(f"Found {len(event_files)} event files in {data_dir}")

    all_trials = []
    all_kinematics = []

    for event_file in event_files:
        # Find corresponding kinematics file
        kin_file = event_file.with_name(
            event_file.name.replace("_events.csv", "_kinematics.csv")
        )

        # Load events
        try:
            trials = load_events(event_file)
            trials["source_file"] = event_file.name
            all_trials.append(trials)
        except Exception as e:
            logger.warning(f"Failed to load {event_file}: {e}")
            continue

        # Load kinematics if exists
        if kin_file.exists():
            try:
                kin = pd.read_csv(kin_file, encoding="utf-8-sig")
                kin["source_file"] = event_file.name
                all_kinematics.append(kin)
            except Exception as e:
                logger.warning(f"Failed to load {kin_file}: {e}")

    if not all_trials:
        raise ValueError("No valid trial data found")

    # Combine all data
    trials_df = pd.concat(all_trials, ignore_index=True)
    kinematics_df = pd.concat(all_kinematics, ignore_index=True) if all_kinematics else pd.DataFrame()

    logger.info(f"Combined {len(trials_df)} trials from {len(all_trials)} sessions")
    logger.info(f"Combined {len(kinematics_df)} kinematics samples")

    return trials_df, kinematics_df


def _classify_conditions_vectorized(trials: pd.DataFrame) -> pd.Series:
    """Classify trials into stimulus conditions using vectorized operations.

    Handles multiple trial types:
    - baseline_visual / wind_only → visual_only
    - baseline_wind / wind_only → wind_only
    - looming_wind → looming_wind_{TTC}
    """
    conditions = pd.Series("unknown", index=trials.index)

    # Visual only (from LoomingParadigm or SingleLoomingParadigm)
    mask_vis = trials["trial_type"] == "baseline_visual"
    conditions[mask_vis] = "visual_only"

    # Wind only (from WindParadigm or LoomingParadigm baseline)
    mask_wind = trials["trial_type"].isin(["baseline_wind", "wind_only"])
    conditions[mask_wind] = "wind_only"

    # Looming + wind - use round() explicitly to handle fractional TTC values
    mask_lw = trials["trial_type"] == "looming_wind"
    has_ttc = mask_lw & trials["target_ttc_ms"].notna()
    conditions[has_ttc] = "looming_wind_" + trials.loc[has_ttc, "target_ttc_ms"].round().astype(int).astype(str)
    no_ttc = mask_lw & trials["target_ttc_ms"].isna()
    conditions[no_ttc] = "looming_wind_unknown"

    return conditions


def load_kinematics(kinematics_path: Path) -> pd.DataFrame:
    """Load kinematics CSV file with column validation."""
    if not kinematics_path.exists():
        raise FileNotFoundError(f"Kinematics file not found: {kinematics_path}")

    logger.info(f"Loading kinematics from {kinematics_path}")
    df = pd.read_csv(kinematics_path)

    # Validate required columns
    required_cols = {"global_trial_id", "dx", "dy", "dz", "sys_time"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required kinematics columns: {missing_cols}")

    logger.info(f"Loaded {len(df)} kinematics samples")
    return df


def detect_escape_responses(
    trials: pd.DataFrame,
    kinematics: pd.DataFrame,
    threshold_mm: float = DEFAULT_ESCAPE_THRESHOLD_MM,
) -> pd.DataFrame:
    """Detect escape responses from kinematics data.

    Escape is defined as cumulative displacement exceeding threshold_mm
    within ESCAPE_WINDOW_SEC seconds after stimulus onset.

    Uses vectorized groupby operations for efficiency.
    Does not mutate the input kinematics DataFrame.
    """
    logger.info("Detecting escape responses...")

    # Work on a copy to avoid mutating the caller's DataFrame
    kin = kinematics.copy()

    # Compute displacement magnitude per sample
    kin["displacement"] = np.sqrt(
        kin["dx"] ** 2 + kin["dy"] ** 2 + kin["dz"] ** 2
    )

    # Pre-compute per-trial kinematics summaries using groupby
    kin_grouped = kin.groupby("global_trial_id")

    escape_decisions = {}
    for trial_id, group in kin_grouped:
        if len(group) == 0:
            escape_decisions[trial_id] = np.nan
            continue

        # Get stimulus onset time and compute cumulative displacement in window
        t_stim = group["sys_time"].iloc[0]
        t_end = t_stim + ESCAPE_WINDOW_SEC

        window_mask = group["sys_time"] <= t_end
        if window_mask.any():
            cum_disp = np.cumsum(np.abs(group.loc[window_mask, "displacement"].values))
            max_disp = cum_disp.max()
            escape_decisions[trial_id] = 1 if max_disp > threshold_mm else 0
        else:
            escape_decisions[trial_id] = 0

    # Map decisions back to trials
    trials["escape_decision"] = trials["global_trial_id"].map(escape_decisions)

    # Log statistics
    valid_trials = trials.dropna(subset=["escape_decision"])
    n_escape = valid_trials["escape_decision"].sum()
    logger.info(f"Escape responses detected: {n_escape}/{len(valid_trials)}")

    for condition in sorted(trials["stim_condition"].unique()):
        cond_trials = valid_trials[valid_trials["stim_condition"] == condition]
        if len(cond_trials) > 0:
            escape_rate = cond_trials["escape_decision"].mean()
            logger.info(f"  {condition}: {escape_rate:.1%} ({len(cond_trials)} trials)")

    return trials


def compute_effective_ttc(trials: pd.DataFrame) -> pd.DataFrame:
    """Compute effective TTC for each trial.

    For looming_wind trials, TTC is the target_ttc_ms value.
    For baseline_visual, TTC is excluded from MCMC (no temporal dimension).
    For baseline_wind, TTC is set based on delay.
    """
    logger.info("Computing effective TTC values...")

    ttc = pd.Series(np.nan, index=trials.index)

    # Wind only: convert delay to equivalent TTC
    mask_wind = trials["stim_condition"] == "wind_only"
    has_delay = mask_wind & trials["delay_sec"].notna()
    ttc[has_delay] = -trials.loc[has_delay, "delay_sec"] * 1000

    # Looming + wind: use target TTC directly
    mask_lw = trials["stim_condition"].str.startswith("looming_wind_")
    has_ttc = mask_lw & trials["target_ttc_ms"].notna()
    ttc[has_ttc] = trials.loc[has_ttc, "target_ttc_ms"]

    # Visual only: keep as NaN (will be excluded from MCMC)
    # Note: visual_only has no temporal dimension in this paradigm

    trials["effective_ttc_ms"] = ttc

    valid_ttc = trials["effective_ttc_ms"].dropna()
    if len(valid_ttc) > 0:
        logger.info(f"TTC range: {valid_ttc.min():.0f} to {valid_ttc.max():.0f} ms")
    else:
        logger.warning("No valid TTC values found")

    return trials


# ============================================================================
# MCMC Model
# ============================================================================


def build_psychometric_model(
    ttc: np.ndarray,
    escape: np.ndarray,
    condition_idx: np.ndarray,
    condition_names: List[str],
) -> pm.Model:
    """Build Bayesian psychometric model.

    Model specification:
    - P(Escape_i = 1) = sigmoid(k_c * (TTC_i - TTC50_c))
    - TTC50_c ~ Normal(0, 500)  # Wide prior covering full TTC range
    - k_c ~ LogNormal(-5, 1)    # Positive slope, well-behaved log-scale geometry

    Parameters
    ----------
    ttc : array-like
        TTC values (ms) for each trial
    escape : array-like
        Binary escape decisions (0/1)
    condition_idx : array-like
        Integer index mapping each trial to a condition
    condition_names : list
        Names of conditions for coordinate mapping
    """
    n_conditions = len(condition_names)
    logger.info(f"Building psychometric model with {n_conditions} conditions...")

    # Compute reasonable initial values from data
    init_ttc50 = np.zeros(n_conditions)
    for i in range(n_conditions):
        mask = condition_idx == i
        if mask.any():
            init_ttc50[i] = np.median(ttc[mask])

    with pm.Model(coords={"condition": condition_names}) as model:
        # Priors for PSE (TTC50) - wide prior covering observed range
        ttc50 = pm.Normal(
            "ttc50",
            mu=0,
            sigma=500,
            shape=n_conditions,
            dims="condition",
            initval=init_ttc50,
        )

        # Prior for slope (k) - LogNormal for positive values with log-scale geometry
        # mu=-5, sigma=1 gives median ~0.007, 95% CI ~[0.001, 0.05]
        k = pm.LogNormal(
            "k",
            mu=-5,
            sigma=1,
            shape=n_conditions,
            dims="condition",
            initval=np.full(n_conditions, 0.01),
        )

        # Linear predictor
        mu = k[condition_idx] * (ttc - ttc50[condition_idx])

        # Sigmoid likelihood (Bernoulli with logit link)
        p_escape = pm.Deterministic("p_escape", pm.math.sigmoid(mu))

        # Likelihood
        escape_obs = pm.Bernoulli(
            "escape_obs",
            p=p_escape,
            observed=escape,
        )

    return model


def run_mcmc_inference(
    model: pm.Model, random_seed: int = RANDOM_SEED
) -> az.InferenceData:
    """Run MCMC sampling using NUTS sampler.

    Returns ArviZ InferenceData object with posterior samples.
    """
    logger.info(
        f"Running MCMC with {N_CHAINS} chains, {N_DRAWS} draws, "
        f"{N_TUNE} tune steps, target_accept={TARGET_ACCEPT}..."
    )

    with model:
        trace = pm.sample(
            draws=N_DRAWS,
            chains=N_CHAINS,
            tune=N_TUNE,
            target_accept=TARGET_ACCEPT,
            random_seed=random_seed,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": False},
        )

    logger.info("MCMC sampling complete")
    return trace


def check_convergence(trace: az.InferenceData) -> bool:
    """Check MCMC convergence using R-hat and effective sample size.

    Returns True if convergence criteria are met.
    """
    logger.info("Checking convergence diagnostics...")

    summary = az.summary(trace, var_names=["ttc50", "k"])
    logger.info(f"\n{summary.to_string()}")

    # Check R-hat
    rhat_ok = (summary["r_hat"] < 1.01).all()
    if not rhat_ok:
        logger.warning("R-hat > 1.01 detected - chains may not have converged")
    else:
        logger.info("R-hat OK: all parameters < 1.01")

    # Check effective sample size
    ess_ok = (summary["ess_bulk"] > 400).all()
    if not ess_ok:
        logger.warning("ESS_bulk < 400 detected - may need more samples")
    else:
        logger.info("ESS OK: all parameters > 400")

    return rhat_ok and ess_ok


# ============================================================================
# Statistical Analysis
# ============================================================================


def compute_hdi(
    samples: np.ndarray, credible_mass: float = 0.95
) -> Tuple[float, float]:
    """Compute Highest Density Interval (HDI).

    Parameters
    ----------
    samples : array-like
        Posterior samples
    credible_mass : float
        Credible mass (default 0.95 for 95% HDI)

    Returns
    -------
    tuple
        (lower, upper) bounds of HDI
    """
    sorted_samples = np.sort(samples)
    n = len(sorted_samples)
    interval_size = int(np.ceil(credible_mass * n))

    if interval_size >= n:
        return sorted_samples[0], sorted_samples[-1]

    # Find shortest interval
    widths = sorted_samples[interval_size:] - sorted_samples[:n - interval_size]
    min_idx = np.argmin(widths)

    return float(sorted_samples[min_idx]), float(sorted_samples[min_idx + interval_size])


def analyze_ttc50_differences(
    trace: az.InferenceData, condition_names: List[str]
) -> Dict:
    """Analyze posterior differences in TTC50 between conditions.

    This tests the hypothesis that multisensory integration shifts the
    subjective equality point (PSE).
    """
    logger.info("Analyzing TTC50 differences between conditions...")

    ttc50_posterior = trace.posterior["ttc50"].values
    # Shape: (chains, draws, conditions)
    n_chains, n_draws, n_conditions = ttc50_posterior.shape

    results = {}

    # Compare visual_only vs multimodal conditions
    if "visual_only" in condition_names:
        vis_idx = condition_names.index("visual_only")

        for i, cond in enumerate(condition_names):
            if cond.startswith("looming_wind_"):
                # Compute posterior difference
                diff = ttc50_posterior[:, :, vis_idx] - ttc50_posterior[:, :, i]
                diff_flat = diff.flatten()

                hdi_low, hdi_high = compute_hdi(diff_flat)
                prob_positive = float((diff_flat > 0).mean())

                results[f"visual_only_vs_{cond}"] = {
                    "mean_diff": float(diff_flat.mean()),
                    "hdi_95": [hdi_low, hdi_high],
                    "prob_diff_positive": prob_positive,
                }

                logger.info(f"  {cond} vs visual_only:")
                logger.info(f"    Mean diff: {diff_flat.mean():.1f} ms")
                logger.info(f"    95% HDI: [{hdi_low:.1f}, {hdi_high:.1f}] ms")
                logger.info(f"    P(diff > 0): {prob_positive:.3f}")

    return results


def compute_variance_reduction(
    trace: az.InferenceData, condition_names: List[str]
) -> Dict:
    """Compute variance reduction for Bayesian optimal integration.

    Compares posterior variance of multimodal conditions to unimodal baselines.
    A reduction in variance indicates reliability-weighted integration.
    A negative value indicates increased variance (possible model violation).
    """
    logger.info("Computing variance reduction for optimal integration...")

    ttc50_posterior = trace.posterior["ttc50"].values
    results = {}

    # Get unimodal variances
    vis_idx = condition_names.index("visual_only") if "visual_only" in condition_names else None
    wind_idx = condition_names.index("wind_only") if "wind_only" in condition_names else None

    if vis_idx is not None:
        vis_var = float(ttc50_posterior[:, :, vis_idx].var())
        results["visual_variance"] = vis_var
    else:
        vis_var = None

    if wind_idx is not None:
        wind_var = float(ttc50_posterior[:, :, wind_idx].var())
        results["wind_variance"] = wind_var
    else:
        wind_var = None

    # Compute multimodal variances and reduction
    for i, cond in enumerate(condition_names):
        if cond.startswith("looming_wind_"):
            mm_var = float(ttc50_posterior[:, :, i].var())
            results[f"{cond}_variance"] = mm_var

            if vis_var is not None and vis_var > 0:
                reduction = 1 - (mm_var / vis_var)
                results[f"{cond}_variance_reduction"] = float(reduction)
                if reduction < 0:
                    logger.info(f"  {cond}: variance INCREASED by {-reduction:.1%} (possible model violation)")
                else:
                    logger.info(f"  {cond}: variance reduced by {reduction:.1%}")

    return results


# ============================================================================
# Visualization
# ============================================================================


def plot_posterior_traces(
    trace: az.InferenceData, condition_names: List[str], output_path: Path
):
    """Generate posterior trace plots for MCMC diagnostics.

    Creates trace plots for TTC50 and k parameters across all conditions.
    """
    logger.info("Generating posterior trace plots...")

    n_conditions = len(condition_names)

    fig, axes = plt.subplots(n_conditions, 2, figsize=(14, 3 * n_conditions))
    if n_conditions == 1:
        axes = axes.reshape(1, -1)

    for i, cond in enumerate(condition_names):
        # TTC50 trace - use condition name, not integer index
        ax_ttc = axes[i, 0]
        for chain in range(trace.posterior.sizes["chain"]):
            samples = trace.posterior["ttc50"].sel(chain=chain, condition=cond).values
            ax_ttc.plot(samples, alpha=0.7, linewidth=0.5, label=f"Chain {chain}")
        ax_ttc.set_ylabel(f"{cond}\nTTC50 (ms)")
        ax_ttc.set_xlabel("Sample")
        if i == 0:
            ax_ttc.set_title("TTC50 Traces")
        ax_ttc.legend(fontsize=8)

        # k trace - use condition name, not integer index
        ax_k = axes[i, 1]
        for chain in range(trace.posterior.sizes["chain"]):
            samples = trace.posterior["k"].sel(chain=chain, condition=cond).values
            ax_k.plot(samples, alpha=0.7, linewidth=0.5, label=f"Chain {chain}")
        ax_k.set_ylabel("k (slope)")
        ax_k.set_xlabel("Sample")
        if i == 0:
            ax_k.set_title("Slope (k) Traces")
        ax_k.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Trace plots saved to {output_path}")


def plot_psychometric_curves(
    trace: az.InferenceData,
    trials: pd.DataFrame,
    condition_names: List[str],
    output_path: Path,
):
    """Generate psychometric curves with 95% HDI shading.

    Plots fitted sigmoid curves with uncertainty bands overlaid on data points.
    Uses proper HDI computation (not percentile approximation).
    """
    logger.info("Generating psychometric curves...")

    # Generate dynamic color palette
    colors = generate_color_palette(condition_names)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Generate TTC grid for smooth curves
    ttc_range = trials["effective_ttc_ms"].dropna()
    if len(ttc_range) == 0:
        logger.warning("No valid TTC data for plotting")
        plt.close(fig)
        return

    ttc_grid = np.linspace(ttc_range.min() - 50, ttc_range.max() + 50, 200)

    for i, cond in enumerate(condition_names):
        # Skip conditions without TTC data
        cond_trials = trials[trials["stim_condition"] == cond].dropna(subset=["effective_ttc_ms", "escape_decision"])
        if len(cond_trials) == 0:
            continue

        color = colors.get(cond, f"C{i}")

        # Get posterior samples for this condition - use condition name
        ttc50_samples = trace.posterior["ttc50"].sel(condition=cond).values.flatten()
        k_samples = trace.posterior["k"].sel(condition=cond).values.flatten()

        # Compute predicted probabilities for each posterior sample
        n_curves = min(500, len(ttc50_samples))
        idx = np.random.choice(len(ttc50_samples), n_curves, replace=False)

        predictions = np.zeros((n_curves, len(ttc_grid)))
        for j, (ttc50, k) in enumerate(zip(ttc50_samples[idx], k_samples[idx])):
            predictions[j] = 1 / (1 + np.exp(-k * (ttc_grid - ttc50)))

        # Compute median and proper HDI
        median_pred = np.median(predictions, axis=0)
        hdi_low = np.zeros(len(ttc_grid))
        hdi_high = np.zeros(len(ttc_grid))
        for col_idx in range(len(ttc_grid)):
            hdi_low[col_idx], hdi_high[col_idx] = compute_hdi(predictions[:, col_idx])

        # Plot fitted curve with HDI shading
        ax.plot(ttc_grid, median_pred, color=color, linewidth=2, label=cond)
        ax.fill_between(ttc_grid, hdi_low, hdi_high, color=color, alpha=0.2)

        # Plot data points (binned for visualization)
        n_bins = max(3, min(10, len(cond_trials) // 5))
        try:
            bins = pd.qcut(cond_trials["effective_ttc_ms"], n_bins, duplicates="drop")
            binned = cond_trials.groupby(bins, observed=True)["escape_decision"].agg(["mean", "count", "std"])
            binned_ttc = cond_trials.groupby(bins, observed=True)["effective_ttc_ms"].mean()

            # Plot with error bars
            sem = binned["std"] / np.sqrt(binned["count"])
            ax.errorbar(
                binned_ttc.values,
                binned["mean"].values,
                yerr=sem.values,
                fmt="o",
                color=color,
                markersize=8,
                capsize=5,
                linewidth=2,
            )
        except (ValueError, KeyError) as e:
            logger.warning(f"Could not bin data for {cond}: {e}")
            # Fallback: plot raw data
            ax.scatter(
                cond_trials["effective_ttc_ms"],
                cond_trials["escape_decision"],
                color=color,
                alpha=0.3,
                s=20,
            )

    # Add reference lines
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="P=0.5")
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5, label="TTC=0 ms")

    ax.set_xlabel("Time-to-Collision (ms)\n(negative = wind before collision)", fontsize=14)
    ax.set_ylabel("P(Escape)", fontsize=14)
    ax.set_title("Psychometric Functions: Multisensory Integration\n(95% HDI shaded)", fontsize=16)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", fontsize=10)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Psychometric curves saved to {output_path}")


def plot_posterior_distributions(
    trace: az.InferenceData, condition_names: List[str], output_path: Path
):
    """Plot posterior distributions of TTC50 and k parameters."""
    logger.info("Generating posterior distribution plots...")

    # Generate dynamic color palette
    colors = generate_color_palette(condition_names)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # TTC50 posteriors
    ax_ttc = axes[0]
    for i, cond in enumerate(condition_names):
        samples = trace.posterior["ttc50"].sel(condition=cond).values.flatten()
        ax_ttc.hist(samples, bins=50, alpha=0.5, density=True, label=cond,
                    color=colors.get(cond, f"C{i}"))

        # Add HDI
        hdi_low, hdi_high = compute_hdi(samples)
        ax_ttc.axvline(hdi_low, color=colors.get(cond, f"C{i}"), linestyle="--", alpha=0.5)
        ax_ttc.axvline(hdi_high, color=colors.get(cond, f"C{i}"), linestyle="--", alpha=0.5)

    ax_ttc.set_xlabel("TTC50 (ms)", fontsize=14)
    ax_ttc.set_ylabel("Density", fontsize=14)
    ax_ttc.set_title("Posterior Distributions: TTC50 (PSE)", fontsize=16)
    ax_ttc.legend(fontsize=10)
    ax_ttc.tick_params(labelsize=12)

    # k posteriors
    ax_k = axes[1]
    for i, cond in enumerate(condition_names):
        samples = trace.posterior["k"].sel(condition=cond).values.flatten()
        ax_k.hist(samples, bins=50, alpha=0.5, density=True, label=cond,
                  color=colors.get(cond, f"C{i}"))

    ax_k.set_xlabel("k (slope)", fontsize=14)
    ax_k.set_ylabel("Density", fontsize=14)
    ax_k.set_title("Posterior Distributions: Slope (k)", fontsize=16)
    ax_k.legend(fontsize=10)
    ax_k.tick_params(labelsize=12)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Posterior distributions saved to {output_path}")


# ============================================================================
# Summary and Reporting
# ============================================================================


def generate_summary(
    trace: az.InferenceData,
    condition_names: List[str],
    ttc50_differences: Dict,
    variance_reduction: Dict,
    output_path: Path,
    converged: bool = True,
):
    """Generate JSON summary of MCMC analysis results.

    Parameters
    ----------
    converged : bool
        Whether MCMC convergence diagnostics passed.
    """
    logger.info("Generating analysis summary...")

    summary = {
        "model_info": {
            "n_chains": N_CHAINS,
            "n_draws": N_DRAWS,
            "n_tune": N_TUNE,
            "target_accept": TARGET_ACCEPT,
            "n_conditions": len(condition_names),
            "conditions": condition_names,
        },
        "convergence": {
            "rhat_ok": bool(converged),
            "ess_ok": bool(converged),
        },
        "posterior_estimates": {},
        "hypothesis_tests": ttc50_differences,
        "variance_reduction": variance_reduction,
    }

    # Extract posterior summaries
    for i, cond in enumerate(condition_names):
        ttc50_samples = trace.posterior["ttc50"].sel(condition=cond).values.flatten()
        k_samples = trace.posterior["k"].sel(condition=cond).values.flatten()

        ttc50_hdi = compute_hdi(ttc50_samples)
        k_hdi = compute_hdi(k_samples)

        summary["posterior_estimates"][cond] = {
            "ttc50": {
                "mean": float(ttc50_samples.mean()),
                "median": float(np.median(ttc50_samples)),
                "std": float(ttc50_samples.std()),
                "hdi_95": [ttc50_hdi[0], ttc50_hdi[1]],
            },
            "k": {
                "mean": float(k_samples.mean()),
                "median": float(np.median(k_samples)),
                "std": float(k_samples.std()),
                "hdi_95": [k_hdi[0], k_hdi[1]],
            },
        }

    # Save to file
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary saved to {output_path}")
    return summary


# ============================================================================
# Main Pipeline
# ============================================================================


def main():
    """Main analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="MCMC Bayesian Psychophysics Analysis for Multisensory Integration"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to events CSV file OR directory containing multiple session files",
    )
    parser.add_argument(
        "--kinematics",
        type=str,
        default=None,
        help="Path to kinematics CSV file (optional, for escape detection)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_ESCAPE_THRESHOLD_MM,
        help=f"Escape detection threshold (default: {DEFAULT_ESCAPE_THRESHOLD_MM} mm)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})",
    )

    args = parser.parse_args()

    # Setup paths
    data_path = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed for reproducibility
    rng = np.random.default_rng(args.seed)

    logger.info("=" * 60)
    logger.info("MCMC Bayesian Psychophysics Analysis")
    logger.info("=" * 60)

    try:
        # Step 1: Load data - support both single file and directory of sessions
        if data_path.is_dir():
            # Load multiple sessions from directory
            trials, kinematics = load_multiple_sessions(data_path)
            if len(kinematics) > 0:
                trials = detect_escape_responses(trials, kinematics, args.threshold)
            else:
                logger.warning("No kinematics data found. Using seeded placeholder escape decisions.")
                trials["escape_decision"] = rng.binomial(1, 0.5, len(trials))
        else:
            # Load single events file
            trials = load_events(data_path)

            # Step 2: Load kinematics and detect escape responses
            if args.kinematics:
                kinematics = load_kinematics(Path(args.kinematics))
                trials = detect_escape_responses(trials, kinematics, args.threshold)
            else:
                # Use seeded RNG for reproducible placeholder decisions
                logger.warning("No kinematics data provided. Using seeded placeholder escape decisions.")
                trials["escape_decision"] = rng.binomial(1, 0.5, len(trials))

        # Step 3: Compute effective TTC
        trials = compute_effective_ttc(trials)

        # Filter valid trials (with TTC and escape decision)
        valid_trials = trials.dropna(subset=["effective_ttc_ms", "escape_decision"])
        logger.info(f"Valid trials for analysis: {len(valid_trials)}")

        if len(valid_trials) == 0:
            logger.error("No valid trials found. Check data format.")
            sys.exit(1)

        # Step 4: Prepare data for MCMC
        conditions = sorted(valid_trials["stim_condition"].unique())
        condition_map = {cond: i for i, cond in enumerate(conditions)}
        condition_idx = valid_trials["stim_condition"].map(condition_map).values

        ttc = valid_trials["effective_ttc_ms"].values
        escape = valid_trials["escape_decision"].values.astype(int)

        logger.info(f"Conditions: {conditions}")
        logger.info(f"Condition indices: {condition_map}")

        # Step 5: Build and run MCMC model
        model = build_psychometric_model(ttc, escape, condition_idx, conditions)
        trace = run_mcmc_inference(model, random_seed=args.seed)

        # Step 6: Check convergence
        converged = check_convergence(trace)
        if not converged:
            logger.warning("Convergence diagnostics failed. Results may be unreliable.")

        # Step 7: Statistical analysis
        ttc50_differences = analyze_ttc50_differences(trace, conditions)
        var_reduction = compute_variance_reduction(trace, conditions)

        # Step 8: Generate visualizations
        plot_posterior_traces(trace, conditions, output_dir / "posterior_traces.png")
        plot_psychometric_curves(trace, valid_trials, conditions, output_dir / "mcmc_psychophysics_curves.png")
        plot_posterior_distributions(trace, conditions, output_dir / "posterior_distributions.png")

        # Step 9: Generate summary
        summary = generate_summary(
            trace, conditions, ttc50_differences, var_reduction, output_dir / "mcmc_summary.json",
            converged=converged,
        )

        logger.info("=" * 60)
        logger.info("Analysis complete!")
        logger.info("=" * 60)
        logger.info(f"Results saved to: {output_dir}")
        logger.info("Output files:")
        logger.info("  - posterior_traces.png")
        logger.info("  - mcmc_psychophysics_curves.png")
        logger.info("  - posterior_distributions.png")
        logger.info("  - mcmc_summary.json")

        return summary

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
