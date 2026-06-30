"""
Cercus Framework — MCMC Bayesian Psychophysics Analysis
========================================================
Bayesian inference on multisensory integration using PyMC.

Model architecture:
- **Bimodal (looming_wind)**: P(Escape) = sigmoid(k * (TTC - TTC50))
  with data-informed priors on k and TTC50.
- **Unimodal (visual_only, wind_only)**: P(Escape) = p_baseline
  with uniform Beta(1, 1) prior.

Key analyses:
1. Psychometric function fitting per stimulus condition
2. Posterior distribution estimation for PSE (TTC50) and slope (k)
3. Hypothesis testing with ROPE-based posterior probability and
   common-scale effect sizes (P(Escape) at TTC=0)
4. Bayesian optimal integration: Variance reduction on matched
   probability scale with HDI confidence intervals
5. Posterior predictive checks with per-draw calibration
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns

from .classifier import label_trials
from .constants import _apply_publication_style
from .io import load_and_concat_sessions, scan_and_pair_sessions
from .kinematics import preprocess

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# MCMC Sampling Parameters (defaults; overridable via CLI)
# ──────────────────────────────────────────────────────────────────────

N_CHAINS: int = 4
N_DRAWS: int = 2000
N_TUNE: int = 3000
TARGET_ACCEPT: float = 0.9
RANDOM_SEED: int = 42

MAX_TTC_BINS: int = 10


# ══════════════════════════════════════════════════════════════════════
# Data Preparation
# ══════════════════════════════════════════════════════════════════════


def load_and_classify(input_dir: str | Path) -> pd.DataFrame:
    """
    Load all sessions from directory, preprocess, and classify trials.

    Extracts ``delay_sec`` from filenames (e.g. ``0.613cricket_001_…``)
    and propagates it to the DataFrame for condition classification.

    Returns
    -------
    df : DataFrame
        Fully preprocessed and classified trial data with ``delay_sec`` column.
    """
    import re

    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    subjects = scan_and_pair_sessions(input_dir)
    if not subjects:
        raise ValueError(f"No valid (events, kinematics) pairs found in {input_dir}")

    # Precompute delay_sec per subject from filename prefix
    _RE_DELAY = re.compile(r"^([\d.]+)cricket")
    delay_map: dict[str, float] = {}
    for subject_name in subjects:
        m = _RE_DELAY.match(subject_name)
        if m:
            try:
                delay_map[subject_name] = float(m.group(1))
            except ValueError:
                pass

    all_dfs = []
    for subject_name, sessions in subjects.items():
        log.info("Processing subject: %s (%d sessions)", subject_name, len(sessions))

        meta, windows, anchors, kin, trial_map = load_and_concat_sessions(sessions)
        df = preprocess(meta, windows, anchors, kin)
        df["global_trial_index"] = df["global_trial_id"]
        df = label_trials(df)
        df["subject"] = subject_name

        if subject_name in delay_map:
            df["delay_sec"] = delay_map[subject_name]
            log.info("  delay_sec = %.3f (from filename)", delay_map[subject_name])

        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


def prepare_mcmc_data(
    df: pd.DataFrame,
    binary_mode: str = "escape_only",
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """
    Aggregate frame-level data to trial level and extract MCMC arrays.

    Parameters
    ----------
    df : DataFrame
        Classified frame-level data from ``load_and_classify()``.
    binary_mode : str
        ``"escape_only"`` or ``"escape_prewalk"``.

    Returns
    -------
    ttc, escape, conditions, condition_idx
    """
    if "subject" in df.columns:
        df = df.copy()
        df["unique_trial_id"] = df["subject"].astype(str) + "_" + df["global_trial_id"].astype(str)
    else:
        df = df.copy()
        df["unique_trial_id"] = df["global_trial_id"].astype(str)

    agg_cols = ["type", "response_type", "target_ttc_ms"]
    if "delay_sec" in df.columns:
        agg_cols.append("delay_sec")
    agg_cols = [c for c in agg_cols if c in df.columns]

    trials = df.groupby("unique_trial_id")[agg_cols].first().reset_index()

    valid = trials[trials["type"].notna() & (trials["type"] != "unknown")].copy()
    if valid.empty:
        raise ValueError("No valid trials with type information")

    valid["stim_condition"] = _classify_conditions(valid)

    if binary_mode == "escape_only":
        valid["escape_binary"] = (valid["response_type"] == "Escape").astype(int)
    elif binary_mode == "escape_prewalk":
        valid["escape_binary"] = valid["response_type"].isin(["Escape", "PreWalk"]).astype(int)
    else:
        raise ValueError(f"Unknown binary_mode: {binary_mode}")

    valid["effective_ttc_ms"] = _compute_effective_ttc(valid)

    # For bimodal trials, require valid TTC; for unimodal, TTC is NaN (acceptable)
    is_bimodal = valid["stim_condition"].str.startswith("looming_wind_")
    keep = (~is_bimodal) | (is_bimodal & valid["effective_ttc_ms"].notna())
    valid = valid[keep].copy()

    if valid.empty:
        raise ValueError("No trials with valid data")

    conditions = sorted(valid["stim_condition"].unique())
    condition_map = {c: i for i, c in enumerate(conditions)}
    condition_idx = valid["stim_condition"].map(condition_map).values.astype(np.int32)

    ttc = valid["effective_ttc_ms"].values.astype(np.float64)
    escape = valid["escape_binary"].values.astype(np.int32)

    log.info("Prepared %d trials across %d conditions", len(ttc), len(conditions))
    for c in conditions:
        n = (valid["stim_condition"] == c).sum()
        n_esc = valid.loc[valid["stim_condition"] == c, "escape_binary"].sum()
        log.info("  %s: %d trials, %d escape (%.1f%%)", c, n, n_esc, 100 * n_esc / n)

    return ttc, escape, conditions, condition_idx


def _classify_conditions(df: pd.DataFrame) -> pd.Series:
    """Classify trials into stimulus conditions.

    Uses ``delay_sec`` (from filename) to distinguish looming_wind sub-conditions
    when available.  Groups large numbers of unique TTC values into bins.
    """
    conditions = pd.Series("unknown", index=df.index)

    ttc_ms = (
        pd.to_numeric(df["target_ttc_ms"], errors="coerce")
        if "target_ttc_ms" in df.columns
        else pd.Series(np.nan, index=df.index)
    )

    # Unimodal conditions
    conditions[df["type"] == "baseline_visual"] = "visual_only"
    conditions[df["type"].isin(["baseline_wind", "wind_only"])] = "wind_only"

    # Bimodal: group TTC values, optionally stratified by delay_sec
    mask_lw = df["type"] == "looming_wind"
    has_ttc = mask_lw & ttc_ms.notna()

    if has_ttc.any():
        has_delay = "delay_sec" in df.columns and df.loc[has_ttc, "delay_sec"].notna().any()

        if has_delay:
            # Stratify by delay_sec → each delay is a separate condition group
            delays = df.loc[has_ttc, "delay_sec"]
            for delay_val in sorted(delays.dropna().unique()):
                sub = has_ttc & (df["delay_sec"] == delay_val)
                lw_ttc = ttc_ms[sub]
                delay_label = f"d{delay_val:.3f}".rstrip("0").rstrip(".")
                if lw_ttc.nunique() > MAX_TTC_BINS:
                    try:
                        bins = pd.qcut(lw_ttc, q=MAX_TTC_BINS, duplicates="drop")
                        midpoints = bins.apply(lambda x: round((x.left + x.right) / 2)).astype(int)
                        conditions[sub] = "looming_wind_" + delay_label + "_" + midpoints.astype(str)
                    except ValueError:
                        rounded = (lw_ttc / 50).round().mul(50).astype(int)
                        conditions[sub] = "looming_wind_" + delay_label + "_" + rounded.astype(str)
                else:
                    conditions[sub] = "looming_wind_" + delay_label + "_" + lw_ttc.round().astype(int).astype(str)
        else:
            # No delay info: original grouping by TTC only
            lw_ttc = ttc_ms[has_ttc]
            if lw_ttc.nunique() > MAX_TTC_BINS:
                try:
                    bins = pd.qcut(lw_ttc, q=MAX_TTC_BINS, duplicates="drop")
                    midpoints = bins.apply(lambda x: round((x.left + x.right) / 2)).astype(int)
                    conditions[has_ttc] = "looming_wind_" + midpoints.astype(str)
                except ValueError:
                    conditions[has_ttc] = "looming_wind_" + (lw_ttc / 50).round().mul(50).astype(int).astype(str)
            else:
                conditions[has_ttc] = "looming_wind_" + lw_ttc.round().astype(int).astype(str)

    return conditions


def _compute_effective_ttc(df: pd.DataFrame) -> pd.Series:
    """Compute effective TTC for each trial.

    - **visual_only / wind_only**: TTC is NaN (no temporal dimension in
      these unimodal reference conditions; modelled as separate intercepts).
    - **looming_wind_***: TTC = ``target_ttc_ms``.
    """
    ttc = pd.Series(np.nan, index=df.index)
    ttc_ms = (
        pd.to_numeric(df["target_ttc_ms"], errors="coerce")
        if "target_ttc_ms" in df.columns
        else pd.Series(np.nan, index=df.index)
    )

    # Only bimodal conditions have meaningful TTC
    mask_lw = df["stim_condition"].str.startswith("looming_wind_")
    has_ttc = mask_lw & ttc_ms.notna()
    ttc[has_ttc] = ttc_ms[has_ttc]

    return ttc


# ══════════════════════════════════════════════════════════════════════
# Model Data Splitting
# ══════════════════════════════════════════════════════════════════════


def split_bimodal_unimodal(
    ttc: np.ndarray,
    escape: np.ndarray,
    conditions: List[str],
    condition_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray, List[str], np.ndarray]:
    """Split data into bimodal (looming_wind_*) and unimodal groups.

    Returns
    -------
    bi_ttc, bi_escape, bi_conditions, bi_condition_idx,
    uni_escape, uni_conditions, uni_condition_idx
    """
    bi_conditions = sorted([c for c in conditions if c.startswith("looming_wind_")])
    uni_conditions = sorted([c for c in conditions if not c.startswith("looming_wind_")])

    bi_map = {c: i for i, c in enumerate(bi_conditions)}
    uni_map = {c: i for i, c in enumerate(uni_conditions)}

    bi_mask = np.array([conditions[c] in bi_map for c in condition_idx])
    uni_mask = ~bi_mask

    bi_ttc = ttc[bi_mask]
    bi_escape = escape[bi_mask]
    bi_condition_idx = np.array(
        [bi_map[conditions[c]] for c in condition_idx[bi_mask]], dtype=np.int32
    )

    uni_escape = escape[uni_mask]
    uni_condition_idx = np.array(
        [uni_map[conditions[c]] for c in condition_idx[uni_mask]], dtype=np.int32
    )

    log.info("Bimodal: %d trials, %d conditions", len(bi_ttc), len(bi_conditions))
    log.info("Unimodal: %d trials, %d conditions", len(uni_escape), len(uni_conditions))

    return (
        bi_ttc, bi_escape, bi_conditions, bi_condition_idx,
        uni_escape, uni_conditions, uni_condition_idx,
    )


# ══════════════════════════════════════════════════════════════════════
# MCMC Model
# ══════════════════════════════════════════════════════════════════════


def build_psychometric_model(
    bi_ttc: np.ndarray,
    bi_escape: np.ndarray,
    bi_condition_idx: np.ndarray,
    bi_condition_names: List[str],
    uni_escape: np.ndarray | None = None,
    uni_condition_idx: np.ndarray | None = None,
    uni_condition_names: List[str] | None = None,
) -> pm.Model:
    """
    Build Bayesian psychometric model with separated bimodal/unimodal structure.

    Bimodal (looming_wind) conditions:
        P(Escape) = sigmoid(k * (TTC - TTC50))

    Unimodal (visual_only, wind_only) conditions:
        P(Escape) = p_baseline    (constant, independent of TTC)

    Prior specification
    -------------------
    - TTC50 ~ Normal(data_median, data_range/2): weakly informative, centred
      on observed TTC median.
    - k ~ LogNormal(mu_k, 0.5): positive slope with data-informed scale.
      ``mu_k`` is calibrated so the median slope produces a 10%-90% rise
      over approximately half the observed TTC range.
    - p_baseline ~ Beta(1, 1): uniform (non-informative) prior for unimodal
      escape rates.
    """
    n_bi = len(bi_condition_names)
    n_uni = len(uni_condition_names) if uni_condition_names else 0

    # Data-informed prior calibration
    if len(bi_ttc) > 0:
        ttc_min, ttc_max = float(np.nanmin(bi_ttc)), float(np.nanmax(bi_ttc))
        ttc_range = max(ttc_max - ttc_min, 100.0)
        ttc_mid = (ttc_min + ttc_max) / 2.0
    else:
        ttc_range = 1000.0
        ttc_mid = 0.0

    # Slope prior: median k * ttc_range_half ≈ 4.4 (10%-90% sigmoid rise)
    mu_k = np.log(4.4 / (ttc_range / 2.0))
    mu_k = min(mu_k, 0.0)  # Clamp to prevent overflow in small TTC ranges
    sigma_k = 0.5

    log.info("Prior calibration: TTC range = %.0f ms, mu_k = %.3f, sigma_k = %.2f",
             ttc_range, mu_k, sigma_k)

    with pm.Model() as model:
        # ── Bimodal parameters ──
        if n_bi > 0:
            init_ttc50 = np.array([
                np.median(bi_ttc[bi_condition_idx == i])
                if (bi_condition_idx == i).any() else ttc_mid
                for i in range(n_bi)
            ])
            ttc50 = pm.Normal(
                "ttc50", mu=ttc_mid, sigma=ttc_range / 2.0,
                shape=n_bi, dims="bi_condition",
                initval=init_ttc50,
            )
            k = pm.LogNormal(
                "k", mu=mu_k, sigma=sigma_k,
                shape=n_bi, dims="bi_condition",
            )
            mu_bi = k[bi_condition_idx] * (bi_ttc - ttc50[bi_condition_idx])
            p_bi = pm.Deterministic("p_bi", pm.math.sigmoid(mu_bi))

        # ── Unimodal parameters ──
        if n_uni > 0:
            p_baseline = pm.Beta(
                "p_baseline", alpha=1, beta=1,
                shape=n_uni, dims="uni_condition",
            )

        # ── Likelihood (single Bernoulli) ──
        if n_bi > 0 and n_uni > 0:
            p_escape = pm.math.concatenate([p_bi, p_baseline[uni_condition_idx]])
            all_escape = np.concatenate([bi_escape, uni_escape])
        elif n_bi > 0:
            p_escape = p_bi
            all_escape = bi_escape
        else:
            p_escape = p_baseline[uni_condition_idx]
            all_escape = uni_escape

        p_escape_obs = pm.Deterministic("p_escape_obs", p_escape)
        pm.Bernoulli("escape_obs", p=p_escape_obs, observed=all_escape)

        # ── Derived: TTC50 as a named coordinate variable for ArviZ ──
        if n_bi > 0:
            pm.Deterministic("ttc50_named", ttc50, dims="bi_condition")
            pm.Deterministic("k_named", k, dims="bi_condition")

    return model


# ══════════════════════════════════════════════════════════════════════
# MCMC Sampling & Convergence
# ══════════════════════════════════════════════════════════════════════


def run_mcmc(
    model: pm.Model,
    random_seed: int = RANDOM_SEED,
    n_chains: int = N_CHAINS,
    n_draws: int = N_DRAWS,
    n_tune: int = N_TUNE,
) -> az.InferenceData:
    """Run MCMC sampling using NUTS sampler.

    Automatically uses numpyro (JAX) if available for 10-100x speedup.
    """
    log.info("Running MCMC: %d chains, %d draws, %d tune, target_accept=%.2f",
             n_chains, n_draws, n_tune, TARGET_ACCEPT)

    try:
        import numpyro  # noqa: F401
        sampler = "numpyro"
        log.info("Using numpyro (JAX) sampler for fast inference")
    except ImportError:
        sampler = "pymc"
        log.info("numpyro not found, using default PyMC sampler")

    with model:
        trace = pm.sample(
            draws=n_draws,
            chains=n_chains,
            tune=n_tune,
            target_accept=TARGET_ACCEPT,
            random_seed=random_seed,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": False},
            nuts_sampler=sampler,
        )

    return trace


def check_convergence(trace: az.InferenceData) -> Dict[str, Any]:
    """Check MCMC convergence with full diagnostic metrics.

    Returns
    -------
    dict
        ``{"converged": bool, "rhat_ok": bool, "ess_ok": bool,
          "rhat_max": float, "ess_bulk_min": float, "ess_tail_min": float}``
    """
    var_names = []
    for v in ["ttc50_named", "k_named", "p_baseline"]:
        if v in trace.posterior:
            var_names.append(v)
    if not var_names:
        var_names = ["ttc50", "k"]

    summary = az.summary(trace, var_names=var_names)
    log.info("Convergence diagnostics:\n%s", summary.to_string())

    rhat_vals = summary["r_hat"].values
    ess_bulk_vals = summary["ess_bulk"].values
    ess_tail_vals = summary["ess_tail"].values if "ess_tail" in summary.columns else ess_bulk_vals

    rhat_max = float(np.nanmax(rhat_vals))
    ess_bulk_min = float(np.nanmin(ess_bulk_vals))
    ess_tail_min = float(np.nanmin(ess_tail_vals))

    rhat_ok = bool(rhat_max < 1.01)
    ess_ok = bool(ess_bulk_min > 400 and ess_tail_min > 400)

    if not rhat_ok:
        log.warning("R-hat = %.4f > 1.01 detected", rhat_max)
    if not ess_ok:
        if ess_bulk_min <= 400:
            log.warning("ESS_bulk min = %.0f < 400 detected", ess_bulk_min)
        if ess_tail_min <= 400:
            log.warning("ESS_tail min = %.0f < 400 detected", ess_tail_min)

    return {
        "converged": rhat_ok and ess_ok,
        "rhat_ok": rhat_ok,
        "ess_ok": ess_ok,
        "rhat_max": rhat_max,
        "ess_bulk_min": ess_bulk_min,
        "ess_tail_min": ess_tail_min,
    }


# ══════════════════════════════════════════════════════════════════════
# Posterior Extraction Helpers
# ══════════════════════════════════════════════════════════════════════


def get_ttc50_posterior(
    trace: az.InferenceData,
    bi_conditions: List[str],
) -> np.ndarray:
    """Extract TTC50 posterior samples as (n_chains, n_draws, n_bi_conditions)."""
    return trace.posterior["ttc50_named"].values


def get_k_posterior(
    trace: az.InferenceData,
    bi_conditions: List[str],
) -> np.ndarray:
    """Extract k posterior samples as (n_chains, n_draws, n_bi_conditions)."""
    return trace.posterior["k_named"].values


def get_p_baseline_posterior(
    trace: az.InferenceData,
    uni_conditions: List[str],
) -> Dict[str, np.ndarray]:
    """Extract p_baseline posteriors per unimodal condition."""
    if "p_baseline" not in trace.posterior:
        return {}
    pb = trace.posterior["p_baseline"].values  # (chains, draws, n_uni)
    return {c: pb[:, :, i].flatten() for i, c in enumerate(uni_conditions)}


# ══════════════════════════════════════════════════════════════════════
# Statistical Analysis
# ══════════════════════════════════════════════════════════════════════


def compute_hdi(samples: np.ndarray, credible_mass: float = 0.95) -> Tuple[float, float]:
    """Compute Highest Density Interval (HDI).

    NaN values are silently filtered before computation.
    Returns ``(nan, nan)`` if no finite samples remain.
    """
    samples = np.asarray(samples, dtype=np.float64)
    samples = samples[~np.isnan(samples)]
    if len(samples) == 0:
        return (np.nan, np.nan)
    sorted_samples = np.sort(samples)
    n = len(sorted_samples)
    interval_size = int(np.ceil(credible_mass * n))
    if interval_size >= n:
        return float(sorted_samples[0]), float(sorted_samples[-1])
    widths = sorted_samples[interval_size:] - sorted_samples[:n - interval_size]
    min_idx = np.argmin(widths)
    return float(sorted_samples[min_idx]), float(sorted_samples[min_idx + interval_size])


def _cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d effect size.

    .. note::
        When applied to posterior samples (as in
        ``analyze_ttc50_differences``), the pooled standard deviation
        reflects **posterior uncertainty** rather than data variability.
        The resulting value is therefore a *descriptive* summary of
        separation between two posterior distributions on a common
        probability scale — it is **not** a classical Neyman-Pearson
        effect size computed from raw observations.
    """
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(
        ((nx - 1) * x.std() ** 2 + (ny - 1) * y.std() ** 2) / (nx + ny - 2)
    )
    if pooled_std == 0:
        return 0.0
    return float((x.mean() - y.mean()) / pooled_std)



def analyze_ttc50_differences(
    trace: az.InferenceData,
    bi_conditions: List[str],
    uni_conditions: List[str],
    rope_ttc50: Tuple[float, float] = (-50.0, 50.0),
) -> Dict[str, Any]:
    """Analyze posterior TTC50 with ROPE-based inference and common-scale comparisons.

    For each bimodal condition:
    - Reports TTC50 posterior summary and 95% HDI.
    - P(TTC50 > 0 | data): probability that bimodal integration shifts PSE
      toward later time points.
    - ROPE assessment: P(TTC50 in ROPE | data) for practical equivalence.
    - Common-scale comparison with visual-only: computes P(Escape) at TTC=0
      for bimodal and compares with visual-only escape rate posterior,
      both on the probability scale.  Reports Cohen's d on this common scale.

    Parameters
    ----------
    rope_ttc50 : tuple of float
        Region of Practical Equivalence for TTC50 in ms.  Default (-50, 50).
    """
    ttc50_post = get_ttc50_posterior(trace, bi_conditions)
    p_baseline_post = get_p_baseline_posterior(trace, uni_conditions)

    vis_p = p_baseline_post.get("visual_only")

    results: Dict[str, Any] = {}

    for i, cond in enumerate(bi_conditions):
        cond_post = ttc50_post[:, :, i].flatten()
        key = f"visual_only_vs_{cond}"

        hdi_low, hdi_high = compute_hdi(cond_post)
        prob_above_zero = float((cond_post > 0).mean())
        prob_in_rope = float(
            ((cond_post >= rope_ttc50[0]) & (cond_post <= rope_ttc50[1])).mean()
        )

        entry: Dict[str, Any] = {
            "ttc50_mean": float(cond_post.mean()),
            "ttc50_hdi_95": [hdi_low, hdi_high],
            "prob_ttc50_positive": prob_above_zero,
            "rope": list(rope_ttc50),
            "prob_ttc50_in_rope": prob_in_rope,
        }

        if vis_p is not None:
            # Common-scale comparison: P(Escape) at TTC=0 for bimodal
            k_post = get_k_posterior(trace, bi_conditions)[:, :, i].flatten()
            p_at_zero = 1.0 / (1.0 + np.exp(k_post * cond_post))  # sigmoid(-k*TTC50)

            # Cohen's d on common probability scale (not ms vs probability)
            d = _cohen_d(p_at_zero, vis_p)

            entry["p_escape_at_ttc0_mean"] = float(p_at_zero.mean())
            entry["p_escape_at_ttc0_hdi_95"] = list(compute_hdi(p_at_zero))
            entry["visual_escape_mean"] = float(vis_p.mean())
            entry["visual_escape_hdi_95"] = list(compute_hdi(vis_p))
            entry["cohen_d_common_scale"] = d

        results[key] = entry

    return results


def compute_variance_reduction(
    trace: az.InferenceData,
    bi_conditions: List[str],
    uni_conditions: List[str],
    n_bootstrap: int = 2000,
) -> Dict[str, Any]:
    """Compute posterior variance reduction on a common probability scale.

    For each bimodal condition, computes P(Escape) at TTC=0 from posterior
    samples of (k, TTC50), then compares posterior variance of this
    P(Escape) against visual-only escape rate posterior variance.  Both
    quantities are on the probability scale [0, 1], so the ratio is
    dimensionally consistent.

    Returns
    -------
    dict
        Per-condition variance reduction with 95% HDI via bootstrap.
    """
    ttc50_post = get_ttc50_posterior(trace, bi_conditions)
    p_baseline_post = get_p_baseline_posterior(trace, uni_conditions)

    results: Dict[str, Any] = {}

    vis_p = p_baseline_post.get("visual_only")
    if vis_p is not None:
        results["visual_escape_variance"] = float(vis_p.var())
        results["visual_escape_mean"] = float(vis_p.mean())

    wind_p = p_baseline_post.get("wind_only")
    if wind_p is not None:
        results["wind_escape_variance"] = float(wind_p.var())
        results["wind_escape_mean"] = float(wind_p.mean())

    for i, cond in enumerate(bi_conditions):
        ttc50_s = ttc50_post[:, :, i].flatten()
        k_s = get_k_posterior(trace, bi_conditions)[:, :, i].flatten()

        # P(Escape) at TTC=0: sigmoid(-k * TTC50) — on probability scale
        p_at_zero = 1.0 / (1.0 + np.exp(k_s * ttc50_s))
        cond_var = float(p_at_zero.var())

        results[f"{cond}_p_escape_at_ttc0_variance"] = cond_var
        results[f"{cond}_p_escape_at_ttc0_mean"] = float(p_at_zero.mean())

        # Variance reduction relative to visual-only (both on probability scale)
        if vis_p is not None and vis_p.var() > 0:
            rng = np.random.default_rng(42)
            reductions = np.empty(n_bootstrap)
            n_vis = len(vis_p)
            n_cond = len(p_at_zero)
            for b in range(n_bootstrap):
                vis_b = rng.choice(vis_p, size=n_vis, replace=True)
                cond_b = rng.choice(p_at_zero, size=n_cond, replace=True)
                reductions[b] = 1.0 - cond_b.var() / vis_b.var()

            hdi_lo, hdi_hi = compute_hdi(reductions)
            results[f"{cond}_variance_reduction"] = float(1.0 - cond_var / vis_p.var())
            results[f"{cond}_variance_reduction_hdi_95"] = [hdi_lo, hdi_hi]

    return results


# ══════════════════════════════════════════════════════════════════════
# Posterior Predictive Checks
# ══════════════════════════════════════════════════════════════════════


def posterior_predictive_check(
    trace: az.InferenceData,
    model: pm.Model,
    bi_escape: np.ndarray,
    uni_escape: np.ndarray,
    bi_conditions: List[str],
    uni_conditions: List[str],
    bi_condition_idx: np.ndarray,
    uni_condition_idx: np.ndarray,
) -> Dict[str, Any]:
    """Run posterior predictive checks and compute Bayesian p-values.

    Calibration is computed per posterior draw and then summarised with HDI,
    avoiding the pitfall of collapsing to point-estimate predictions.

    Bayesian p-value = P(T(y_rep) >= T(y_obs) | y), where T is a summary
    statistic (per-condition escape rate).  Values near 0.5 indicate good
    calibration; values near 0 or 1 indicate systematic misfit.
    """
    with model:
        ppc = pm.sample_posterior_predictive(trace, random_seed=42)

    y_rep = ppc.posterior_predictive["escape_obs"].values  # (chains, draws, n_obs)
    n_chains, n_draws, n_obs = y_rep.shape
    y_rep_flat = y_rep.reshape(-1, n_obs)

    all_escape = np.concatenate([bi_escape, uni_escape])
    n_bi = len(bi_escape)

    results: Dict[str, Any] = {}

    # Per-condition Bayesian p-value
    all_conditions = bi_conditions + uni_conditions
    all_condition_idx = np.concatenate([bi_condition_idx, uni_condition_idx + len(bi_conditions)])

    for ci, cond in enumerate(all_conditions):
        mask = all_condition_idx == ci
        if not mask.any():
            continue
        obs_rate = all_escape[mask].mean()
        rep_rates = y_rep_flat[:, mask].mean(axis=1)
        bayes_p = float((rep_rates >= obs_rate).mean())
        results[cond] = {
            "observed_escape_rate": float(obs_rate),
            "ppc_mean_rate": float(rep_rates.mean()),
            "bayesian_p_value": bayes_p,
        }

    # Calibration: compute per posterior draw, then summarise with HDI
    n_bins = 5
    try:
        bin_edges = np.quantile(all_escape.astype(float), np.linspace(0, 1, n_bins + 1))
        bin_edges[0] -= 0.01   # ensure inclusion of minimum value
        bin_edges[-1] += 0.01
        trial_bins = np.digitize(all_escape.astype(float), bin_edges[1:-1])  # 0-indexed

        calibration: Dict[str, Any] = {}
        for b in range(n_bins):
            bin_mask = trial_bins == b
            if not bin_mask.any():
                continue
            obs_mean = float(all_escape[bin_mask].mean())
            # Per-draw predicted rate for this bin
            pred_per_draw = y_rep_flat[:, bin_mask].mean(axis=1)
            pred_lo, pred_hi = compute_hdi(pred_per_draw)
            calibration[f"bin_{b}"] = {
                "observed_mean": obs_mean,
                "ppc_predicted_mean": float(pred_per_draw.mean()),
                "ppc_predicted_hdi_95": [pred_lo, pred_hi],
                "n_trials": int(bin_mask.sum()),
            }
        results["calibration"] = calibration
    except (ValueError, IndexError) as exc:
        log.warning("Calibration computation failed: %s", exc)

    return results


# ══════════════════════════════════════════════════════════════════════
# Visualization
# ══════════════════════════════════════════════════════════════════════


def _generate_color_palette(
    bi_conditions: List[str],
    uni_conditions: List[str],
) -> Dict[str, str]:
    """Generate Lancet/Cell style colour palette.

    - visual_only -> Navy Blue
    - wind_only -> Sand Orange
    - looming_wind_* -> gradient from Crimson Red to Slate Grey
    """
    from .constants import COLOR_LEFT, COLOR_OSCI_HW, COLOR_RIGHT, COLOR_CONTROL

    palette: Dict[str, str] = {}
    for cond in uni_conditions:
        if cond == "visual_only":
            palette[cond] = COLOR_LEFT
        elif cond == "wind_only":
            palette[cond] = COLOR_OSCI_HW
        else:
            palette[cond] = COLOR_CONTROL

    n_bi = len(bi_conditions)
    for idx, cond in enumerate(bi_conditions):
        if n_bi <= 1:
            palette[cond] = COLOR_RIGHT
        else:
            r1, g1, b1 = int(COLOR_RIGHT[1:3], 16), int(COLOR_RIGHT[3:5], 16), int(COLOR_RIGHT[5:7], 16)
            r2, g2, b2 = int(COLOR_CONTROL[1:3], 16), int(COLOR_CONTROL[3:5], 16), int(COLOR_CONTROL[5:7], 16)
            t = idx / (n_bi - 1)
            r = int(r1 + (r2 - r1) * t)
            g = int(g1 + (g2 - g1) * t)
            b = int(b1 + (b2 - b1) * t)
            palette[cond] = f"#{r:02x}{g:02x}{b:02x}"

    return palette


def plot_posterior_traces(
    trace: az.InferenceData,
    bi_conditions: List[str],
    uni_conditions: List[str],
    output_path: Path,
) -> None:
    """Generate posterior trace plots for MCMC diagnostics.

    Shows TTC50 and k traces for bimodal conditions, plus p_baseline
    traces for unimodal conditions.
    """
    panels: List[Tuple[str, str, str]] = []  # (param, condition, ylabel)
    for cond in bi_conditions:
        panels.append(("ttc50_named", cond, f"{cond}\nTTC50 (ms)"))
        panels.append(("k_named", cond, "k (slope)"))
    for cond in uni_conditions:
        panels.append(("p_baseline", cond, f"{cond}\np_escape"))

    n = len(panels)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 1, figsize=(10, 2.5 * n), squeeze=False)
    axes = axes.flatten()

    for ax, (param, cond, ylabel) in zip(axes, panels):
        if param in trace.posterior:
            # Try to select by condition dimension
            try:
                for chain in range(trace.posterior.sizes["chain"]):
                    samples = trace.posterior[param].sel(chain=chain, **{trace.posterior[param].dims[-1]: cond}).values
                    ax.plot(samples, alpha=0.7, linewidth=0.5, label=f"Chain {chain}")
            except (KeyError, ValueError):
                for chain in range(trace.posterior.sizes["chain"]):
                    samples = trace.posterior[param].sel(chain=chain).values.flatten()
                    ax.plot(samples, alpha=0.7, linewidth=0.5, label=f"Chain {chain}")

        ax.set_ylabel(ylabel)
        ax.set_xlabel("Sample")
        ax.legend(fontsize=6, ncol=min(4, trace.posterior.sizes["chain"]))

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_psychometric_curves(
    trace: az.InferenceData,
    bi_ttc: np.ndarray,
    bi_escape: np.ndarray,
    bi_conditions: List[str],
    bi_condition_idx: np.ndarray,
    uni_conditions: List[str],
    uni_escape: np.ndarray | None,
    uni_condition_idx: np.ndarray | None,
    output_path: Path,
) -> None:
    """Generate psychometric curves with 95% HDI shading.

    Bimodal conditions: fitted sigmoid curves with data points.
    Unimodal conditions: horizontal reference lines with data rates.
    """
    palette = _generate_color_palette(bi_conditions, uni_conditions)

    # Determine TTC grid range
    if len(bi_ttc) > 0:
        ttc_lo, ttc_hi = bi_ttc.min() - 50, bi_ttc.max() + 50
    else:
        ttc_lo, ttc_hi = -500, 500
    ttc_grid = np.linspace(ttc_lo, ttc_hi, 200)

    fig, ax = plt.subplots(figsize=(8, 5))

    # ── Bimodal conditions: sigmoid curves ──
    if len(bi_conditions) > 0:
        ttc50_post = get_ttc50_posterior(trace, bi_conditions)
        k_post = get_k_posterior(trace, bi_conditions)

        for i, cond in enumerate(bi_conditions):
            color = palette[cond]
            t50_s = ttc50_post[:, :, i].flatten()
            k_s = k_post[:, :, i].flatten()

            n_curves = min(500, len(t50_s))
            idx = np.random.default_rng(42).choice(len(t50_s), n_curves, replace=False)
            preds = np.zeros((n_curves, len(ttc_grid)))
            for j, (t50, kk) in enumerate(zip(t50_s[idx], k_s[idx])):
                preds[j] = 1.0 / (1.0 + np.exp(-kk * (ttc_grid - t50)))

            median = np.median(preds, axis=0)
            hdi_lo = np.zeros(len(ttc_grid))
            hdi_hi = np.zeros(len(ttc_grid))
            for col in range(len(ttc_grid)):
                hdi_lo[col], hdi_hi[col] = compute_hdi(preds[:, col])

            ax.plot(ttc_grid, median, color=color, linewidth=1.5, label=cond)
            ax.fill_between(ttc_grid, hdi_lo, hdi_hi, color=color, alpha=0.15)

            # Data points (binned)
            mask = bi_condition_idx == i
            cond_ttc, cond_esc = bi_ttc[mask], bi_escape[mask]
            n_bins = max(3, min(10, len(cond_ttc) // 5))
            try:
                bins = pd.qcut(cond_ttc, n_bins, duplicates="drop")
                binned = pd.DataFrame({"ttc": cond_ttc, "esc": cond_esc, "bin": bins})
                grp = binned.groupby("bin", observed=True)
                mean_ttc = grp["ttc"].mean()
                mean_esc = grp["esc"].mean()
                sem = grp["esc"].std() / np.sqrt(grp["esc"].count())
                ax.errorbar(mean_ttc, mean_esc, yerr=sem, fmt="o", color=color,
                            markersize=5, capsize=3, linewidth=1.0, markeredgewidth=0.8)
            except (ValueError, KeyError):
                ax.scatter(cond_ttc, cond_esc, color=color, alpha=0.3, s=15)

    # ── Unimodal conditions: horizontal reference lines ──
    p_baseline_post = get_p_baseline_posterior(trace, uni_conditions)
    for i, cond in enumerate(uni_conditions):
        color = palette[cond]
        if cond in p_baseline_post:
            pb = p_baseline_post[cond]
            rate = float(np.median(pb))
            hdi_lo, hdi_hi = compute_hdi(pb)
            ax.axhline(rate, color=color, linewidth=1.5, linestyle="--", label=f"{cond} ({rate:.2f})")
            ax.axhspan(hdi_lo, hdi_hi, color=color, alpha=0.07)

        # Show data rate as a point at TTC=0
        if uni_escape is not None and uni_condition_idx is not None:
            mask = uni_condition_idx == i
            if mask.any():
                rate_data = uni_escape[mask].mean()
                ax.plot(0, rate_data, "s", color=color, markersize=8, markeredgecolor="black",
                        markeredgewidth=0.8)

    ax.axhline(0.5, color="grey", linestyle=":", alpha=0.5, linewidth=0.75)
    ax.axvline(0, color="grey", linestyle=":", alpha=0.5, linewidth=0.75)
    ax.set_xlabel("Time-to-Collision (ms)\n(negative = wind before collision)")
    ax.set_ylabel("P(Escape)")
    ax.set_title("Psychometric Functions: Multisensory Integration\n(95% HDI shaded)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", fontsize=7)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_posterior_distributions(
    trace: az.InferenceData,
    bi_conditions: List[str],
    uni_conditions: List[str],
    output_path: Path,
) -> None:
    """Plot posterior distributions of TTC50, k, and p_baseline."""
    palette = _generate_color_palette(bi_conditions, uni_conditions)

    n_panels = 2 + (1 if uni_conditions else 0)
    fig, axes = plt.subplots(n_panels, 1, figsize=(8, 3.5 * n_panels))
    if n_panels == 1:
        axes = [axes]

    # TTC50
    ax = axes[0]
    for i, cond in enumerate(bi_conditions):
        s = get_ttc50_posterior(trace, bi_conditions)[:, :, i].flatten()
        ax.hist(s, bins=50, alpha=0.5, density=True, label=cond, color=palette[cond])
        lo, hi = compute_hdi(s)
        ax.axvline(lo, color=palette[cond], linestyle="--", alpha=0.5, linewidth=0.75)
        ax.axvline(hi, color=palette[cond], linestyle="--", alpha=0.5, linewidth=0.75)
    ax.set_xlabel("TTC50 (ms)")
    ax.set_ylabel("Density")
    ax.set_title("Posterior: TTC50 (PSE)")
    ax.legend(fontsize=7)

    # k
    ax = axes[1]
    for i, cond in enumerate(bi_conditions):
        s = get_k_posterior(trace, bi_conditions)[:, :, i].flatten()
        ax.hist(s, bins=50, alpha=0.5, density=True, label=cond, color=palette[cond])
    ax.set_xlabel("k (slope)")
    ax.set_ylabel("Density")
    ax.set_title("Posterior: Slope (k)")
    ax.legend(fontsize=7)

    # p_baseline
    if uni_conditions:
        ax = axes[2]
        p_baseline_post = get_p_baseline_posterior(trace, uni_conditions)
        for cond in uni_conditions:
            if cond in p_baseline_post:
                s = p_baseline_post[cond]
                ax.hist(s, bins=50, alpha=0.5, density=True, label=cond, color=palette[cond])
                lo, hi = compute_hdi(s)
                ax.axvline(lo, color=palette[cond], linestyle="--", alpha=0.5, linewidth=0.75)
                ax.axvline(hi, color=palette[cond], linestyle="--", alpha=0.5, linewidth=0.75)
        ax.set_xlabel("P(Escape)")
        ax.set_ylabel("Density")
        ax.set_title("Posterior: Unimodal Baseline Escape Rate")
        ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ttc50_differences(
    ttc50_diff: Dict[str, Any],
    bi_conditions: List[str],
    trace: az.InferenceData,
    output_path: Path,
) -> None:
    """Plot posterior distributions of TTC50 with ROPE-based inference.

    Each panel shows the TTC50 posterior histogram with 95% HDI, ROPE
    shading, and annotation of P(TTC50 > 0) and P(in ROPE).
    """
    comparisons = {k: v for k, v in ttc50_diff.items() if "ttc50_mean" in v}
    if not comparisons:
        return

    ttc50_post = get_ttc50_posterior(trace, bi_conditions)
    n = len(comparisons)
    fig, axes = plt.subplots(n, 1, figsize=(8, 3.5 * n))
    if n == 1:
        axes = [axes]

    for ax, (key, vals) in zip(axes, comparisons.items()):
        cond_name = key.replace("visual_only_vs_", "")
        if cond_name in bi_conditions:
            ci = bi_conditions.index(cond_name)
            samples = ttc50_post[:, :, ci].flatten()
            ax.hist(samples, bins=60, alpha=0.6, density=True, color="#00468B")
            lo, hi = vals["ttc50_hdi_95"]
            ax.axvline(lo, color="#ED0000", linestyle="--", linewidth=1.0)
            ax.axvline(hi, color="#ED0000", linestyle="--", linewidth=1.0)
            ax.axvline(0, color="grey", linestyle=":", linewidth=0.75)

            # ROPE shading
            rope = vals.get("rope", [-50.0, 50.0])
            ax.axvspan(rope[0], rope[1], color="green", alpha=0.08)

            # Annotation with ROPE-based statistics
            prob_pos = vals.get("prob_ttc50_positive", 0.5)
            prob_rope = vals.get("prob_ttc50_in_rope", 0.0)
            d = vals.get("cohen_d_common_scale", None)
            text = (
                f"mean = {vals['ttc50_mean']:.1f} ms\n"
                f"95% HDI: [{lo:.1f}, {hi:.1f}]\n"
                f"P(TTC50>0) = {prob_pos:.3f}\n"
                f"P(in ROPE) = {prob_rope:.3f}"
            )
            if d is not None:
                text += f"\nCohen's d = {d:.2f}"
            ax.text(0.97, 0.95, text, transform=ax.transAxes, fontsize=7,
                    verticalalignment="top", horizontalalignment="right",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_xlabel("TTC50 (ms)")
        ax.set_ylabel("Density")
        ax.set_title(f"TTC50: {key}")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_variance_reduction(
    var_red: Dict[str, Any],
    bi_conditions: List[str],
    output_path: Path,
) -> None:
    """Plot variance reduction with 95% HDI error bars."""
    conditions = []
    reductions = []
    ci_lo = []
    ci_hi = []

    for cond in bi_conditions:
        key = f"{cond}_variance_reduction"
        hdi_key = f"{cond}_variance_reduction_hdi_95"
        if key in var_red:
            conditions.append(cond)
            reductions.append(var_red[key])
            if hdi_key in var_red:
                ci_lo.append(var_red[hdi_key][0])
                ci_hi.append(var_red[hdi_key][1])
            else:
                ci_lo.append(var_red[key])
                ci_hi.append(var_red[key])

    if not conditions:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(conditions) * 1.2), 5))
    x = np.arange(len(conditions))
    yerr_lo = [r - lo for r, lo in zip(reductions, ci_lo)]
    yerr_hi = [hi - r for r, hi in zip(reductions, ci_hi)]

    colours = []
    for cond in conditions:
        colours.append("#00468B" if reductions[conditions.index(cond)] >= 0 else "#ED0000")

    ax.bar(x, reductions, color=colours, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.errorbar(x, reductions, yerr=[yerr_lo, yerr_hi], fmt="none", ecolor="black",
                capsize=4, linewidth=1.5)
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("P(Escape) Variance Reduction\n(relative to visual-only)")
    ax.set_title("Bayesian Optimal Integration: P(Escape) Variance Reduction\n(95% HDI error bars)")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# Summary Export
# ══════════════════════════════════════════════════════════════════════


def _extract_delay_from_condition(cond_name: str) -> float | None:
    """Extract delay_sec from a bimodal condition name.

    Condition names produced by ``_classify_conditions`` when delay info is
    available follow the pattern ``looming_wind_d0.613_500``.  Returns
    ``None`` for unimodal conditions or when no delay is encoded.
    """
    import re
    m = re.match(r"looming_wind_d([\d.]+)_", cond_name)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _make_json_serializable(obj: Any) -> Any:
    """Recursively convert numpy types for JSON export."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def generate_summary(
    trace: az.InferenceData,
    bi_conditions: List[str],
    uni_conditions: List[str],
    ttc50_differences: Dict,
    variance_reduction: Dict,
    convergence: Dict[str, Any],
    output_path: Path,
    n_chains: int = N_CHAINS,
    n_draws: int = N_DRAWS,
    n_tune: int = N_TUNE,
    ppc_results: Dict[str, Any] | None = None,
) -> Dict:
    """Generate JSON summary with full diagnostic metrics.

    Includes actual R-hat and ESS values (not just booleans), ROPE-based
    hypothesis tests, common-scale effect sizes, variance reduction on
    the probability scale, and delay-stratified posterior estimates.
    """
    summary: Dict[str, Any] = {
        "model_info": {
            "n_chains": n_chains,
            "n_draws": n_draws,
            "n_tune": n_tune,
            "target_accept": TARGET_ACCEPT,
            "n_bimodal_conditions": len(bi_conditions),
            "n_unimodal_conditions": len(uni_conditions),
            "bimodal_conditions": bi_conditions,
            "unimodal_conditions": uni_conditions,
        },
        "convergence": {
            "converged": convergence["converged"],
            "rhat_ok": convergence["rhat_ok"],
            "ess_ok": convergence["ess_ok"],
            "rhat_max": convergence["rhat_max"],
            "ess_bulk_min": convergence["ess_bulk_min"],
            "ess_tail_min": convergence["ess_tail_min"],
        },
        "posterior_estimates": {},
        "hypothesis_tests": ttc50_differences,
        "variance_reduction": variance_reduction,
    }

    # Bimodal posterior estimates
    ttc50_post = get_ttc50_posterior(trace, bi_conditions)
    k_post = get_k_posterior(trace, bi_conditions)
    for i, cond in enumerate(bi_conditions):
        t50 = ttc50_post[:, :, i].flatten()
        kk = k_post[:, :, i].flatten()
        summary["posterior_estimates"][cond] = {
            "ttc50": {
                "mean": float(t50.mean()),
                "median": float(np.median(t50)),
                "std": float(t50.std()),
                "hdi_95": list(compute_hdi(t50)),
            },
            "k": {
                "mean": float(kk.mean()),
                "median": float(np.median(kk)),
                "std": float(kk.std()),
                "hdi_95": list(compute_hdi(kk)),
            },
        }

    # Unimodal posterior estimates
    p_baseline_post = get_p_baseline_posterior(trace, uni_conditions)
    for cond in uni_conditions:
        if cond in p_baseline_post:
            pb = p_baseline_post[cond]
            summary["posterior_estimates"][cond] = {
                "p_escape": {
                    "mean": float(pb.mean()),
                    "median": float(np.median(pb)),
                    "std": float(pb.std()),
                    "hdi_95": list(compute_hdi(pb)),
                },
            }

    # Delay-stratified summary (when delay_sec was encoded in condition names)
    delay_groups: Dict[str, List[str]] = {}
    for cond in bi_conditions:
        delay = _extract_delay_from_condition(cond)
        if delay is not None:
            delay_groups.setdefault(f"delay_{delay:.3f}", []).append(cond)
    if delay_groups:
        summary["delay_analysis"] = {}
        for delay_key, conds in sorted(delay_groups.items()):
            delay_entry: Dict[str, Any] = {
                "conditions": conds,
                "n_conditions": len(conds),
            }
            for cond in conds:
                if cond in summary["posterior_estimates"]:
                    delay_entry[cond] = summary["posterior_estimates"][cond]
            summary["delay_analysis"][delay_key] = delay_entry

    if ppc_results:
        summary["posterior_predictive"] = ppc_results

    summary = _make_json_serializable(summary)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ══════════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════


def run_mcmc_analysis(
    input_dir: str | Path,
    output_dir: str | Path,
    binary_mode: str = "escape_only",
    seed: int = RANDOM_SEED,
    n_chains: int = N_CHAINS,
    n_draws: int = N_DRAWS,
    n_tune: int = N_TUNE,
    plot: bool = True,
    plot_format: str = "svg",
) -> Dict:
    """
    Complete MCMC analysis pipeline.

    Parameters
    ----------
    input_dir : path
        Directory containing session CSV files.
    output_dir : path
        Directory for output files.
    binary_mode : str
        ``"escape_only"`` or ``"escape_prewalk"``.
    seed : int
        Random seed for reproducibility.
    n_chains, n_draws, n_tune : int
        MCMC sampling parameters.
    plot : bool
        Whether to generate plots.
    plot_format : str
        Output format for plots (``"svg"`` or ``"png"``).

    Returns
    -------
    dict
        Analysis summary (also written to ``mcmc_summary.json``).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _apply_publication_style()

    log.info("=" * 60)
    log.info("MCMC Bayesian Psychophysics Analysis")
    log.info("=" * 60)

    # Step 1: Load and classify
    log.info("Loading data from %s", input_dir)
    df = load_and_classify(input_dir)
    log.info("Loaded %d frame-level rows", len(df))

    # Step 2: Prepare MCMC data
    ttc, escape, conditions, condition_idx = prepare_mcmc_data(df, binary_mode)

    # Step 3: Split bimodal vs unimodal
    (
        bi_ttc, bi_escape, bi_conditions, bi_condition_idx,
        uni_escape, uni_conditions, uni_condition_idx,
    ) = split_bimodal_unimodal(ttc, escape, conditions, condition_idx)

    # Step 4: Build and run model
    model = build_psychometric_model(
        bi_ttc, bi_escape, bi_condition_idx, bi_conditions,
        uni_escape, uni_condition_idx, uni_conditions,
    )
    trace = run_mcmc(model, random_seed=seed, n_chains=n_chains, n_draws=n_draws, n_tune=n_tune)

    # Step 5: Convergence diagnostics
    convergence = check_convergence(trace)
    if not convergence["converged"]:
        log.warning("CONVERGENCE WARNING: Results may be unreliable. "
                     "Consider increasing n_chains or n_tune.")

    # Step 6: Statistical analyses
    ttc50_diff = analyze_ttc50_differences(trace, bi_conditions, uni_conditions)
    var_red = compute_variance_reduction(trace, bi_conditions, uni_conditions)

    # Step 7: Posterior predictive check
    ppc_results = None
    if len(bi_conditions) > 0:
        try:
            ppc_results = posterior_predictive_check(
                trace, model, bi_escape, uni_escape,
                bi_conditions, uni_conditions,
                bi_condition_idx, uni_condition_idx,
            )
        except Exception as exc:
            log.warning("Posterior predictive check failed: %s", exc)

    # Step 8: Plots
    if plot:
        log.info("Generating plots (%s)...", plot_format)
        plot_posterior_traces(
            trace, bi_conditions, uni_conditions,
            output_dir / f"posterior_traces.{plot_format}",
        )
        plot_psychometric_curves(
            trace, bi_ttc, bi_escape, bi_conditions, bi_condition_idx,
            uni_conditions, uni_escape, uni_condition_idx,
            output_dir / f"psychometric_curves.{plot_format}",
        )
        plot_posterior_distributions(
            trace, bi_conditions, uni_conditions,
            output_dir / f"posterior_distributions.{plot_format}",
        )
        if ttc50_diff:
            plot_ttc50_differences(
                ttc50_diff, bi_conditions, trace,
                output_dir / f"ttc50_differences.{plot_format}",
            )
        if var_red:
            plot_variance_reduction(
                var_red, bi_conditions,
                output_dir / f"variance_reduction.{plot_format}",
            )

    # Step 9: Summary
    summary = generate_summary(
        trace, bi_conditions, uni_conditions,
        ttc50_diff, var_red, convergence,
        output_dir / "mcmc_summary.json",
        n_chains=n_chains, n_draws=n_draws, n_tune=n_tune,
        ppc_results=ppc_results,
    )

    log.info("=" * 60)
    log.info("Analysis complete! Results in %s", output_dir)
    log.info("=" * 60)

    return summary
