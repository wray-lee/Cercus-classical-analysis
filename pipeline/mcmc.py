"""
Cercus Framework — MCMC Bayesian Psychophysics Analysis
========================================================
Bayesian inference on multisensory integration using PyMC.

Model architecture:
- **Bimodal (looming_wind_*)**: P(Escape) = sigmoid(k * (TTC - TTC50))
  with data-informed priors on k and TTC50.
- **Unimodal (visual_only, wind_only)**: P(Escape) = p_baseline
  with uniform Beta(1, 1) prior.
  visual_only receives a dynamic KM-based CDF overlay in the rendering layer.

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
import re
from pathlib import Path
from typing import Any

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from lifelines import KaplanMeierFitter
from scipy.interpolate import make_interp_spline

from .classifier import label_trials
from .constants import NPG_PALETTE, _apply_publication_style
from .io import load_and_concat_sessions, scan_and_pair_sessions
from .kinematics import preprocess

# ── JAX / numpyro CPU multi-device setup (must run before JAX initializes) ──
try:
    import numpyro
    numpyro.set_host_device_count(8)
except ImportError:
    pass

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# MCMC Sampling Parameters (defaults; overridable via CLI)
# ──────────────────────────────────────────────────────────────────────

N_CHAINS: int = 8
N_DRAWS: int = 2000
N_TUNE: int = 3000
TARGET_ACCEPT: float = 0.9
RANDOM_SEED: int = 42

MAX_TTC_BINS: int = 10

# ── Physical Stimulus Parameters for Angle Mapping ──
# l_v_ratio (ms) = (Object Half-Size / Approach Speed).
# For example, a 10cm radius object approaching at 50cm/s -> l/v = 0.2s = 200ms.
L_V_RATIO_DEFAULT: float = 120.0


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
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
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

    # 强制丢弃所有未能被路由引擎识别的残缺/废弃试验
    valid = valid[valid["stim_condition"] != "unknown"].copy()

    if binary_mode == "escape_only":
        valid["escape_binary"] = (valid["response_type"] == "Escape").astype(int)
    elif binary_mode == "escape_prewalk":
        valid["escape_binary"] = valid["response_type"].isin(["Escape", "PreWalk"]).astype(int)
    else:
        raise ValueError(f"Unknown binary_mode: {binary_mode}")

    valid["effective_ttc_ms"] = _compute_effective_ttc(valid)

    # Only looming_wind_* conditions require valid TTC for sigmoid fitting;
    # unimodal (visual_only, wind_only) pass unconditionally.
    is_curve_fitted = valid["stim_condition"].str.startswith("looming_wind_")
    keep = (~is_curve_fitted) | (is_curve_fitted & valid["effective_ttc_ms"].notna())
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

    # 强制清理空白字符并统一小写
    type_cleaned = df["type"].astype(str).str.strip().str.lower()
    conditions[type_cleaned.isin(["baseline_visual", "visual_only", "visual"])] = "visual_only"
    conditions[type_cleaned.isin(["baseline_wind", "wind_only", "wind"])] = "wind_only"

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
      the MCMC sampling graph; modelled as separate intercepts via
      ``p_baseline``).  visual_only receives a dynamic KM-based CDF
      overlay in the rendering layer, not here.
    - **looming_wind_***: TTC = ``target_ttc_ms``.
    """
    ttc = pd.Series(np.nan, index=df.index)
    ttc_ms = (
        pd.to_numeric(df["target_ttc_ms"], errors="coerce")
        if "target_ttc_ms" in df.columns
        else pd.Series(np.nan, index=df.index)
    )

    # Only bimodal conditions have meaningful TTC for MCMC sampling
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
    conditions: list[str],
    condition_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Split data into bimodal (sigmoid) and unimodal (static baseline) groups.

    - **Bimodal** (sigmoid): ``looming_wind_*``
      — fitted with ``sigmoid(k * (TTC - TTC50))``.
    - **Unimodal** (static baseline): ``visual_only``, ``wind_only``
      — modelled as ``p_baseline`` (constant, independent of TTC).
      visual_only receives a dynamic KM-based CDF overlay in the rendering
      layer; the MCMC model itself computes a flat escape rate.

    Returns
    -------
    bi_ttc, bi_escape, bi_conditions, bi_condition_idx,
    uni_escape, uni_conditions, uni_condition_idx
    """
    bi_conditions = sorted([c for c in conditions if c.startswith("looming_wind_")])
    uni_conditions = sorted([c for c in conditions if c not in bi_conditions])

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
    bi_condition_names: list[str],
    uni_escape: np.ndarray | None = None,
    uni_condition_idx: np.ndarray | None = None,
    uni_condition_names: list[str] | None = None,
) -> pm.Model:
    """
    Build Bayesian psychometric model with separated bimodal/unimodal structure.

    Bimodal (looming_wind_*) conditions:
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
        import jax
        import numpyro  # noqa: F401

        n_devices = jax.device_count()
        sampler = "numpyro"
        log.info("Using numpyro (JAX) sampler: cpu, %d device(s)", n_devices)
    except ImportError:
        sampler = "pymc"
        n_devices = 0
        log.info("numpyro not found, using default PyMC sampler")

    with model:
        trace = pm.sample(
            draws=n_draws,
            chains=n_chains,
            cores=n_chains,
            tune=n_tune,
            target_accept=TARGET_ACCEPT,
            random_seed=random_seed,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": False},
            nuts_sampler=sampler,
        )

    return trace


def check_convergence(trace: az.InferenceData) -> dict[str, Any]:
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
    bi_conditions: list[str],
) -> np.ndarray:
    """Extract TTC50 posterior samples as (n_chains, n_draws, n_bi_conditions)."""
    return trace.posterior["ttc50_named"].values


def get_k_posterior(
    trace: az.InferenceData,
    bi_conditions: list[str],
) -> np.ndarray:
    """Extract k posterior samples as (n_chains, n_draws, n_bi_conditions)."""
    return trace.posterior["k_named"].values


def get_p_baseline_posterior(
    trace: az.InferenceData,
    uni_conditions: list[str],
) -> dict[str, np.ndarray]:
    """Extract p_baseline posteriors per unimodal condition."""
    if "p_baseline" not in trace.posterior:
        return {}
    pb = trace.posterior["p_baseline"].values  # (chains, draws, n_uni)
    return {c: pb[:, :, i].flatten() for i, c in enumerate(uni_conditions)}


# ══════════════════════════════════════════════════════════════════════
# Statistical Analysis
# ══════════════════════════════════════════════════════════════════════


def compute_hdi(samples: np.ndarray, credible_mass: float = 0.95) -> tuple[float, float]:
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
    bi_conditions: list[str],
    uni_conditions: list[str],
    rope_ttc50: tuple[float, float] = (-50.0, 50.0),
) -> dict[str, Any]:
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

    results: dict[str, Any] = {}

    for i, cond in enumerate(bi_conditions):
        cond_post = ttc50_post[:, :, i].flatten()
        key = f"visual_only_vs_{cond}"

        hdi_low, hdi_high = compute_hdi(cond_post)
        prob_above_zero = float((cond_post > 0).mean())
        prob_in_rope = float(
            ((cond_post >= rope_ttc50[0]) & (cond_post <= rope_ttc50[1])).mean()
        )

        entry: dict[str, Any] = {
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
    bi_conditions: list[str],
    uni_conditions: list[str],
    n_bootstrap: int = 2000,
) -> dict[str, Any]:
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

    results: dict[str, Any] = {}

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
    bi_conditions: list[str],
    uni_conditions: list[str],
    bi_condition_idx: np.ndarray,
    uni_condition_idx: np.ndarray,
) -> dict[str, Any]:
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

    results: dict[str, Any] = {}

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

        calibration: dict[str, Any] = {}
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
    bi_conditions: list[str],
    uni_conditions: list[str],
) -> dict[str, str]:
    """Generate NPG-based discrete colour palette.

    - visual_only -> Muted Teal (#5B7B8A) — dynamic KM curve in psychometric plot
    - wind_only -> NPG Salmon (#F39B7F) — static baseline
    - looming_wind_* -> cycles through NPG palette; brightness-stepped
      when count exceeds palette size.
    """
    palette: dict[str, str] = {}
    for cond in uni_conditions:
        if cond == "visual_only":
            palette[cond] = "#5B7B8A"   # Muted Teal (dynamic KM curve)
        elif cond == "wind_only":
            palette[cond] = "#F39B7F"   # NPG Salmon
        else:
            palette[cond] = "#8491B4"   # NPG Slate Blue fallback

    n_presets = len(NPG_PALETTE)
    for idx, cond in enumerate(bi_conditions):
        if idx < n_presets:
            palette[cond] = NPG_PALETTE[idx]
        else:
            # Brightness step-down for overflow conditions
            base = NPG_PALETTE[idx % n_presets]
            step = (idx // n_presets) + 1
            r = int(base[1:3], 16)
            g = int(base[3:5], 16)
            b = int(base[5:7], 16)
            factor = max(0.4, 1.0 - 0.2 * step)
            r, g, b = int(r * factor), int(g * factor), int(b * factor)
            palette[cond] = f"#{r:02x}{g:02x}{b:02x}"

    return palette


# ── Chain trace colours (NPG-derived maximum mutual contrast) ──────

_CHAIN_COLOURS: list[str] = [
    "#E64B35",   # NPG Red
    "#3C5488",   # NPG Navy
    "#00A087",   # NPG Teal
    "#F39B7F",   # NPG Salmon
]


def plot_posterior_traces(
    trace: az.InferenceData,
    bi_conditions: list[str],
    uni_conditions: list[str],
    output_path: Path,
) -> None:
    """Generate posterior trace plots for MCMC diagnostics.

    Each chain receives a fixed high-contrast colour for instant visual
    separation.  Legend placed horizontally above the figure.
    """
    _apply_publication_style()

    panels: list[tuple[str, str, str]] = []
    for cond in bi_conditions:
        panels.append(("ttc50_named", cond, f"{cond}\nTTC50 (ms)"))
        panels.append(("k_named", cond, "k (slope)"))
    for cond in uni_conditions:
        panels.append(("p_baseline", cond, f"{cond}\np_escape"))

    n = len(panels)
    if n == 0:
        return

    n_chains = trace.posterior.sizes["chain"]
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n), squeeze=False)
    axes = axes.flatten()

    for ax, (param, cond, ylabel) in zip(axes, panels):
        if param in trace.posterior:
            try:
                for chain in range(n_chains):
                    samples = trace.posterior[param].sel(
                        chain=chain,
                        **{trace.posterior[param].dims[-1]: cond},
                    ).values
                    ax.plot(
                        samples,
                        alpha=0.8,
                        linewidth=0.8,
                        color=_CHAIN_COLOURS[chain % len(_CHAIN_COLOURS)],
                        label=f"Chain {chain}",
                    )
            except (KeyError, ValueError):
                for chain in range(n_chains):
                    samples = trace.posterior[param].sel(chain=chain).values.flatten()
                    ax.plot(
                        samples,
                        alpha=0.8,
                        linewidth=0.8,
                        color=_CHAIN_COLOURS[chain % len(_CHAIN_COLOURS)],
                        label=f"Chain {chain}",
                    )

        ax.set_ylabel(ylabel)
        ax.set_xlabel("Sample")

    # Single legend above the entire figure, horizontal
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=n_chains,
        fontsize=11,
        frameon=False,
    )
    # Remove per-axes legends
    for ax in axes:
        ax.legend_.remove() if ax.legend_ else None

    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_psychometric_curves(
    trace: az.InferenceData,
    bi_ttc: np.ndarray,
    bi_escape: np.ndarray,
    bi_conditions: list[str],
    bi_condition_idx: np.ndarray,
    uni_conditions: list[str],
    uni_escape: np.ndarray | None,
    uni_condition_idx: np.ndarray | None,
    output_path: Path,
    true_trial_start: float = -5000.0,
    survival_df: pd.DataFrame | None = None,
) -> None:
    """Generate psychometric curves (median posterior only, no shading).

    Publication-standard: sharp lines, white-edged data markers,
    legend outside right, stimulus onset marker, response window shading.

    For **visual_only**, instead of a horizontal p_baseline line, the plot
    renders a dynamic Kaplan-Meier cumulative escape curve extracted from
    ``survival_df``.  The KM curve is fitted on positive-shifted durations
    (to satisfy lifelines' strict positivity requirement) and then shifted
    back to the TTC coordinate system, sharing the same physical anchor
    (TTC=0 = collision) as the bimodal sigmoid curves.

    For **wind_only**, a horizontal dashed line at the MCMC p_baseline
    posterior median is drawn (pure mechanical trigger, no temporal
    dimension).
    """
    _apply_publication_style()
    palette = _generate_color_palette(bi_conditions, uni_conditions)

    # ── Dynamic window from true trial start ──
    ttc_lo = true_trial_start
    ttc_hi = 1000.0
    ttc_grid = np.linspace(ttc_lo, ttc_hi, 500)

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

            median_curve = np.median(preds, axis=0)

            ax.plot(ttc_grid, median_curve, color=color, linewidth=2.5, label=cond)

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
                ax.errorbar(
                    mean_ttc, mean_esc, yerr=sem,
                    fmt="o", color=color,
                    markersize=7, capsize=3, linewidth=1.0,
                    markeredgecolor="white", markeredgewidth=1.0,
                )
            except (ValueError, KeyError):
                ax.scatter(
                    cond_ttc, cond_esc,
                    color=color, alpha=0.3, s=25,
                    edgecolors="white", linewidths=0.5,
                )

    # ── Unimodal conditions ──
    p_baseline_post = get_p_baseline_posterior(trace, uni_conditions)

    # ── visual_only: dynamic Kaplan-Meier CDF interpolated onto ttc_grid ──
    if "visual_only" in uni_conditions:
        color = palette["visual_only"]
        if survival_df is not None:
            vis_mask = survival_df["stim_condition"] == "visual_only"
            df_vis = survival_df.loc[vis_mask]
            if len(df_vis) >= 2:
                global_min_ttc = min(
                    float(df_vis["TTC_at_event"].min()), true_trial_start
                )
                time_shift = abs(global_min_ttc) + 10.0

                anchor = pd.DataFrame([{
                    "TTC_at_event": true_trial_start,
                    "Event_Observed": 0,
                    "stim_condition": "visual_only",
                }])
                df_vis = pd.concat([df_vis, anchor], ignore_index=True)

                positive_durations = df_vis["TTC_at_event"].values + time_shift
                assert np.all(positive_durations > 0), (
                    f"[visual_only] shift failed: min={positive_durations.min():.1f}"
                )

                kmf = KaplanMeierFitter()
                kmf.fit(
                    durations=positive_durations,
                    event_observed=df_vis["Event_Observed"].values,
                    label="visual_only",
                )
                x_ttc = kmf.timeline - time_shift
                y_cdf = 1.0 - kmf.survival_function_.values.flatten()

                if len(x_ttc) >= 4:
                    interp_fn = make_interp_spline(x_ttc, y_cdf, k=3)
                    y_grid = interp_fn(ttc_grid)
                    y_grid = np.clip(y_grid, 0.0, 1.0)
                else:
                    y_grid = np.interp(ttc_grid, x_ttc, y_cdf)

                ax.plot(
                    ttc_grid, y_grid,
                    color=color, linewidth=2.5, linestyle="--",
                    label="visual_only (dynamic)",
                )
            else:
                log.warning("visual_only: insufficient survival data (%d rows), "
                            "falling back to p_baseline", len(df_vis))
                _draw_static_baseline(ax, p_baseline_post, "visual_only", color)
        else:
            _draw_static_baseline(ax, p_baseline_post, "visual_only", color)

    # ── wind_only (and any other residual unimodal): static baseline ──
    for cond in uni_conditions:
        if cond == "visual_only":
            continue
        color = palette[cond]
        _draw_static_baseline(ax, p_baseline_post, cond, color)

    # ── Canvas purge: force white background ──
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ── Response window highlight ──
    ax.axvspan(-1000, 500, color="#FFE599", alpha=0.8, zorder=0)

    # ── Stimulus onset marker ──
    ax.axvline(ttc_lo, color="#94A3B8", linestyle="--", alpha=0.8, linewidth=1.2, zorder=0)

    ax.axhline(0.5, color="grey", linestyle=":", alpha=0.4, linewidth=0.75)
    ax.axvline(0, color="grey", linestyle=":", alpha=0.4, linewidth=0.75)
    ax.set_xlabel("Time-to-Collision (ms)\n(negative = wind before collision)")
    ax.set_ylabel("P(Escape)")
    ax.set_title("Psychometric Functions: Multisensory Integration")
    # ── Clamp viewport to dynamic data window ──
    ax.set_xlim(ttc_lo, ttc_hi)
    ax.set_ylim(-0.05, 1.05)

    # Optional faint horizontal grid
    ax.yaxis.grid(True, linestyle="--", alpha=0.15)
    ax.set_axisbelow(True)

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _draw_static_baseline(
    ax: plt.Axes,
    p_baseline_post: dict[str, np.ndarray],
    cond: str,
    color: str,
) -> None:
    """Draw a horizontal dashed line for a static unimodal baseline."""
    if cond in p_baseline_post:
        pb = p_baseline_post[cond]
        rate = float(np.median(pb))
        lo, hi = compute_hdi(pb)
        ax.axhline(
            rate, color=color, linewidth=1.5, linestyle="--",
            label=f"{cond} ({rate:.2f} [{lo:.2f}, {hi:.2f}])",
        )


def plot_posterior_distributions(
    trace: az.InferenceData,
    bi_conditions: list[str],
    uni_conditions: list[str],
    output_path: Path,
) -> None:
    """Plot posterior distributions of TTC50, k, and p_baseline.

    Cell-standard: solid-edge histograms, black HDI lines, legend outside.
    """
    _apply_publication_style()
    palette = _generate_color_palette(bi_conditions, uni_conditions)

    n_panels = 2 + (1 if uni_conditions else 0)
    fig, axes = plt.subplots(n_panels, 1, figsize=(8, 3.5 * n_panels))
    if n_panels == 1:
        axes = [axes]

    # TTC50
    ax = axes[0]
    for i, cond in enumerate(bi_conditions):
        s = get_ttc50_posterior(trace, bi_conditions)[:, :, i].flatten()
        ax.hist(
            s, bins=50, alpha=0.7, density=True, label=cond,
            color=palette[cond], edgecolor="black", linewidth=0.5,
        )
        lo, hi = compute_hdi(s)
        ax.axvline(lo, color="black", linestyle="--", linewidth=1.2)
        ax.axvline(hi, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("TTC50 (ms)")
    ax.set_ylabel("Density")
    ax.set_title("Posterior: TTC50 (PSE)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)

    # k
    ax = axes[1]
    for i, cond in enumerate(bi_conditions):
        s = get_k_posterior(trace, bi_conditions)[:, :, i].flatten()
        ax.hist(
            s, bins=50, alpha=0.7, density=True, label=cond,
            color=palette[cond], edgecolor="black", linewidth=0.5,
        )
    ax.set_xlabel("k (slope)")
    ax.set_ylabel("Density")
    ax.set_title("Posterior: Slope (k)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)

    # p_baseline
    if uni_conditions:
        ax = axes[2]
        p_baseline_post = get_p_baseline_posterior(trace, uni_conditions)
        for cond in uni_conditions:
            if cond in p_baseline_post:
                s = p_baseline_post[cond]
                ax.hist(
                    s, bins=50, alpha=0.7, density=True, label=cond,
                    color=palette[cond], edgecolor="black", linewidth=0.5,
                )
                lo, hi = compute_hdi(s)
                ax.axvline(lo, color="black", linestyle="--", linewidth=1.2)
                ax.axvline(hi, color="black", linestyle="--", linewidth=1.2)
        ax.set_xlabel("P(Escape)")
        ax.set_ylabel("Density")
        ax.set_title("Posterior: Unimodal Baseline Escape Rate")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ttc50_differences(
    ttc50_diff: dict[str, Any],
    bi_conditions: list[str],
    trace: az.InferenceData,
    output_path: Path,
) -> None:
    """Plot posterior distributions of TTC50 with ROPE-based inference.

    Cell-standard: heavy-colour histogram, grey ROPE band, clean text box
    with black border (no shadow), legend outside.
    """
    _apply_publication_style()

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
            ax.hist(
                samples, bins=60, alpha=0.7, density=True,
                color="#3C5488", edgecolor="black", linewidth=0.5,
            )
            lo, hi = vals["ttc50_hdi_95"]
            ax.axvline(lo, color="black", linestyle="--", linewidth=1.2)
            ax.axvline(hi, color="black", linestyle="--", linewidth=1.2)
            ax.axvline(0, color="grey", linestyle=":", linewidth=0.75)

            # ROPE shading — neutral grey
            rope = vals.get("rope", [-50.0, 50.0])
            ax.axvspan(rope[0], rope[1], color="#E0E0E0", alpha=0.4)

            # Annotation text box — outside right, white bg, black border
            prob_pos = vals.get("prob_ttc50_positive", 0.5)
            prob_rope = vals.get("prob_ttc50_in_rope", 0.0)
            d = vals.get("cohen_d_common_scale", None)
            lines = [
                f"mean    = {vals['ttc50_mean']:>8.1f} ms",
                f"95% HDI = [{lo:>7.1f}, {hi:>7.1f}]",
                f"P(>0)   = {prob_pos:>8.3f}",
                f"P(ROPE) = {prob_rope:>8.3f}",
            ]
            if d is not None:
                lines.append(f"Cohen d = {d:>8.2f}")
            text = "\n".join(lines)
            ax.text(
                1.02, 0.95, text,
                transform=ax.transAxes, fontsize=10,
                verticalalignment="top", horizontalalignment="left",
                fontfamily="monospace",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor="white",
                    edgecolor="black",
                    linewidth=0.8,
                ),
            )

        ax.set_xlabel("TTC50 (ms)")
        ax.set_ylabel("Density")
        ax.set_title(f"TTC50: {key}")

    plt.tight_layout()
    fig.subplots_adjust(right=0.75)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_variance_reduction(
    var_red: dict[str, Any],
    bi_conditions: list[str],
    output_path: Path,
) -> None:
    """Plot variance reduction with 95% HDI error bars.

    Cell-standard: solid-fill bars with black edges, heavy error bars
    with square caps, legend outside.
    """
    _apply_publication_style()

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

    colours = [
        "#3C5488" if reductions[i] >= 0 else "#E64B35"
        for i in range(len(conditions))
    ]

    ax.bar(
        x, reductions,
        color=colours, alpha=0.85,
        edgecolor="black", linewidth=1.5,
    )
    ax.errorbar(
        x, reductions, yerr=[yerr_lo, yerr_hi],
        fmt="none", ecolor="black",
        linewidth=2, capsize=6, capthick=2,
    )
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("P(Escape) Variance Reduction\n(relative to visual-only)")
    ax.set_title("Bayesian Optimal Integration: P(Escape) Variance Reduction\n(95% HDI error bars)")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# Kaplan-Meier Survival Analysis
# ══════════════════════════════════════════════════════════════════════


def prepare_survival_data(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Prepare trial-level survival data for Kaplan-Meier analysis.

    Each trial yields a single TTC-aligned time value (``TTC_at_event``):
    the ``latency_ms`` value at escape onset (event) or the ``t_rel``
    value at the last observed frame (censored).  This value is in the
    same coordinate system as the MCMC model's TTC axis (negative =
    before collision, 0 = collision).

    Returns
    -------
    result : DataFrame
        Columns: ``TTC_at_event``, ``Event_Observed``, ``stim_condition``.
    true_trial_start : float
        Earliest frame-level ``t_rel`` across all valid trials — the true
        physical stimulus onset time in TTC coordinates.
    """
    if "subject" in df.columns:
        df = df.copy()
        df["unique_trial_id"] = df["subject"].astype(str) + "_" + df["global_trial_id"].astype(str)
    else:
        df = df.copy()
        df["unique_trial_id"] = df["global_trial_id"].astype(str)

    # ── STEP 1: Global Data Purge — classify and drop 'unknown' ──
    trial_level = df.groupby("unique_trial_id").first()
    stim_conditions = _classify_conditions(trial_level)

    valid_mask = stim_conditions != "unknown"
    stim_conditions = stim_conditions[valid_mask]
    valid_trial_ids = set(stim_conditions.index)

    n_total = df["unique_trial_id"].nunique()
    log.info("Survival data purge: %d valid / %d total trials (dropped 'unknown')",
             len(valid_trial_ids), n_total)

    # Locate the TTC-aligned time column (produced by kinematics.preprocess).
    time_col = None
    if "t_rel" in df.columns:
        time_col = "t_rel"
    else:
        for candidate in ("ttc_ms", "time_to_collision_ms", "ttc", "frame_ttc"):
            if candidate in df.columns:
                time_col = candidate
                break

    if time_col is None:
        raise ValueError("No TTC-aligned time column found in DataFrame")

    # ── Extract true trial start from raw frame-level data ──
    # The earliest t_rel across ALL frames in valid trials = stimulus onset
    true_trial_start = float(df.loc[df["unique_trial_id"].isin(valid_trial_ids), time_col].min())
    log.info("True trial start (stimulus onset): %.1f ms", true_trial_start)

    trials = df.groupby("unique_trial_id")
    records = []

    # ── VERIFICATION: log first 10 raw (TTC, event) pairs ──
    _debug_count = 0

    for trial_id, grp in trials:
        if trial_id not in valid_trial_ids:
            continue

        stim = stim_conditions.loc[trial_id]
        response = grp.iloc[0].get("response_type", "Unknown")
        t = grp[time_col]

        if response == "Escape":
            # Use the pre-computed escape onset latency (t_rel at escape start).
            # This is set per trial by label_trials → compute_escape_latency.
            lat = grp.iloc[0].get("latency_ms", np.nan)
            if not np.isnan(lat):
                ttc_at_event = float(lat)
            else:
                # Fallback: last observed TTC (shouldn't happen for valid Escape)
                ttc_at_event = float(t.iloc[-1])
                log.warning("Escape trial %s has NaN latency_ms, using last frame t_rel=%.1f",
                            trial_id, ttc_at_event)
        else:
            # Censored: TTC at last observed frame
            ttc_at_event = float(t.iloc[-1])

        event = 1 if response == "Escape" else 0

        # ── VERIFICATION: print first 10 samples ──
        if _debug_count < 10:
            log.info("  KM sample %d: trial=%s, stim=%s, response=%s, "
                     "latency_ms=%.1f, t_last=%.1f → TTC_at_event=%.1f, event=%d",
                     _debug_count, trial_id, stim, response,
                     grp.iloc[0].get("latency_ms", np.nan), float(t.iloc[-1]),
                     ttc_at_event, event)
            _debug_count += 1

        records.append({
            "TTC_at_event": ttc_at_event,
            "Event_Observed": event,
            "stim_condition": stim,
        })

    result = pd.DataFrame(records)
    log.info("Survival data: %d trials, conditions=%s",
             len(result), sorted(result["stim_condition"].unique()))

    # ── POST-FIT SANITY CHECK ──
    escape_events = result[result["Event_Observed"] == 1]
    censored_events = result[result["Event_Observed"] == 0]

    if not escape_events.empty:
        esc_ttc = escape_events["TTC_at_event"]
        log.info("Escape events (n=%d): TTC mean=%.1f, median=%.1f, "
                 "min=%.1f, max=%.1f, std=%.1f",
                 len(esc_ttc), esc_ttc.mean(), esc_ttc.median(),
                 esc_ttc.min(), esc_ttc.max(), esc_ttc.std())
        # Guard: if >50% of escape events sit at the global min TTC,
        # the mapping is almost certainly broken (trial-start artefact).
        global_min = result["TTC_at_event"].min()
        pct_at_min = (esc_ttc <= global_min + 1).mean() * 100
        if pct_at_min > 50:
            log.error(
                "DATA ANOMALY: %.0f%% of escape events sit at TTC≈%.0f ms "
                "(global min). Escape onset mapping is likely broken — "
                "check that latency_ms is populated correctly.", pct_at_min, global_min
            )
    if not censored_events.empty:
        cen_ttc = censored_events["TTC_at_event"]
        log.info("Censored events (n=%d): TTC mean=%.1f, median=%.1f, "
                 "min=%.1f, max=%.1f",
                 len(cen_ttc), cen_ttc.mean(), cen_ttc.median(),
                 cen_ttc.min(), cen_ttc.max())

    return result, true_trial_start


def plot_kaplan_meier_cumulative(
    df: pd.DataFrame,
    bi_conditions: list[str],
    uni_conditions: list[str],
    output_path: Path,
    true_trial_start: float = -5000.0,
) -> None:
    """Plot cumulative escape probability anchored to TTC.

    Uses the **positive-shift** method to satisfy lifelines' strict
    requirement for non-negative durations while preserving the true
    TTC-aligned coordinate system:

    1. Read ``TTC_at_event`` directly (already in TTC coordinates).
    2. Compute a single global shift = |min(TTC_at_event)| + 10 ms.
    3. Fit KM on (TTC_at_event + shift)  — all values strictly positive.
    4. Shift fitted timeline back:  x_ttc = kmf.timeline - shift.

    Parameters
    ----------
    true_trial_start : float
        Earliest frame-level t_rel (stimulus onset) in TTC coordinates.
        Used to set the X-axis lower bound and inject a censored anchor
        so the KM curve renders from the true physical origin.
    """
    _apply_publication_style()
    palette = _generate_color_palette(bi_conditions, uni_conditions)

    # ── STEP 1: Assert no 'unknown' contamination ──
    unique_conds = df["stim_condition"].unique()
    assert "unknown" not in unique_conds, (
        f"Data contamination: 'unknown' found in stim_condition: {unique_conds}"
    )
    log.info("KM purge verified: %d conditions, no 'unknown'", len(unique_conds))

    # ── STEP 2: Dynamic viewport from true trial start ──
    X_MIN = true_trial_start
    X_MAX = 1000.0

    # ── STEP 3: Positive-shift alignment ──
    # Global shift to make ALL TTC_at_event values strictly positive
    # Use true_trial_start (not event min) so the anchor placeholder is included
    global_min_ttc = min(float(df["TTC_at_event"].min()), true_trial_start)
    time_shift = abs(global_min_ttc) + 10.0
    log.info("KM positive-shift: true_trial_start=%.1f, event_min=%.1f, "
             "time_shift=%.1f ms",
             true_trial_start, float(df["TTC_at_event"].min()), time_shift)

    all_conditions = sorted(unique_conds)

    # ── Inject censored anchor at true trial start ──
    # Forces KM curve to render from the physical origin (Y=0 baseline)
    anchor_rows = []
    for cond in all_conditions:
        anchor_rows.append({
            "TTC_at_event": true_trial_start,
            "Event_Observed": 0,   # censored
            "stim_condition": cond,
        })
    df = pd.concat([df, pd.DataFrame(anchor_rows)], ignore_index=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    kmf = KaplanMeierFitter()

    for cond in all_conditions:
        mask = df["stim_condition"] == cond
        if mask.sum() < 2:
            continue

        df_cond = df.loc[mask]
        ttc_raw = df_cond["TTC_at_event"].values
        events = df_cond["Event_Observed"].values

        # 3a. Shift to strictly positive domain for lifelines
        positive_durations = ttc_raw + time_shift

        # Safety: lifelines rejects non-positive values
        assert np.all(positive_durations > 0), (
            f"[{cond}] Shift failed: min={positive_durations.min():.1f}"
        )

        # 3b. Fit KM on positive-shifted durations
        kmf.fit(durations=positive_durations, event_observed=events, label=cond)

        # 3c. Extract arrays and shift timeline BACK to true TTC
        x_true_ttc = kmf.timeline - time_shift
        y_cum_escape = 1.0 - kmf.survival_function_.values.flatten()

        # ── VERIFICATION: log x_true_ttc bounds ──
        log.info("  [%s] x_true_ttc: min=%.1f, max=%.1f  (n=%d, events=%d)",
                 cond, float(x_true_ttc.min()), float(x_true_ttc.max()),
                 mask.sum(), int(events.sum()))

        color = palette.get(cond, "#7C878E")
        is_unimodal = not cond.startswith("looming_wind_")

        # ── STEP 4: Pure step rendering (NO fill_between, NO CIs) ──
        ax.step(x_true_ttc, y_cum_escape, where="post",
                label=cond, color=color,
                linestyle="--" if is_unimodal else "-",
                linewidth=2.0 if is_unimodal else 2.5)

    # ── Canvas purge: force white background ──
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ── Response window highlight ──
    ax.axvspan(-1000, 500, color="#FFE599", alpha=0.8, zorder=0)

    # ── Stimulus onset marker (mixed coordinate system) ──
    ax.axvline(true_trial_start, color="#94A3B8", linestyle="--", alpha=0.8, linewidth=1.2, zorder=0)
    # ax.text(true_trial_start + 30, 0.95, "Stimulus Onset",
    #         ha="left", va="top",
    #         fontsize=10, color="#64748B", fontweight="bold",
    #         transform=ax.get_xaxis_transform(), zorder=1)

    # ── Physical anchor lines ──
    ax.axvline(0, color="black", linestyle="--", alpha=0.5, linewidth=1.2, zorder=0)
    ax.text(0, 0.97, "Collision\n(TTC=0)", ha="center", va="top",
            fontsize=10, color="black", fontweight="bold")
    ax.axhline(0.5, color="grey", linestyle=":", alpha=0.5, linewidth=1.0, zorder=0)

    # ── Dynamic viewport from data ──
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(-0.05, 1.05)

    ax.set_xlabel("Time to Collision (ms)\n(negative = before collision)")
    ax.set_ylabel("Cumulative Escape Probability")
    ax.set_title("Kaplan-Meier: Cumulative Escape Dynamics")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# Critical Retinal Angle Mapping
# ══════════════════════════════════════════════════════════════════════


def plot_race_model_violation(
    df: pd.DataFrame,
    bi_conditions: list[str],
    uni_conditions: list[str],
    output_path: Path,
    true_trial_start: float = -5000.0,
) -> None:
    """Plot Race Model Inequality violation for multisensory integration.

    Computes empirical CDFs via Kaplan-Meier for each bimodal condition and
    compares them against Miller's Race Model Bound:
        CDF_bound(t) = min(1, CDF_visual(t) + CDF_wind(t))

    Regions where a bimodal CDF exceeds the bound are shaded red, providing
    direct mathematical evidence for super-additive Bayesian integration that
    violates the independent race model.

    Parameters
    ----------
    df : DataFrame
        Survival-level data with ``TTC_at_event``, ``Event_Observed``,
        ``stim_condition`` (from ``prepare_survival_data``).
    bi_conditions, uni_conditions : list of str
        Condition names.
    output_path : Path
        Output file path.
    true_trial_start : float
        Earliest frame-level ``t_rel`` (stimulus onset) in TTC coordinates.
    """
    _apply_publication_style()
    palette = _generate_color_palette(bi_conditions, uni_conditions)

    # ── Positive-shift to satisfy lifelines' strict positive-duration req ──
    global_min_ttc = min(float(df["TTC_at_event"].min()), true_trial_start)
    time_shift = abs(global_min_ttc) + 10.0
    log.info("Race Model plot — positive-shift: true_trial_start=%.1f, "
             "event_min=%.1f, time_shift=%.1f ms",
             true_trial_start, float(df["TTC_at_event"].min()), time_shift)

    # Inject censored anchor at true trial start per condition (Y=0 baseline)
    all_conds = sorted(df["stim_condition"].unique())
    anchor_rows = [
        {"TTC_at_event": true_trial_start, "Event_Observed": 0,
         "stim_condition": c}
        for c in all_conds
    ]
    df = pd.concat([df, pd.DataFrame(anchor_rows)], ignore_index=True)

    kmf = KaplanMeierFitter()
    cdf_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # ── Fit empirical CDFs for all bimodal conditions ──
    for cond in all_conds:
        mask = df["stim_condition"] == cond
        if mask.sum() < 2:
            continue
        df_cond = df.loc[mask]
        pos_dur = df_cond["TTC_at_event"].values + time_shift
        assert np.all(pos_dur > 0), f"[{cond}] shift failed: min={pos_dur.min():.1f}"
        kmf.fit(durations=pos_dur,
                event_observed=df_cond["Event_Observed"].values, label=cond)
        x_ttc = kmf.timeline - time_shift
        y_cdf = 1.0 - kmf.survival_function_.values.flatten()
        cdf_curves[cond] = (x_ttc, y_cdf)

    # ── Fit unimodal baselines for the race model bound ──
    uni_cdf: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cond in ("visual_only", "wind_only"):
        if cond not in cdf_curves:
            mask = df["stim_condition"] == cond
            if mask.sum() < 2:
                continue
            df_cond = df.loc[mask]
            pos_dur = df_cond["TTC_at_event"].values + time_shift
            assert np.all(pos_dur > 0)
            kmf.fit(durations=pos_dur,
                    event_observed=df_cond["Event_Observed"].values, label=cond)
            x_ttc = kmf.timeline - time_shift
            y_cdf = 1.0 - kmf.survival_function_.values.flatten()
            uni_cdf[cond] = (x_ttc, y_cdf)
        else:
            uni_cdf[cond] = cdf_curves[cond]

    # ── Miller's Race Model Bound on a shared time grid ──
    bound_x = bound_y = None
    if "visual_only" in uni_cdf and "wind_only" in uni_cdf:
        x_v, y_v = uni_cdf["visual_only"]
        x_w, y_w = uni_cdf["wind_only"]
        bound_x = np.union1d(x_v, x_w)
        cdf_v = np.interp(bound_x, x_v, y_v)
        cdf_w = np.interp(bound_x, x_w, y_w)
        bound_y = np.minimum(1.0, cdf_v + cdf_w)
    elif "visual_only" in uni_cdf:
        bound_x, bound_y = uni_cdf["visual_only"]
    elif "wind_only" in uni_cdf:
        bound_x, bound_y = uni_cdf["wind_only"]

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Response window highlight
    ax.axvspan(-1000, 500, color="#FFE599", alpha=0.8, zorder=0)

    # Empirical CDFs for bimodal conditions
    for cond in bi_conditions:
        if cond not in cdf_curves:
            continue
        x, y = cdf_curves[cond]
        color = palette.get(cond, "#7C878E")
        ax.plot(x, y, color=color, linewidth=2.5, label=cond, zorder=2)

    # Race Model Bound + violation shading
    if bound_x is not None and bound_y is not None:
        ax.plot(bound_x, bound_y, color="#333333", linewidth=2.5,
                linestyle="--", label="Race Model Bound", zorder=3)

        for cond in bi_conditions:
            if cond not in cdf_curves:
                continue
            x_bi, y_bi = cdf_curves[cond]
            x_shared = np.union1d(x_bi, bound_x)
            y_bi_interp = np.interp(x_shared, x_bi, y_bi)
            y_bound_interp = np.interp(x_shared, bound_x, bound_y)
            violation = y_bi_interp > y_bound_interp
            if np.any(violation):
                ax.fill_between(
                    x_shared, y_bound_interp, y_bi_interp,
                    where=violation, color="#E64B35", alpha=0.20,
                    interpolate=True, zorder=1,
                )

    # Physical anchor lines
    ax.axvline(0, color="black", linestyle="--", alpha=0.5, linewidth=1.2, zorder=0)
    ax.text(0, 0.97, "Collision\n(TTC=0)", ha="center", va="top",
            fontsize=10, color="black", fontweight="bold")
    ax.axhline(0.5, color="grey", linestyle=":", alpha=0.5, linewidth=1.0, zorder=0)

    ax.set_xlim(true_trial_start, 1000.0)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Time to Collision (ms)\n(negative = before collision)")
    ax.set_ylabel("Cumulative Escape Probability")
    ax.set_title("Race Model Inequality: Bayesian Integration Evidence")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_time_window_of_integration(
    var_red: dict[str, Any],
    bi_conditions: list[str],
    output_path: Path,
) -> None:
    """Plot Time-Window of Integration (TWoI) curve.

    Maps each bimodal condition's TTC offset (extracted from the condition
    name, e.g. ``looming_wind_d0.625_-119`` → SOA = −119 ms) against its
    posterior Variance Reduction, revealing the temporal profile of
    multisensory integration.

    Parameters
    ----------
    var_red : dict
        Output of ``compute_variance_reduction``.
    bi_conditions : list of str
        Bimodal condition names.
    output_path : Path
        Output file path.
    """
    _apply_publication_style()

    soa_vals: list[int] = []
    vr_vals: list[float] = []
    err_lo: list[float] = []
    err_hi: list[float] = []

    for cond in bi_conditions:
        m = re.search(r"_(-?\d+)$", cond)
        if m is None:
            continue
        soa = int(m.group(1))
        vr_key = f"{cond}_variance_reduction"
        hdi_key = f"{cond}_variance_reduction_hdi_95"
        if vr_key not in var_red:
            continue
        vr = var_red[vr_key]
        if not np.isfinite(vr):
            continue
        soa_vals.append(soa)
        vr_vals.append(float(vr))
        if hdi_key in var_red:
            err_lo.append(var_red[hdi_key][0])
            err_hi.append(var_red[hdi_key][1])
        else:
            err_lo.append(float(vr))
            err_hi.append(float(vr))

    if len(soa_vals) < 2:
        log.warning("TWoI plot skipped: fewer than 2 valid SOA points")
        return

    order = np.argsort(soa_vals)
    soa_arr = np.asarray(soa_vals)[order]
    vr_arr = np.asarray(vr_vals)[order]
    lo_arr = np.asarray(err_lo)[order]
    hi_arr = np.asarray(err_hi)[order]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Smooth curve
    try:
        n_pts = max(200, len(soa_arr) * 20)
        x_smooth = np.linspace(soa_arr[0], soa_arr[-1], n_pts)
        y_smooth = make_interp_spline(soa_arr, vr_arr, k=min(3, len(soa_arr) - 1))(x_smooth)
        ax.plot(x_smooth, y_smooth, color="#3C5488", linewidth=2.5, zorder=2)
    except Exception:
        ax.plot(soa_arr, vr_arr, color="#3C5488", linewidth=2.5, zorder=2)

    # Points with HDI error bars
    yerr = np.vstack([vr_arr - lo_arr, hi_arr - vr_arr])
    ax.errorbar(
        soa_arr, vr_arr, yerr=yerr,
        fmt="o", color="#3C5488", markersize=8,
        markeredgecolor="white", markeredgewidth=1.2,
        ecolor="black", elinewidth=1.5, capsize=5, capthick=1.5,
        zorder=3,
    )

    ax.axhline(0, color="grey", linestyle="--", linewidth=0.75, zorder=0)

    ax.set_xlabel("Stimulus Onset Asynchrony (ms)")
    ax.set_ylabel("Posterior Variance Reduction\n(relative to visual-only)")
    ax.set_title("Time-Window of Multisensory Integration")

    ax.yaxis.grid(True, linestyle="--", alpha=0.15)
    ax.set_axisbelow(True)

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
    m = re.match(r"looming_wind_d([\d.]+)_", cond_name)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


class _NumpyEncoder(json.JSONEncoder):
    """ponytail: stdlib JSONEncoder replaces recursive _make_json_serializable."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def generate_summary(
    trace: az.InferenceData,
    bi_conditions: list[str],
    uni_conditions: list[str],
    ttc50_differences: dict,
    variance_reduction: dict,
    convergence: dict[str, Any],
    output_path: Path,
    n_chains: int = N_CHAINS,
    n_draws: int = N_DRAWS,
    n_tune: int = N_TUNE,
    ppc_results: dict[str, Any] | None = None,
) -> dict:
    """Generate JSON summary with full diagnostic metrics.

    Includes actual R-hat and ESS values (not just booleans), ROPE-based
    hypothesis tests, common-scale effect sizes, variance reduction on
    the probability scale, and delay-stratified posterior estimates.
    """
    summary: dict[str, Any] = {
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
    delay_groups: dict[str, list[str]] = {}
    for cond in bi_conditions:
        delay = _extract_delay_from_condition(cond)
        if delay is not None:
            delay_groups.setdefault(f"delay_{delay:.3f}", []).append(cond)
    if delay_groups:
        summary["delay_analysis"] = {}
        for delay_key, conds in sorted(delay_groups.items()):
            delay_entry: dict[str, Any] = {
                "conditions": conds,
                "n_conditions": len(conds),
            }
            for cond in conds:
                if cond in summary["posterior_estimates"]:
                    delay_entry[cond] = summary["posterior_estimates"][cond]
            summary["delay_analysis"][delay_key] = delay_entry

    if ppc_results:
        summary["posterior_predictive"] = ppc_results

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, cls=_NumpyEncoder)

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

        # Extract true trial start once from valid trials (shared by KM & psychometric)
        try:
            survival_df, true_trial_start = prepare_survival_data(df)
        except Exception as exc:
            log.warning("Survival data preparation failed: %s", exc)
            survival_df, true_trial_start = None, -5000.0

        plot_posterior_traces(
            trace, bi_conditions, uni_conditions,
            output_dir / f"posterior_traces.{plot_format}",
        )
        plot_psychometric_curves(
            trace, bi_ttc, bi_escape, bi_conditions, bi_condition_idx,
            uni_conditions, uni_escape, uni_condition_idx,
            output_dir / f"psychometric_curves.{plot_format}",
            true_trial_start=true_trial_start,
            survival_df=survival_df,
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
        if survival_df is not None:
            plot_race_model_violation(
                survival_df, bi_conditions, uni_conditions,
                output_dir / f"race_model_violation.{plot_format}",
                true_trial_start=true_trial_start,
            )
        plot_time_window_of_integration(
            var_red, bi_conditions,
            output_dir / f"time_window_of_integration.{plot_format}",
        )
        # Kaplan-Meier cumulative escape dynamics
        if survival_df is not None:
            try:
                plot_kaplan_meier_cumulative(
                    survival_df, bi_conditions, uni_conditions,
                    output_dir / "kaplan_meier_cumulative.svg",
                    true_trial_start=true_trial_start,
                )
            except Exception as exc:
                log.warning("Kaplan-Meier plot failed: %s", exc)

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
