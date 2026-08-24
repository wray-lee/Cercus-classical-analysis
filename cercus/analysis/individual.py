"""
Cercus Framework — Individual-Level Robustness Checks
=====================================================
Cross-animal robustness of the population escape-direction estimate.

Why total trial n does not solve pseudo-replication
----------------------------------------------------
The pooled escape direction is computed over all escape trials (e.g. n = 109),
but those trials are NESTED within animals (N ≈ 20). Trials from the same
animal are not independent observations: they share that animal's idiosyncratic
anatomy, baseline state, and response bias. If one animal contributes 40
trials and another 2, a trial-level mean is silently dominated by the
high-count animal, and a trial-level Rayleigh test treats every trial as an
independent draw — inflating the effective sample size and shrinking the
confidence interval on the mean direction far below what the number of
*animals* would justify (Hurlbert 1984; Aarts, Coppens, & Coenen 2014).

The checks below therefore treat the ANIMAL as the unit of replication:
per-animal mean directions are the input to the second-order Rayleigh test
and the bootstrap resamples whole animals.

Why low-n animals cannot enter a k-sample Wallraff directly
------------------------------------------------------------
The Wallraff / Kruskal-Wallis test compares the angular *dispersion* of each
group about its own reference. For an animal with 1–3 trials the within-animal
resultant length R_k is a badly biased statistic: with a single trial R_k is
identically 1.0 (one direction has zero spread), so low-n animals appear
"perfectly concentrated" and quietly distort the pooled dispersion comparison.
A k-sample Wallraff across animals is therefore only meaningful once every
included group has a reasonably reliable dispersion estimate (n >= 5).
The primary population-level inference should come from a hierarchical /
partial-pooling model — see ``mcmc_analysis.py``, which lets each animal borrow
strength from the group posterior rather than demanding stable per-animal
point estimates.

All circular means are computed with ``circ_mean_rad``; linear means are never
used on angular data.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from cercus.config import get_analysis, get_visualization
from cercus.visualization._circstats import circ_mean_rad, rayleigh_p, wallraff_test
from cercus.visualization._core import compute_trajectory_masks
from pipeline.constants import _get_unified_side

log = logging.getLogger(__name__)

# ── Analysis thresholds (config-driven; see cercus/config/defaults/analysis.yaml) ──
_ANALYSIS = get_analysis()
MIN_TRIALS_SECOND_ORDER: int = int(_ANALYSIS.individual.min_trials_second_order)
MIN_TRIALS_WALLRAFF: int = int(_ANALYSIS.individual.min_trials_wallraff)
BOOTSTRAP_ITERATIONS: int = int(_ANALYSIS.individual.bootstrap_iterations)

# ── Response classes pooled behind `population --individual-checks` ──
# Defaults to ("Escape", "PreWalk") when the visualization toggle
# ``individual_checks_include_prewalk`` is true (so most animals reach the
# n>=3 second-order threshold); collapses to Escape-only when false. The
# column values are capitalized ("Escape"/"PreWalk") — kept as exact matches.
_INCLUDE_PREWALK: bool = bool(
    get_visualization().get("individual_checks_include_prewalk", True)
)
INDIVIDUAL_CHECK_RESPONSE_TYPES: tuple[str, ...] = (
    ("Escape", "PreWalk") if _INCLUDE_PREWALK else ("Escape",)
)


def _animal_col(df: pd.DataFrame) -> str:
    """Return the column carrying the animal identifier."""
    for col in ("animal_id", "subject_id"):
        if col in df.columns:
            return col
    raise KeyError(
        "DataFrame must contain an 'animal_id' or 'subject_id' column."
    )


def _as_angle_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-trial table with ``angle_deg`` (response trials only).

    Accepts either the raw frame-level DataFrame (recomputes per-trial angles
    from trajectories) or an already-aggregated per-trial table that carries an
    ``angle_deg`` column and no ``t_rel`` column. If a ``response_type`` column
    is present it is filtered to the individual-check response classes (by
    default Escape + PreWalk, see :data:`INDIVIDUAL_CHECK_RESPONSE_TYPES`) so
    both paths feed identical input to the downstream tests.
    """
    if "angle_deg" in df.columns and "t_rel" not in df.columns:
        if "response_type" in df.columns:
            df = df[df["response_type"].isin(INDIVIDUAL_CHECK_RESPONSE_TYPES)]
        return df
    return trial_escape_angles(df)


def trial_escape_angles(
    df: pd.DataFrame,
    response_type: str | None = None,
) -> pd.DataFrame:
    """One row per trial with its animal id and escape/response direction.

    The angle convention mirrors :func:`cercus.visualization.polar.plot_population_polar_histogram`
    (``atan2(traj_x[-1], traj_y[-1])`` after unifying the stimulus side), so the
    pooled mean recovered here is identical to the one drawn in the population
    rose plot.

    Parameters
    ----------
    df : pd.DataFrame
        Either frame-level data (rows = time samples) or a trial-level table
        that already contains an ``angle_deg`` column.
    response_type : str | None
        A single response class to keep. Defaults to *all* individual-check
        response classes (:data:`INDIVIDUAL_CHECK_RESPONSE_TYPES`).

    Returns
    -------
    pd.DataFrame
        Columns: ``<animal>``, ``trial_id``, ``angle_deg``.
    """
    animal = _animal_col(df)
    keep_types = (
        (response_type,) if response_type is not None else INDIVIDUAL_CHECK_RESPONSE_TYPES
    )
    df = df[df["response_type"].isin(keep_types)].copy()
    if df.empty:
        return pd.DataFrame(columns=[animal, "trial_id", "angle_deg"])

    # Already aggregated: per-trial angle column, no time series.
    if "angle_deg" in df.columns and "t_rel" not in df.columns:
        out = df[[animal, "angle_deg"]].copy()
        out = out.dropna(subset=["angle_deg"]).reset_index(drop=True)
        out["trial_id"] = range(len(out))
        return out[[animal, "trial_id", "angle_deg"]]

    trial_col = next(
        (c for c in ("global_trial_id", "global_trial_index") if c in df.columns),
        None,
    )
    if trial_col is None:
        raise KeyError(
            "No per-trial id column found (need 'global_trial_id' or "
            "'global_trial_index')."
        )

    rows: list[tuple] = []
    for (_animal, _trial), grp in df.groupby([animal, trial_col]):
        grp = grp.sort_values("t_rel")
        onset_ms = (
            grp["interval_onset_ms"].iloc[0]
            if "interval_onset_ms" in grp.columns
            else np.nan
        )
        offset_ms = (
            grp["interval_offset_ms"].iloc[0]
            if "interval_offset_ms" in grp.columns
            else np.nan
        )

        result = compute_trajectory_masks(grp, onset_ms, offset_ms, context="individual")
        if result is None:
            continue
        traj_x, traj_y, *_rest = result
        if traj_x is None or len(traj_x) < 2:
            continue

        # Unify stimulus side so left/right stimuli are pooled about 0°.
        if _get_unified_side(grp) == "left":
            traj_x = -traj_x

        angle_deg = float(np.degrees(np.arctan2(traj_x[-1], traj_y[-1])))
        rows.append((_animal, _trial, angle_deg))

    return pd.DataFrame(rows, columns=[animal, "trial_id", "angle_deg"])


def trial_counts_per_animal(df: pd.DataFrame) -> pd.Series:
    """Number of escape trials per animal (``Series`` indexed by animal id).

    Central to the pseudo-replication concern: the histogram of this series
    shows how the total trial n is distributed across the N animals that
    actually carry it.
    """
    angles = _as_angle_table(df)
    animal = _animal_col(angles)
    return angles.groupby(animal)["angle_deg"].size().astype(int)


def _per_animal_summary(angles: pd.DataFrame) -> pd.DataFrame:
    """Per-animal circular mean direction and resultant length.

    Uses ``circ_mean_rad`` for the mean direction; R_k is the within-animal
    resultant length (a concentration measure). Included for ALL animals with
    at least one escape trial, but note that R_k for n=1 is trivially 1.
    """
    animal = _animal_col(angles)
    rows: list[dict] = []
    for a, grp_deg in angles.groupby(animal)["angle_deg"]:
        rad = np.radians(np.asarray(grp_deg, dtype=float))
        rows.append(
            {
                "animal_id": a,
                "n_trials": int(rad.size),
                "mu_deg": float(np.degrees(circ_mean_rad(rad))),
                "R": float(np.abs(np.mean(np.exp(1j * rad)))),
            }
        )
    return pd.DataFrame(rows)


def second_order_analysis(
    df: pd.DataFrame,
    min_trials: int | None = None,
) -> dict:
    """Second-order analysis: per-animal means -> second-order Rayleigh test.

    For each animal with at least ``min_trials`` escape trials, compute its
    mean direction mu_k (circular) and resultant length R_k. Treating each
    animal as one observation (the correct replication unit — see module
    docstring), test the per-animal mean directions for a common population
    preference with a Rayleigh test on n = number of animals.

    Parameters
    ----------
    df : pd.DataFrame
        Frame-level or per-trial angle data.
    min_trials : int | None
        Minimum escape trials per animal. Defaults to the config value
        (``analysis.individual.min_trials_second_order``).

    Returns
    -------
    dict
        Keys: ``mu_deg`` (second-order mean), ``R`` (second-order resultant),
        ``p`` (second-order Rayleigh p), ``N`` (number of animals),
        ``animals`` (per-animal summary DataFrame), ``excluded`` (n animals
        dropped for low trial count).
    """
    min_trials = MIN_TRIALS_SECOND_ORDER if min_trials is None else min_trials
    angles = _as_angle_table(df)
    animal = _animal_col(angles)

    counts = trial_counts_per_animal(angles)
    eligible = counts[counts >= min_trials]
    n_excluded = int((counts < min_trials).sum())
    log.info(
        "Second-order: %d/%d animals have >=%d response trials "
        "(%d excluded: n<%d)",
        len(eligible), len(counts), min_trials, n_excluded, min_trials,
    )

    all_summary = _per_animal_summary(angles)
    animals = all_summary[all_summary["n_trials"] >= min_trials].reset_index(drop=True)

    if len(animals) < 2:
        log.warning(
            "Second-order: fewer than 2 eligible animals — test skipped.",
        )
        return {
            "mu_deg": np.nan, "R": np.nan, "p": np.nan,
            "N": len(animals), "animals": animals, "excluded": n_excluded,
        }

    mus_rad = np.radians(animals["mu_deg"].values)
    mu2 = circ_mean_rad(mus_rad)          # second-order mean direction
    R2 = float(np.abs(np.mean(np.exp(1j * mus_rad))))  # second-order resultant
    p2 = rayleigh_p(R2, mus_rad.size)

    log.info(
        "Second-order Rayleigh: mu=%.1f°, R=%.3f, N=%d animals, p=%.3g%s",
        np.degrees(mu2), R2, mus_rad.size, p2,
        " (significant)" if p2 < 0.05 else " (n.s.)",
    )
    return {
        "mu_deg": float(np.degrees(mu2)),
        "R": R2,
        "p": p2,
        "N": int(mus_rad.size),
        "animals": animals,
        "excluded": n_excluded,
    }


def loo_robustness(df: pd.DataFrame) -> dict:
    """Leave-one-animal-out sensitivity of the pooled escape direction.

    Recompute the trial-level pooled circular mean after removing each animal
    in turn. If the pooled −74° preference is driven by a single dominant
    animal, one row here will stand out with |Δμ| >> 5°.

    Returns
    -------
    dict
        ``full_mu_deg`` (pooled mean over all escape trials),
        ``max_delta_deg`` (largest absolute deviation), ``loo`` (DataFrame
        with ``animal_id``, ``n_trials``, ``mu_deg``, ``delta_deg``).
    """
    angles = _as_angle_table(df)
    animal = _animal_col(angles)

    all_deg = angles["angle_deg"].astype(float).values
    all_rad = np.radians(all_deg)
    full_mu = circ_mean_rad(all_rad)
    full_mu_deg = float(np.degrees(full_mu))
    log.info("Pooled response direction over all trials: μ=%.1f° (n=%d trials)", full_mu_deg, all_rad.size)

    counts = angles.groupby(animal)["angle_deg"].size()
    rows: list[dict] = []
    for a in counts.index:
        keep = angles[animal] != a
        loo_rad = np.radians(angles.loc[keep, "angle_deg"].astype(float).values)
        loo_mu = circ_mean_rad(loo_rad)
        loo_mu_deg = float(np.degrees(loo_mu))
        # Signed circular deviation from the full pooled mean, in degrees.
        delta_deg = float(np.degrees((loo_mu - full_mu + np.pi) % (2 * np.pi) - np.pi))
        rows.append(
            {"animal_id": a, "n_trials": int(counts[a]),
             "mu_deg": loo_mu_deg, "delta_deg": delta_deg}
        )
    loo_df = pd.DataFrame(rows)
    max_abs = float(loo_df["delta_deg"].abs().max())
    log.info(
        "Leave-one-out: max |Δμ| = %.2f° across %d animals (target ≤5°)",
        max_abs, len(loo_df),
    )
    return {"full_mu_deg": full_mu_deg, "max_delta_deg": max_abs, "loo": loo_df}


def bootstrap_by_animal(
    df: pd.DataFrame,
    n_iter: int | None = None,
    seed: int = 0,
) -> dict:
    """Bootstrap the pooled response direction by resampling ANIMALS.

    Each iteration draws N animals with replacement, pools ALL response trials of
    the drawn animals, and recomputes the trial-level circular mean. This
    propagates between-animal variability into the CI of the pooled direction
    — something a trial-level bootstrap (which would resample individual
    trials and ignore nesting) cannot do.

    Returns
    -------
    dict
        ``mu_deg`` (bootstrap mean direction), ``ci_low`` / ``ci_high`` (95%
        percentile CI in degrees), ``n_animals``, ``n_iter``, ``mus_deg``
        (all bootstrap means).
    """
    n_iter = BOOTSTRAP_ITERATIONS if n_iter is None else n_iter
    angles = _as_angle_table(df)
    animal = _animal_col(angles)

    animals = np.asarray(sorted(angles[animal].unique()))
    n_animals = len(animals)
    if n_animals < 2:
        log.warning("Bootstrap: need >=2 animals, got %d — skipped.", n_animals)
        return {"mu_deg": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "n_animals": n_animals, "n_iter": n_iter, "mus_deg": np.array([])}

    trial_angles = {
        a: np.radians(angles.loc[angles[animal] == a, "angle_deg"].astype(float).values)
        for a in animals
    }
    rng = np.random.default_rng(seed)
    mus = np.empty(n_iter)
    for i in range(n_iter):
        draw = rng.choice(n_animals, size=n_animals, replace=True)
        pooled = np.concatenate([trial_angles[animals[j]] for j in draw])
        mus[i] = circ_mean_rad(pooled)

    mu_boot = circ_mean_rad(mus)
    # Percentile CI in circular space: center on the bootstrap mean so the
    # wrap-around at ±180° is handled correctly.
    centered = np.degrees((mus - mu_boot + np.pi) % (2 * np.pi) - np.pi)
    lo, hi = np.percentile(centered, [2.5, 97.5])
    ci_low = float(np.degrees(mu_boot) + lo)
    ci_high = float(np.degrees(mu_boot) + hi)

    log.info(
        "Bootstrap (animal-resampled, %d iters, N=%d animals): "
        "mu=%.1f°, 95%% CI [%.1f°, %.1f°]",
        n_iter, n_animals, np.degrees(mu_boot), ci_low, ci_high,
    )
    return {
        "mu_deg": float(np.degrees(mu_boot)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_animals": n_animals,
        "n_iter": n_iter,
        "mus_deg": np.degrees(mus),
    }


def wallraff_by_animal(
    df: pd.DataFrame,
    min_trials: int | None = None,
) -> dict:
    """k-sample Wallraff dispersion test across animals (n >= min_trials only).

    CAUTION — pseudo-replication guard: animals with fewer than ``min_trials``
    escape trials are EXCLUDED. With 1–3 trials the within-animal resultant
    length R_k ≈ 1 (no measurable spread), so including them would bias the
    dispersion comparison. Figure captions must state:

        "Animals with <5 trials excluded from k-sample Wallraff."

    The result is reported for transparency, but the primary population-level
    inference should come from the second-order Rayleigh test and the
    hierarchical model in ``mcmc_analysis.py`` (partial pooling).
    """
    min_trials = MIN_TRIALS_WALLRAFF if min_trials is None else min_trials
    angles = _as_angle_table(df)
    animal = _animal_col(angles)

    counts = trial_counts_per_animal(angles)
    eligible = counts[counts >= min_trials].index
    n_excluded = int((counts < min_trials).sum())
    log.info(
        "Wallraff-by-animal: %d/%d animals have >=%d trials "
        "(%d excluded — 'Animals with <%d trials excluded from k-sample "
        "Wallraff')",
        len(eligible), len(counts), min_trials, n_excluded, min_trials,
    )
    if len(eligible) < 2:
        log.warning("Wallraff-by-animal: fewer than 2 eligible animals — skipped.")
        return {"H": np.nan, "p": np.nan, "n_groups": 0, "excluded": n_excluded,
                "min_trials": min_trials}

    grouped = angles.groupby(animal)["angle_deg"]
    groups = [np.radians(grouped.get_group(a).astype(float).values) for a in eligible]
    result = wallraff_test(*groups, use_common_ref=True)
    result["n_groups"] = len(eligible)
    result["excluded"] = n_excluded
    result["min_trials"] = min_trials
    log.info(
        "Wallraff-by-animal: H=%.3f, p=%.3g, k=%d animals (n>=%d)",
        result["H"], result["p"], len(eligible), min_trials,
    )
    return result


# ══════════════════════════════════════════════════════════════════════
# CLI driver
# ══════════════════════════════════════════════════════════════════════


def _safe_savefig(fig, path: Path, **kwargs) -> None:
    """Save figure, removing a stale file first (Windows OSError 22 workaround)."""
    path = Path(path)
    if path.exists():
        path.unlink()
    fig.savefig(path, **kwargs)


def run_individual_checks(
    df: pd.DataFrame,
    output_dir: str | Path,
    filter_low_n: bool = False,
) -> dict:
    """Run the individual-robustness battery and save figures + summary CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Frame-level DataFrame with all animals and trials.
    output_dir : str | Path
        Directory for output CSV and figures/ subdirectory.
    filter_low_n : bool
        If True, exclude animals with <3 (second-order Rayleigh) or <5 (Wallraff)
        response trials, reproducing the old behaviour. If False (default),
        include all animals with ≥1 response trial.

    Generates under ``output_dir``:

    * ``figures/suppl_trial_counts.png`` — distribution of response trials/animal
    * ``figures/suppl_second_order.png`` — per-animal mean directions + 2nd-order mean
    * ``figures/suppl_loo.png`` — leave-one-animal-out pooled direction
    * ``individual_summary.csv`` — per-animal columns (animal_id, n_trials, mu_deg, R)

    Returns a dict of the analysis results for logging/callers.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from cercus.visualization.individual import (
        plot_loo,
        plot_second_order,
        plot_trial_counts,
    )

    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── All animal IDs from the full input df (before filtering) ──
    all_animals = pd.Index(df[_animal_col(df)].unique())
    n_total = len(all_animals)

    angles = trial_escape_angles(df)
    animal = _animal_col(angles)
    if angles.empty:
        log.warning("No response trials for individual robustness checks — skipping.")
        return {}

    # 1) Trial-count distribution — include zero-count animals so N = n_total
    counts = trial_counts_per_animal(angles)
    counts = counts.reindex(all_animals, fill_value=0).astype(int)

    # k-sample Wallraff — filter by filter_low_n
    wallraff_min = MIN_TRIALS_WALLRAFF if filter_low_n else 1
    wallraff_res = wallraff_by_animal(angles, min_trials=wallraff_min)

    fig = plot_trial_counts(counts, wallraff=wallraff_res, n_total=n_total, filter_low_n=filter_low_n)
    _safe_savefig(fig, fig_dir / "suppl_trial_counts.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2) Second-order (per-animal) analysis — filter by filter_low_n
    second_min = MIN_TRIALS_SECOND_ORDER if filter_low_n else 1
    second = second_order_analysis(angles, min_trials=second_min)

    # Pad `second["animals"]` with zero-count animals so the polar plot
    # shows the full N_total even for animals with 0 response trials.
    # ponytail: one reindex fills the gap — no second analysis pass needed.
    animals_in_second = set(second["animals"]["animal_id"])
    zero_animals = all_animals.difference(animals_in_second)
    if len(zero_animals) > 0:
        zero_rows = pd.DataFrame({
            "animal_id": list(zero_animals),
            "n_trials": 0, "mu_deg": np.nan, "R": np.nan,
        })
        second["animals"] = pd.concat(
            [second["animals"], zero_rows], ignore_index=True
        ).sort_values("animal_id").reset_index(drop=True)
        # Keep second["N"] as the valid-animal count for Rayleigh stats
        # (the padded zero-count rows have NaN mu_deg and are skipped by
        # plot_second_order's validity mask).

    # 3) Leave-one-animal-out sensitivity — always all animals with response trials
    loo = loo_robustness(angles)

    # Pad LOO with zero-count animals (removing them doesn't change the pooled mean).
    animals_in_loo = set(loo["loo"]["animal_id"])
    zero_loo = all_animals.difference(animals_in_loo)
    if len(zero_loo) > 0:
        zero_rows = pd.DataFrame({
            "animal_id": list(zero_loo),
            "n_trials": 0,
            "mu_deg": loo["full_mu_deg"],
            "delta_deg": 0.0,
        })
        loo["loo"] = pd.concat(
            [loo["loo"], zero_rows], ignore_index=True
        ).sort_values("animal_id").reset_index(drop=True)

    # 4) Animal-level bootstrap CI (logged)
    boot = bootstrap_by_animal(angles)

    # ── Supplementary figures ──
    n_plotted = second["N"]
    fig = plot_second_order(
        second["animals"],
        n_total=n_total, filter_low_n=filter_low_n,
    )
    _safe_savefig(fig, fig_dir / "suppl_second_order.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig = plot_loo(loo["loo"], pooled_mu_deg=loo["full_mu_deg"], n_total=n_total)
    _safe_savefig(fig, fig_dir / "suppl_loo.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ── Per-animal summary CSV ──
    summary = _per_animal_summary(angles)
    summary["n3_eligible"] = summary["n_trials"] >= MIN_TRIALS_SECOND_ORDER
    # Pad summary with zero-count animals too.
    animals_in_summary = set(summary["animal_id"])
    zero_summary = all_animals.difference(animals_in_summary)
    if len(zero_summary) > 0:
        zero_rows = pd.DataFrame({
            "animal_id": list(zero_summary),
            "n_trials": 0, "mu_deg": np.nan, "R": np.nan, "n3_eligible": False,
        })
        summary = pd.concat(
            [summary, zero_rows], ignore_index=True
        ).sort_values("animal_id").reset_index(drop=True)
    summary_csv = output_dir / "individual_summary.csv"
    summary.to_csv(summary_csv, index=False, float_format="%.3f")

    n_response_animals = angles[animal].nunique()
    log.info(
        "Individual robustness checks complete: %d total animals, %d with ≥1 response trials, "
        "pooled μ=%.1f°, second-order μ=%.1f° (N=%d / %d plotted, p=%.3g), "
        "max LOO |Δμ|=%.2f°, bootstrap CI [%.1f°, %.1f°]",
        n_total, n_response_animals,
        loo["full_mu_deg"], second["mu_deg"], n_plotted, n_total,
        second["p"], loo["max_delta_deg"], boot["ci_low"], boot["ci_high"],
    )
    if wallraff_res and not np.isnan(wallraff_res.get("p", np.nan)):
        n_eligible = wallraff_res.get("n_groups", 0)
        log.info(
            "Wallraff-by-animal: H=%.3f, p=%.3g, k=%d/%d animals (filter_low_n=%s)",
            wallraff_res["H"], wallraff_res["p"], n_eligible, n_total,
            filter_low_n,
        )
    log.info("Supplementary figures: %s/suppl_trial_counts.png, %s/suppl_second_order.png, %s/suppl_loo.png", fig_dir, fig_dir, fig_dir)
    log.info("Individual summary CSV: %s", summary_csv)

    return {"second_order": second, "loo": loo, "bootstrap": boot, "summary": summary, "wallraff": wallraff_res}
