"""
Cercus Framework — Circular Statistics
=======================================
Pure functions for circular statistical tests used in polar plots.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.stats import f as f_dist
from scipy.stats import mannwhitneyu

log = logging.getLogger(__name__)


def rayleigh_p(resultant: float, n: int) -> float:
    """Rayleigh test for non-uniformity of circular data.

    Computes the p-value for the null hypothesis that the data is uniformly
    distributed around the circle. Uses Zar (1999) approximation valid for n > 10.
    """
    if n <= 10:
        r_sq = resultant**2
        p = np.exp(-n * r_sq) * (1 + (2 * n * r_sq - r_sq**2) / (4 * n))
    else:
        p = np.exp(-n * resultant**2)
    return float(np.clip(p, 0.0, 1.0))


def watson_williams_test(
    angles1: np.ndarray, angles2: np.ndarray
) -> tuple[float, float]:
    """Watson-Williams F-test: test whether two groups share the same mean angle.

    Implementation follows Zar (2010) Circular Statistics, §26.13–26.14.
    """
    a1 = np.asarray(angles1)
    a2 = np.asarray(angles2)

    n1 = a1.size
    n2 = a2.size
    N = n1 + n2

    if n1 < 2 or n2 < 2 or N < 3:
        log.warning("Watson-Williams: insufficient sample size (n1=%d, n2=%d)", n1, n2)
        return np.nan, np.nan

    R1 = float(np.abs(np.exp(1j * a1).sum()))
    R2 = float(np.abs(np.exp(1j * a2).sum()))
    R_pooled = float(np.abs(np.exp(1j * np.concatenate([a1, a2])).sum()))

    numerator = (N - 2) * (R1 + R2 - R_pooled)
    denominator = N - R1 - R2

    if denominator <= 0:
        log.warning(
            "Watson-Williams: denominator <= 0 (N=%d, R1=%.3f, R2=%.3f)", N, R1, R2
        )
        return np.nan, np.nan

    F = numerator / denominator

    if R_pooled < 0.45:
        log.warning(
            "Watson-Williams: R_pooled=%.3f < 0.45; correction factor may be needed "
            "(see Zar 2010 Table 26.4). Returning uncorrected F.",
            R_pooled,
        )

    p_ww = float(f_dist.sf(F, 1, N - 2))
    return F, p_ww


def circ_mean(angles_rad: np.ndarray) -> float:
    """Circular mean of angles in radians, returns in [-pi, pi]."""
    return float(np.angle(np.sum(np.exp(1j * angles_rad))))


def circ_dist(a: np.ndarray, mu: float) -> np.ndarray:
    """Unsigned angular distance from each angle to mean, in [0, pi]."""
    return np.abs(np.angle(np.exp(1j * (a - mu))))


def wallraff_test(
    a1: np.ndarray, a2: np.ndarray
) -> dict[str, float]:
    """Wallraff test (1979) for difference in concentration between two groups.

    Non-parametric: computes angular distance of each observation to its own
    group circular mean, then compares the two distance distributions with
    Mann-Whitney U.

    Parameters
    ----------
    a1, a2 : array-like
        Angles in radians.

    Returns
    -------
    dict with keys: U, p, median_disp1, median_disp2, mean_disp1, mean_disp2,
    Rbar1, Rbar2, dR
    """
    a1 = np.asarray(a1)
    a2 = np.asarray(a2)
    mu1 = circ_mean(a1)
    mu2 = circ_mean(a2)
    d1 = circ_dist(a1, mu1)
    d2 = circ_dist(a2, mu2)
    U, p = mannwhitneyu(d1, d2, alternative="two-sided")
    Rbar1 = float(np.abs(np.mean(np.exp(1j * a1))))
    Rbar2 = float(np.abs(np.mean(np.exp(1j * a2))))
    return {
        "U": float(U),
        "p": float(p),
        "median_disp1": float(np.median(d1)),
        "median_disp2": float(np.median(d2)),
        "mean_disp1": float(np.mean(d1)),
        "mean_disp2": float(np.mean(d2)),
        "Rbar1": Rbar1,
        "Rbar2": Rbar2,
        "dR": Rbar1 - Rbar2,
    }


def bootstrap_dR_ci(
    a1: np.ndarray, a2: np.ndarray, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Bootstrap 95% CI for ΔR = R1 - R2.

    Returns (dR_obs, ci_low, ci_high).
    """
    rng = np.random.default_rng(seed)
    R1 = float(np.abs(np.exp(1j * a1).mean()))
    R2 = float(np.abs(np.exp(1j * a2).mean()))
    obs = R1 - R2
    pooled = np.concatenate([a1, a2])
    n1 = len(a1)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        rng.shuffle(pooled)
        b1 = pooled[:n1]
        b2 = pooled[n1:]
        diffs[i] = np.abs(np.exp(1j * b1).mean()) - np.abs(np.exp(1j * b2).mean())
    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return obs, lo, hi
