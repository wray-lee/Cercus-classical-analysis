"""
Cercus Framework — Circular Statistics
=======================================
Pure functions for circular statistical tests used in polar plots.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.stats import f as f_dist
from scipy.stats import kruskal

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


def circ_mean_rad(ang_rad: np.ndarray) -> float:
    """Compute the circular mean of angles in radians.

    Args:
        ang_rad: Angular values in radians (any range; wrapped internally).

    Returns:
        Mean angle in radians in [-π, π).
    """
    ang_rad = np.asarray(ang_rad)
    if ang_rad.size == 0:
        return np.nan
    return float(np.angle(np.mean(np.exp(1j * ang_rad))))


def circ_dist_rad(a: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Compute pairwise circular distance between angles and a reference.

    The circular distance is the minimum angular difference, always in
    [0, π]. Implements Batschelet (1981), eq. 1.3.8.

    Args:
        a: Input angles in radians.
        ref: Reference angle(s) in radians.

    Returns:
        Angular distances in radians, each in [0, π].
    """
    a = np.asarray(a)
    ref = np.asarray(ref)
    return np.abs((a - ref + np.pi) % (2 * np.pi) - np.pi)


def wallraff_test(
    *groups_rad: np.ndarray,
    use_common_ref: bool = False,
) -> dict:
    """Wallraff test of angular dispersion (Batschelet 1981, chap 6.10; Zar 2010).

    Non-parametric test for whether two or more groups have equal angular
    dispersion. Each angle is first converted to its circular distance from
    a reference direction, and the resulting linear distances are compared
    with a Kruskal-Wallis test (scipy.stats.kruskal).

    By default (``use_common_ref=False``) each group is referenced to its own
    circular mean, so the test compares *within-group* dispersions — the
    appropriate analogue of the parametric concentration comparison. With
    ``use_common_ref=True`` all groups share the pooled circular mean.

    Args:
        *groups_rad: Positional groups of angle arrays (radians).
        use_common_ref: If True, reference all groups to the pooled circular
            mean; if False (default), reference each group to its own mean.

    Returns:
        dict with keys:
            - ``H``: Kruskal-Wallis statistic.
            - ``p``: asymptotic p-value.
            - ``dof``: degrees of freedom (number of groups - 1).
            - ``distances``: list of per-group distance arrays (radians).
            - ``warnings``: list of logged warning strings.
    """
    warnings: list[str] = []
    for i, g in enumerate(groups_rad):
        g_arr = np.asarray(g)
        if g_arr.size == 0:
            log.warning("Wallraff: group %d is empty; excluded from test.", i)
            continue
        if g_arr.size < 5:
            msg = f"group {i}: n={g_arr.size} < 5; dispersion estimate may be unstable"
            log.warning("Wallraff: %s", msg)
            warnings.append(msg)

    groups = [np.asarray(g) for g in groups_rad if np.asarray(g).size > 0]
    k = len(groups)
    if k < 2:
        log.warning("Wallraff: need at least 2 non-empty groups (got %d).", k)
        return {"H": np.nan, "p": np.nan, "dof": np.nan, "distances": [], "warnings": warnings}

    if use_common_ref:
        pooled = np.concatenate(groups)
        ref = circ_mean_rad(pooled)
    else:
        ref = None

    distances = [
        circ_dist_rad(g, circ_mean_rad(g) if ref is None else ref) for g in groups
    ]

    H, p_val = kruskal(*distances)
    dof = k - 1

    return {"H": H, "p": p_val, "dof": dof, "distances": distances, "warnings": warnings}


def wallraff_test_with_ref(groups: list[np.ndarray], ref_deg: float) -> dict:
    """Wallraff test of angular dispersion against a fixed reference direction.

    Variant of :func:`wallraff_test` in which all groups are referenced to a
    single externally specified direction (in degrees) rather than their own
    means — useful when a biologically meaningful reference (e.g. the stimulus
    axis) is known a priori. See Batschelet (1981) chap 6.10.

    Args:
        groups: List of angle arrays (radians).
        ref_deg: Fixed reference direction in degrees.

    Returns:
        Same dict structure as :func:`wallraff_test` (``H``, ``p``, ``dof``,
        ``distances``, ``warnings``).
    """
    ref_rad = np.radians(ref_deg)
    distances = [circ_dist_rad(np.asarray(g), ref_rad) for g in groups]
    H, p_val = kruskal(*distances)
    return {
        "H": H,
        "p": p_val,
        "dof": len(distances) - 1,
        "distances": distances,
        "warnings": [],
    }
