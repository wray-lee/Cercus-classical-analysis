"""
Cercus Framework — Circular Statistics
=======================================
Pure functions for circular statistical tests used in polar plots.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.stats import f as f_dist

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
