"""
Cercus Framework — Adaptive V_max Threshold Selection
=====================================================
KDE valley → log-space GMM → (IQR-GMM) → fallback 级联，单一事实源。
从 ``population_analysis.py`` 原样迁移；full 模式跨范式池化复用同一入口。

Pure numerics — no matplotlib, no file I/O.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import brentq
from scipy.signal import argrelmin, argrelmax
from scipy.stats import gaussian_kde, norm
from sklearn.mixture import GaussianMixture

from cercus.config import get_config
from cercus.constants.thresholds import ESCAPE_VMAX_THRESHOLD

log = logging.getLogger(__name__)

_cfg = get_config()
USE_ADAPTIVE_THRESHOLD: bool = bool(_cfg.analysis.vmax_thresholding.use_adaptive)
FALLBACK_VMAX_THRESHOLD: float = float(_cfg.analysis.vmax_thresholding.fallback_threshold)
KDE_SEARCH_UPPER: float = float(_cfg.analysis.vmax_thresholding.kde_search_upper)
ENABLE_IQR_GMM: bool = bool(_cfg.analysis.vmax_thresholding.enable_iqr_gmm)


def compute_kde_valley_threshold(vmax_values: np.ndarray) -> float | None:
    """Locate the valley between the low-speed cluster and the escape-burst
    cluster by finding the local minimum of a KDE density curve.

    Steps:
        1. Fit ``scipy.stats.gaussian_kde`` with a slightly narrow bandwidth
           to preserve local features (``bw_method=0.3``).
        2. Evaluate the density on ``[0, KDE_SEARCH_UPPER]``.
        3. Use ``argrelmax`` to find peaks and ``argrelmin`` to find troughs.
        4. Return the deepest trough that sits *between* the first two peaks
           (the "quiet" peak and the "burst" peak).

    Returns
    -------
    float | None
        The valley threshold in mm/s, or ``None`` if no valid valley is found.
    """
    vmax_clean = vmax_values[~np.isnan(vmax_values)]
    if len(vmax_clean) < 10:
        log.warning("KDE valley: only %d valid samples — skipping.", len(vmax_clean))
        return None

    try:
        kde = gaussian_kde(vmax_clean, bw_method=0.3)
        x = np.linspace(0, KDE_SEARCH_UPPER, 1000)
        density = kde(x)

        # ── Find peaks and valleys ──
        peak_idx = argrelmax(density, order=5)[0]
        valley_idx = argrelmin(density, order=5)[0]

        if len(peak_idx) < 2:
            log.warning("KDE valley: fewer than 2 peaks detected (%d) — skipping.", len(peak_idx))
            return None

        # The first two peaks represent the two main modes
        first_peak_x = x[peak_idx[0]]
        second_peak_x = x[peak_idx[1]]
        log.info("KDE valley: peaks at %.1f and %.1f mm/s", first_peak_x, second_peak_x)

        # Filter valleys that lie between the two peaks
        between = valley_idx[(x[valley_idx] > first_peak_x) & (x[valley_idx] < second_peak_x)]

        if len(between) == 0:
            log.warning("KDE valley: no trough found between peaks — skipping.")
            return None

        # Pick the deepest trough (lowest density) as the threshold
        deepest = between[np.argmin(density[between])]
        threshold = float(x[deepest])
        log.info("KDE valley threshold: %.1f mm/s (density=%.6f)", threshold, density[deepest])
        return threshold

    except Exception as exc:
        log.warning("KDE valley detection failed (%s) — skipping.", exc)
        return None


def compute_log_gmm_threshold(
    vmax_values: np.ndarray,
    min_speed_floor: float = 10.0,
) -> tuple[float, float] | None:
    """Fit a **2-component** GMM in log-space on moving trials (v_max >= min_speed_floor)
    to separate Walking (spontaneous movement) from Escape bursts.

    Filtering out the stationary noise floor (< 10 mm/s) avoids spending GMM components
    on near-zero noise, yielding the clean empirical escape threshold (~96-98 mm/s).

    Returns
    -------
    tuple[float, float] | None
        ``(start_threshold, escape_threshold)`` in mm/s, or ``None`` if
        fitting fails or data is insufficient (< 15 valid moving samples).
    """
    vmax_clean = vmax_values[~np.isnan(vmax_values) & (vmax_values >= min_speed_floor)]
    if len(vmax_clean) < 15:
        log.warning("Log-GMM: only %d valid active samples (>= %.1f mm/s) — skipping.", len(vmax_clean), min_speed_floor)
        return None

    try:
        log_vmax = np.log(vmax_clean)
        X = log_vmax.reshape(-1, 1)

        gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
        gmm.fit(X)

        # Sort components by mean in log-space: low (walking) / high (escape)
        order = np.argsort(gmm.means_.ravel())
        means = gmm.means_.ravel()[order]
        vars_ = gmm.covariances_.ravel()[order]
        weights = gmm.weights_.ravel()[order]

        def _pair_diff(log_x: float) -> float:
            return (weights[0] * norm.pdf(log_x, means[0], np.sqrt(vars_[0]))
                    - weights[1] * norm.pdf(log_x, means[1], np.sqrt(vars_[1])))

        log_escape = brentq(_pair_diff, means[0], means[1])
        escape_threshold = float(np.exp(log_escape))
        start_threshold = float(min_speed_floor)

        log.info("Log-GMM 2-component (moving >= %.1f mm/s): start=%.1f, escape=%.1f mm/s  "
                 "(mu_walk=%.1f, mu_esc=%.1f mm/s)",
                 min_speed_floor, start_threshold, escape_threshold,
                 np.exp(means[0]), np.exp(means[1]))
        return start_threshold, escape_threshold

    except Exception as exc:
        log.warning("Log-GMM fitting failed (%s) — skipping.", exc)
        return None


def compute_iqr_gmm_threshold(vmax_values: np.ndarray) -> float | None:
    """Fit a 2-component GMM on V_max values truncated by the IQR rule.

    Uses ``Q3 + 1.5 * IQR`` as an adaptive upper bound derived from the data
    itself, replacing the previous hard-coded 300 mm/s ceiling.  The GMM then
    operates in the original physical domain on the truncated data.

    Returns
    -------
    float | None
        The threshold in mm/s, or ``None`` if fitting fails or data is
        insufficient (< 10 valid samples after truncation).
    """
    vmax_clean = vmax_values[~np.isnan(vmax_values)]
    if len(vmax_clean) < 10:
        log.warning("IQR-GMM: only %d valid samples — skipping.", len(vmax_clean))
        return None

    try:
        q1 = np.percentile(vmax_clean, 25)
        q3 = np.percentile(vmax_clean, 75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr

        vmax_truncated = vmax_clean[vmax_clean < upper_bound]

        if len(vmax_truncated) < 10:
            log.warning("IQR-GMM: only %d samples below IQR upper bound (%.1f) — skipping.",
                        len(vmax_truncated), upper_bound)
            return None

        X = vmax_truncated.reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
        gmm.fit(X)

        # Sort components by mean
        order = np.argsort(gmm.means_.ravel())
        mu0, mu1 = gmm.means_.ravel()[order]
        var0, var1 = gmm.covariances_.ravel()[order]
        w0, w1 = gmm.weights_.ravel()[order]

        def _diff(x: float) -> float:
            return w0 * norm.pdf(x, mu0, np.sqrt(var0)) - w1 * norm.pdf(x, mu1, np.sqrt(var1))

        threshold = brentq(_diff, mu0, mu1)
        log.info("IQR-GMM threshold: %.1f mm/s  (Q1=%.1f, Q3=%.1f, IQR=%.1f, upper=%.1f, mu0=%.1f, mu1=%.1f)",
                 threshold, q1, q3, iqr, upper_bound, mu0, mu1)
        return float(threshold)

    except Exception as exc:
        log.warning("IQR-GMM fitting failed (%s) — skipping.", exc)
        return None


def select_vmax_threshold(vmax_values: np.ndarray) -> tuple[float, str, dict]:
    """Priority cascade KDE → Log-GMM → (IQR-GMM) → fallback; or fixed config.

    Returns
    -------
    (threshold, method, candidates)
        ``method`` ∈ {"KDE valley", "Log-GMM (escape)", "IQR-GMM",
        "hardcoded fallback", "config (fixed)"};
        ``candidates`` = {"gmm_escape_threshold": float | None} for plotting.
    """
    kde_threshold = compute_kde_valley_threshold(vmax_values)
    log_gmm_result = compute_log_gmm_threshold(vmax_values)
    iqr_gmm_threshold = compute_iqr_gmm_threshold(vmax_values) if ENABLE_IQR_GMM else None

    gmm_start_threshold: float | None = None
    gmm_escape_threshold: float | None = None
    if log_gmm_result is not None:
        gmm_start_threshold, gmm_escape_threshold = log_gmm_result

    log.info("Candidate thresholds:  KDE=%.1f  Log-GMM start=%s escape=%s  IQR-GMM=%s",
             kde_threshold if kde_threshold else -1,
             f"{gmm_start_threshold:.1f}" if gmm_start_threshold else "None",
             f"{gmm_escape_threshold:.1f}" if gmm_escape_threshold else "None",
             f"{iqr_gmm_threshold:.1f}" if iqr_gmm_threshold else "None")

    if USE_ADAPTIVE_THRESHOLD:
        if kde_threshold is not None:
            threshold, method = kde_threshold, "KDE valley"
        elif gmm_escape_threshold is not None:
            threshold, method = gmm_escape_threshold, "Log-GMM (escape)"
        elif ENABLE_IQR_GMM and iqr_gmm_threshold is not None:
            threshold, method = iqr_gmm_threshold, "IQR-GMM"
        else:
            threshold, method = FALLBACK_VMAX_THRESHOLD, "hardcoded fallback"
        log.info("Adaptive threshold selected: %.1f mm/s  [method=%s]", threshold, method)
    else:
        threshold, method = ESCAPE_VMAX_THRESHOLD, "config (fixed)"
        log.info("Using fixed threshold from config: %.1f mm/s  [adaptive disabled]", threshold)

    return threshold, method, {"gmm_escape_threshold": gmm_escape_threshold}


__all__ = [
    "compute_kde_valley_threshold",
    "compute_log_gmm_threshold",
    "compute_iqr_gmm_threshold",
    "select_vmax_threshold",
]
