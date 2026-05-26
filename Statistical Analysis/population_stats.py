"""
Population Statistics — GEE & Circular Statistics
=================================================
Reads population_master.csv from aggregator.py and runs:
    1. Generalized Estimating Equation (GEE) for escape probability
    2. Circular statistics (Rayleigh Test) for escape direction preference
    3. Publication-grade statistical plots

Outputs:
    escape_probability_lmm.png — GEE escape probability bar/box plot
    escape_direction_rose.png  — Polar rose plot with KDE and mean vector

Usage:
    python population_stats.py --input population_master.csv --output results/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Publication Style & NPG Palette
# ──────────────────────────────────────────────────────────────────────

def _apply_publication_style():
    """Inject Nature/Science/Cell compliant rcParams."""
    rc = plt.rcParams
    rc["font.family"] = "sans-serif"
    rc["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    rc["svg.fonttype"] = "none"
    rc["pdf.fonttype"] = 42
    rc["font.size"] = 7
    rc["axes.titlesize"] = 9
    rc["axes.labelsize"] = 8
    rc["legend.fontsize"] = 7
    rc["xtick.labelsize"] = 7
    rc["ytick.labelsize"] = 7
    rc["lines.linewidth"] = 1.0
    rc["axes.linewidth"] = 0.75
    rc["axes.spines.top"] = False
    rc["axes.spines.right"] = False
    rc["xtick.direction"] = "in"
    rc["ytick.direction"] = "in"
    rc["xtick.major.size"] = 3
    rc["ytick.major.size"] = 3
    rc["xtick.major.width"] = 0.75
    rc["ytick.major.width"] = 0.75
    rc["xtick.minor.size"] = 1.5
    rc["ytick.minor.size"] = 1.5
    rc["legend.frameon"] = False
    rc["legend.borderaxespad"] = 0
    rc["figure.dpi"] = 150
    rc["savefig.dpi"] = 300
    rc["savefig.transparent"] = True


_apply_publication_style()

# NPG palette
COLOR_LEFT = "#4DBBD5"
COLOR_RIGHT = "#E64B35"
COLOR_CONTROL = "#999999"
CONDITION_COLORS = {
    "looming_wind_left": COLOR_LEFT,
    "looming_wind_right": COLOR_RIGHT,
    "air_puff_left": COLOR_LEFT,
    "air_puff_right": COLOR_RIGHT,
    "baseline_visual": COLOR_CONTROL,
    "baseline_wind": COLOR_CONTROL,
}


def _resolve_color(condition: str) -> str:
    """Resolve condition name to NPG color."""
    cond_lower = condition.lower()
    for key, color in CONDITION_COLORS.items():
        if key in cond_lower:
            return color
    if "left" in cond_lower:
        return COLOR_LEFT
    if "right" in cond_lower:
        return COLOR_RIGHT
    return COLOR_CONTROL


# ══════════════════════════════════════════════════════════════════════
# 1. Escape Probability — Linear Mixed-Effects Model
# ══════════════════════════════════════════════════════════════════════

def run_escape_lmm(master: pd.DataFrame) -> dict:
    """
    Fit a Generalized Estimating Equation (GEE) for escape probability.

    Model: is_escaped ~ type  (Binomial family, grouped by subject_id)
    Uses statsmodels GEE — appropriate for binary (0/1) outcomes where
    a linear mixed model violates the Gaussian assumption.

    Returns dict with model summary and per-condition escape rates.
    """
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except ImportError:
        log.warning("statsmodels not installed. Skipping GEE. "
                    "Install with: pip install statsmodels")
        return {"model": None, "rates": _compute_escape_rates(master)}

    # Encode is_escaped as float
    df = master.copy()
    df["is_escaped"] = df["is_escaped"].astype(float)

    # Drop rows with missing subject_id or type
    df = df.dropna(subset=["subject_id", "type", "is_escaped"])

    if df.empty or df["subject_id"].nunique() < 2:
        log.warning("Insufficient data for GEE (need >= 2 subjects). Skipping.")
        return {"model": None, "rates": _compute_escape_rates(master)}

    try:
        model = smf.gee("is_escaped ~ type", data=df, groups=df["subject_id"],
                        family=sm.families.Binomial())
        result = model.fit()
        log.info("GEE fitted successfully.\n%s", result.summary().as_text())
        return {"model": result, "rates": _compute_escape_rates(master)}
    except Exception as exc:
        log.warning("GEE fitting failed: %s", exc)
        return {"model": None, "rates": _compute_escape_rates(master)}


def _compute_escape_rates(master: pd.DataFrame) -> pd.DataFrame:
    """Compute per-condition escape rate with 95% CI (Wilson score interval)."""
    records = []
    for cond, grp in master.groupby("type"):
        n = len(grp)
        k = int(grp["is_escaped"].sum())
        rate = k / n if n > 0 else 0.0
        # Wilson score interval
        if n > 0:
            z = 1.96
            denom = 1 + z**2 / n
            center = (rate + z**2 / (2 * n)) / denom
            margin = z * np.sqrt((rate * (1 - rate) + z**2 / (4 * n)) / n) / denom
            ci_lo = max(0, center - margin)
            ci_hi = min(1, center + margin)
        else:
            ci_lo = ci_hi = 0.0
        records.append({
            "type": cond, "n_trials": n, "n_escaped": k,
            "escape_rate": rate, "ci_lo": ci_lo, "ci_hi": ci_hi,
        })
    return pd.DataFrame(records)


def plot_escape_probability(
    master: pd.DataFrame,
    lmm_result: dict | None = None,
    figsize: tuple[float, float] = (5, 4),
) -> plt.Figure:
    """
    Bar plot of escape probability per condition with error bars (95% CI).
    Overlaid with individual subject data points.
    """
    rates = _compute_escape_rates(master)
    if rates.empty:
        log.warning("No data for escape probability plot.")
        fig, ax = plt.subplots(figsize=figsize)
        return fig

    fig, ax = plt.subplots(figsize=figsize)

    x_pos = np.arange(len(rates))
    colors = [_resolve_color(row["type"]) for _, row in rates.iterrows()]

    # Bars
    bars = ax.bar(x_pos, rates["escape_rate"], width=0.6, color=colors,
                  edgecolor="none", alpha=0.7, zorder=2)

    # Error bars (95% CI)
    yerr_lo = rates["escape_rate"] - rates["ci_lo"]
    yerr_hi = rates["ci_hi"] - rates["escape_rate"]
    ax.errorbar(x_pos, rates["escape_rate"],
                yerr=[yerr_lo.values, yerr_hi.values],
                fmt="none", ecolor="black", elinewidth=0.75, capsize=3, zorder=3)

    # Individual subject data points (jittered)
    for i, (_, row) in enumerate(rates.iterrows()):
        cond_data = master[master["type"] == row["type"]]
        subject_rates = cond_data.groupby("subject_id")["is_escaped"].mean()
        jitter = np.random.normal(0, 0.05, len(subject_rates))
        ax.scatter(np.full(len(subject_rates), i) + jitter, subject_rates.values,
                   color="black", s=12, alpha=0.5, zorder=4, edgecolors="none")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(rates["type"], rotation=30, ha="right")
    ax.set_ylabel("Escape Probability")
    ax.set_ylim(0, 1.05)

    # Add GEE p-value annotation if available
    if lmm_result and lmm_result.get("model") is not None:
        try:
            pvals = lmm_result["model"].pvalues
            type_pvals = {k: v for k, v in pvals.items() if k != "Intercept"}
            if type_pvals:
                min_p = min(type_pvals.values())
                sig = "***" if min_p < 0.001 else "**" if min_p < 0.01 else "*" if min_p < 0.05 else "n.s."
                ax.text(0.98, 0.95, f"GEE p={min_p:.4f} {sig}",
                        transform=ax.transAxes, ha="right", va="top", fontsize=7)
        except Exception:
            pass

    fig.tight_layout(pad=1.0)
    return fig


# ══════════════════════════════════════════════════════════════════════
# 2. Circular Statistics — Escape Direction
# ══════════════════════════════════════════════════════════════════════

def _rayleigh_test(angles: np.ndarray) -> dict:
    """
    Rayleigh Test for uniformity of circular data.

    H0: the population is uniformly distributed around the circle.
    Returns: R (mean resultant length), z (test statistic), p (p-value)
    """
    n = len(angles)
    if n < 3:
        return {"R": np.nan, "z": np.nan, "p": np.nan, "mean_angle": np.nan}

    # Mean resultant vector
    C = np.sum(np.cos(angles))
    S = np.sum(np.sin(angles))
    R_bar = np.sqrt(C**2 + S**2) / n

    # Rayleigh test statistic: z = n * R_bar^2
    z = n * R_bar**2

    # Approximate p-value (large-n approximation)
    p = np.exp(-z) * (1 + (2 * z - z**2) / (4 * n) -
                       (24 * z - 132 * z**2 + 76 * z**3 - 9 * z**4) / (288 * n**2))

    mean_angle = np.arctan2(S, C)

    return {"R": R_bar, "z": z, "p": min(p, 1.0), "mean_angle": mean_angle}


def _von_mises_kde(angles: np.ndarray, n_points: int = 360) -> tuple[np.ndarray, np.ndarray]:
    """
    Kernel Density Estimation on circular data using von Mises kernel.
    Returns (theta_grid, density).
    """
    from scipy.special import i0 as bessel_i0

    n = len(angles)
    if n < 2:
        theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        return theta, np.ones(n_points) / (2 * np.pi)

    # Bandwidth selection (Silverman-like for circular data)
    R = np.abs(np.exp(1j * angles).mean())
    if R < 0.5:
        kappa = 0.5  # low concentration → wider kernel
    else:
        kappa = max(0.5, R * (2 - R**2) / (1 - R**2 + 1e-10))

    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    # KDE: sum of von Mises kernels
    density = np.zeros(n_points)
    for a in angles:
        diff = theta - a
        density += np.exp(kappa * np.cos(diff))
    density /= (n * 2 * np.pi * bessel_i0(kappa))

    return theta, density


def run_circular_stats(master: pd.DataFrame) -> dict:
    """
    Run circular statistics on escape directions.

    Returns per-condition Rayleigh test results and KDE estimates.
    """
    escaped = master[master["is_escaped"] == True].copy()
    escaped = escaped.dropna(subset=["escape_direction_rad"])

    if escaped.empty:
        log.warning("No escaped trials with direction data for circular stats.")
        return {}

    results = {}
    for cond, grp in escaped.groupby("type"):
        angles = grp["escape_direction_rad"].values
        if len(angles) < 3:
            log.warning("Condition '%s': only %d directions, skipping Rayleigh test.", cond, len(angles))
            continue

        rt = _rayleigh_test(angles)
        theta_kde, density_kde = _von_mises_kde(angles)

        results[cond] = {
            "angles": angles,
            "rayleigh": rt,
            "kde_theta": theta_kde,
            "kde_density": density_kde,
            "n": len(angles),
        }
        log.info("Condition '%s': n=%d, R=%.3f, z=%.3f, p=%.4f, mean=%.1f°",
                 cond, len(angles), rt["R"], rt["z"], rt["p"],
                 np.degrees(rt["mean_angle"]) % 360)

    return results


def plot_escape_direction_rose(
    circ_stats: dict,
    figsize: tuple[float, float] = (5, 5),
) -> plt.Figure:
    """
    Polar rose plot with KDE overlay and mean vector arrow.
    One subplot per condition.
    """
    if not circ_stats:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
        ax.set_title("No escape direction data")
        return fig

    conditions = sorted(circ_stats.keys())
    n_conds = len(conditions)

    fig, axes = plt.subplots(1, n_conds, figsize=(figsize[0] * n_conds, figsize[1]),
                             subplot_kw={"projection": "polar"},
                             squeeze=False)
    axes = axes[0]

    for idx, cond in enumerate(conditions):
        ax = axes[idx]
        data = circ_stats[cond]
        angles = data["angles"]
        rt = data["rayleigh"]

        color = _resolve_color(cond)

        # Rose plot (histogram on polar axes)
        n_bins = 36
        bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)
        counts, _ = np.histogram(angles, bins=bin_edges)
        # Normalize so max bar = 0.8 of radius
        if counts.max() > 0:
            counts_norm = counts / counts.max() * 0.8
        else:
            counts_norm = counts
        width = 2 * np.pi / n_bins
        ax.bar(bin_edges[:-1], counts_norm, width=width, color=color,
               alpha=0.4, edgecolor=color, linewidth=0.5, bottom=0)

        # KDE overlay
        theta_kde = data["kde_theta"]
        density_kde = data["kde_density"]
        if density_kde.max() > 0:
            density_norm = density_kde / density_kde.max() * 0.8
        else:
            density_norm = density_kde
        ax.plot(theta_kde, density_norm, color=color, lw=1.5, alpha=0.9)

        # Mean vector arrow
        if not np.isnan(rt["mean_angle"]):
            arrow_len = rt["R"] * 0.75  # scale by resultant length
            ax.annotate(
                "", xy=(rt["mean_angle"], arrow_len), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2),
            )

        # Formatting
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["", "", "", ""], fontsize=5)
        ax.set_rlabel_position(45)

        # Title with stats
        sig = "***" if rt["p"] < 0.001 else "**" if rt["p"] < 0.01 else "*" if rt["p"] < 0.05 else "n.s."
        mean_deg = np.degrees(rt["mean_angle"]) % 360
        ax.set_title(f"{cond}\nR={rt['R']:.3f} p={rt['p']:.4f} {sig}\n"
                     f"n={data['n']} mean={mean_deg:.0f}°",
                     fontsize=7, pad=15)

    fig.tight_layout(pad=2.0)
    return fig


# ══════════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════

def run_stats_pipeline(
    master_path: str | Path,
    output_dir: str | Path | None = None,
) -> tuple[dict, dict]:
    """
    Full statistical analysis pipeline.

    Parameters
    ----------
    master_path : path to population_master.csv
    output_dir : directory to save figures

    Returns
    -------
    lmm_result : dict with GEE model and escape rates
    circ_result : dict with circular statistics per condition
    """
    master = pd.read_csv(master_path)
    log.info("Loaded population_master.csv: %d trials, %d subjects",
             len(master), master["subject_id"].nunique() if "subject_id" in master.columns else 0)

    # 1. GEE for escape probability
    lmm_result = run_escape_lmm(master)

    # 2. Circular statistics
    circ_result = run_circular_stats(master)

    # 3. Generate and save plots
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        fig_esc = plot_escape_probability(master, lmm_result)
        fig_esc.savefig(out / "escape_probability_lmm.png", dpi=300, bbox_inches="tight")
        log.info("Saved: %s", out / "escape_probability_lmm.png")
        plt.close(fig_esc)

        fig_rose = plot_escape_direction_rose(circ_result)
        fig_rose.savefig(out / "escape_direction_rose.png", dpi=300, bbox_inches="tight")
        log.info("Saved: %s", out / "escape_direction_rose.png")
        plt.close(fig_rose)

    return lmm_result, circ_result


# ══════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ══════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Population statistics — GEE escape probability & circular direction analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Path to population_master.csv")
    p.add_argument("--output", default=".", help="Output directory for figures (default: current dir)")
    return p


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    lmm_result, circ_result = run_stats_pipeline(
        master_path=args.input,
        output_dir=args.output,
    )
    return lmm_result, circ_result


if __name__ == "__main__":
    main()
