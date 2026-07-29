"""
Cercus Framework — Polar & Angle Distribution Plots
===================================================
"""

from __future__ import annotations

import logging

import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy.stats import circmean, gaussian_kde

from pipeline.constants import (
    COLOR_ESCAPE,
    COLOR_PREWALK,
    _get_unified_side,
)
from cercus.visualization._circstats import (
    bootstrap_dR_ci,
    rayleigh_p,
    wallraff_test,
    watson_williams_test,
)
from cercus.visualization._core import (
    build_angular_peak_dz_mask,
    body_to_traj,
    compute_trajectory_masks,
)

log = logging.getLogger(__name__)


def plot_escape_angle_distribution(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (6.0, 4.0),
    bins: int = 36,
) -> plt.Figure:
    """Histogram + KDE of final escape angles for Escape and PreWalk trials."""
    group_cols = (
        ["subject_id", "global_trial_id"]
        if "subject_id" in df.columns
        else ["global_trial_id"]
    )
    escape_types = ["Escape", "PreWalk"]
    df_esc = df[df["response_type"].isin(escape_types)].copy()

    if df_esc.empty:
        log.warning("No Escape/PreWalk trials for angle distribution plot.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5, 0.5, "No Escape / PreWalk trials",
            ha="center", va="center", transform=ax.transAxes, fontsize=12,
        )
        return fig

    angles_by_type: dict[str, list[float]] = {"Escape": [], "PreWalk": []}

    for keys, grp in df_esc.groupby(group_cols):
        grp = grp.sort_values("t_rel")
        response_type = grp["response_type"].iloc[0]
        if response_type not in angles_by_type:
            continue

        _onset_ms = (
            grp["interval_onset_ms"].iloc[0]
            if "interval_onset_ms" in grp.columns
            else np.nan
        )
        _offset_ms = (
            grp["interval_offset_ms"].iloc[0]
            if "interval_offset_ms" in grp.columns
            else np.nan
        )

        result = compute_trajectory_masks(grp, _onset_ms, _offset_ms)
        if result is None:
            continue
        traj_x, traj_y, *_rest = result
        if traj_x is None or len(traj_x) < 2:
            continue

        angle_deg = float(np.degrees(np.arctan2(traj_y[-1], traj_x[-1])))
        angles_by_type[response_type].append(angle_deg)

    esc_angles = np.array(angles_by_type["Escape"])
    pw_angles = np.array(angles_by_type["PreWalk"])

    if len(esc_angles) == 0 and len(pw_angles) == 0:
        log.warning("No valid trajectory endpoints for angle distribution.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5, 0.5, "No valid escape angles",
            ha="center", va="center", transform=ax.transAxes, fontsize=12,
        )
        return fig

    fig, ax = plt.subplots(figsize=figsize)
    bin_edges = np.linspace(-180, 180, bins + 1)
    x_kde = np.linspace(-180, 180, 300)

    color_esc = COLOR_ESCAPE
    color_pw = COLOR_PREWALK

    fc_esc = mcolors.to_rgba(color_esc, 0.35)
    fc_pw = mcolors.to_rgba(color_pw, 0.35)

    ax.axvline(0, color="#9CA3AF", linestyle="--", linewidth=1.0, alpha=0.5, zorder=0)

    if len(esc_angles) > 0:
        ax.hist(
            esc_angles,
            bins=bin_edges,
            density=True,
            facecolor=fc_esc,
            edgecolor=color_esc,
            linewidth=1.2,
            label=f"Escape (n={len(esc_angles)})",
            zorder=1,
        )
        if len(esc_angles) > 1:
            kde_esc = gaussian_kde(esc_angles, bw_method="scott")
            ax.plot(x_kde, kde_esc(x_kde), color=color_esc, lw=2.5, alpha=0.9, zorder=3)

    if len(pw_angles) > 0:
        ax.hist(
            pw_angles,
            bins=bin_edges,
            density=True,
            facecolor=fc_pw,
            edgecolor=color_pw,
            linewidth=1.2,
            label=f"PreWalk (n={len(pw_angles)})",
            zorder=1,
        )
        if len(pw_angles) > 1:
            kde_pw = gaussian_kde(pw_angles, bw_method="scott")
            ax.plot(x_kde, kde_pw(x_kde), color=color_pw, lw=2.5, alpha=0.9, zorder=3)

    ax.set_xlabel("Escape Angle (°)")
    ax.set_ylabel("Probability Density")
    ax.set_title("Escape Angle Distribution", fontweight="bold")
    ax.set_xlim(-180, 180)
    ax.set_xticks(np.arange(-180, 181, 45))
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    ax.yaxis.grid(True, color="#E5E7EB", linestyle="-", linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)

    fig.tight_layout(pad=1.0)
    return fig


def plot_population_polar_histogram(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (6.0, 4.8),
    bins: int = 36,
    comparison_mode: bool = False,
    modalities: list[str] | None = None,
) -> plt.Figure:
    """360° polar rose of population escape-direction dispersion.

    Parameters
    ----------
    comparison_mode : bool
        If True, forces unified angular_peak trajectory preset across all
        modalities for fair cross-modal comparison. Requires df with 'modality' column.
    modalities : list[str] | None
        Modalities to compare (default ['V', 'M', 'VM']).
    """
    if comparison_mode:
        return _plot_comparison_polar(df, figsize, bins, modalities or ["V", "M", "VM"])

    # ── Original population mode: Escape vs PreWalk ──
    group_cols = (
        ["subject_id", "global_trial_id"]
        if "subject_id" in df.columns
        else ["global_trial_id"]
    )
    escape_types = ["Escape", "PreWalk"]
    df_esc = df[df["response_type"].isin(escape_types)].copy()

    angles_by_type: dict[str, list[float]] = {"Escape": [], "PreWalk": []}

    for keys, grp in df_esc.groupby(group_cols):
        grp = grp.sort_values("t_rel")
        response_type = grp["response_type"].iloc[0]

        t_vals = grp["t_rel"].values
        _onset_ms = (
            grp["interval_onset_ms"].iloc[0]
            if "interval_onset_ms" in grp.columns
            else np.nan
        )
        _offset_ms = (
            grp["interval_offset_ms"].iloc[0]
            if "interval_offset_ms" in grp.columns
            else np.nan
        )

        result = compute_trajectory_masks(grp, _onset_ms, _offset_ms)
        if result is None:
            continue
        traj_x, traj_y, *_rest = result
        if traj_x is None or len(traj_x) < 2:
            continue

        ss = _get_unified_side(grp)
        if ss == "left":
            traj_x = -traj_x

        angle_rad = float(np.arctan2(traj_x[-1], traj_y[-1]))
        angles_by_type[response_type].append(angle_rad)

    esc = np.asarray(angles_by_type["Escape"])
    pw = np.asarray(angles_by_type["PreWalk"])

    if esc.size == 0 and pw.size == 0:
        log.warning("No valid trials for polar direction histogram.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5, 0.5, "No valid escape trials",
            ha="center", va="center", transform=ax.transAxes, fontsize=12, color="0.5",
        )
        return fig

    F_ww, p_ww = np.nan, np.nan
    if esc.size >= 2 and pw.size >= 2:
        F_ww, p_ww = watson_williams_test(esc, pw)

    # Wallraff test for concentration difference (non-parametric)
    wallraff = None
    dR_obs, dR_lo, dR_hi = np.nan, np.nan, np.nan
    if esc.size >= 2 and pw.size >= 2:
        wallraff = wallraff_test(esc, pw)
        dR_obs, dR_lo, dR_hi = bootstrap_dR_ci(esc, pw, n_boot=10000)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="polar")

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetagrids(
        [0, 90, 180, 270],
        ["0°", "90°\nIpsi", "±180°", "−90°\nContra"],
        fontsize=9,
        color="0.2",
    )
    ax.tick_params(axis="x", pad=6)

    ax.grid(True, color="0.88", linewidth=0.7, linestyle="-")
    ax.set_axisbelow(True)

    bin_edges = np.linspace(-np.pi, np.pi, bins + 1)
    bin_width = 2 * np.pi / bins

    series = [
        (name, angles, color)
        for name, angles, color in [
            ("Escape", esc, COLOR_ESCAPE),
            ("PreWalk", pw, COLOR_PREWALK),
        ]
        if angles.size > 0
    ]

    radii_by_name: dict[str, np.ndarray] = {}
    for name, angles, _color in series:
        counts, _ = np.histogram(angles, bins=bin_edges)
        radii_by_name[name] = counts / angles.size * 100.0
    rmax = max(r.max() for r in radii_by_name.values()) * 1.30

    legend_handles: list[Patch] = []
    mu_by_name: dict[str, float] = {}
    for name, angles, color in series:
        ax.bar(
            bin_edges[:-1],
            radii_by_name[name],
            width=bin_width,
            align="edge",
            facecolor=mcolors.to_rgba(color, 0.30),
            edgecolor="white",
            linewidth=0.7,
            zorder=2,
        )

        if angles.size > 1 and np.std(angles) > 1e-6:
            kde = gaussian_kde(angles, bw_method="scott")
            x_kde = np.linspace(-np.pi, np.pi, 361)
            dens = kde(x_kde) + kde(x_kde - 2 * np.pi) + kde(x_kde + 2 * np.pi)
            dens[0] = dens[-1] = 0.5 * (dens[0] + dens[-1])
            ax.plot(
                x_kde,
                dens * bin_width * 100.0,
                color=color,
                lw=2.0,
                solid_capstyle="round",
                zorder=4,
            )

        mu = float(circmean(angles, high=np.pi, low=-np.pi))
        mu_by_name[name] = mu
        resultant = float(np.abs(np.exp(1j * angles).mean()))
        p_rayleigh = rayleigh_p(resultant, angles.size)
        arrow = ax.annotate(
            "",
            xy=(mu, resultant * rmax),
            xytext=(mu, 0.03 * rmax),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=1.4,
                mutation_scale=10,
                shrinkA=0,
                shrinkB=0,
                connectionstyle="arc3,rad=0",
            ),
            zorder=5,
        )
        arrow.arrow_patch.set_path_effects(
            [path_effects.withStroke(linewidth=2.6, foreground="white")]
        )

        p_rayleigh_str = "p < 1e-15" if p_rayleigh < 1e-15 else f"p = {p_rayleigh:.2e}"
        legend_handles.append(
            Patch(
                facecolor=mcolors.to_rgba(color, 0.30),
                edgecolor=color,
                linewidth=1.2,
                label=f"{name} (n = {angles.size})\nμ = {np.degrees(mu):.0f}°, R = {resultant:.2f}\n{p_rayleigh_str}",
            )
        )

    ax.set_ylim(0, rmax)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="lower"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}%"))
    combined = np.sum(
        np.stack([radii_by_name[name] for name, _a, _c in series]), axis=0
    )
    spokes = np.arange(0, 360, 45)
    spoke_bins = [
        min(int(((np.deg2rad(s) + np.pi) % (2 * np.pi)) / bin_width), bins - 1)
        for s in spokes
    ]
    spoke_mass = [combined[max(b - 1, 0) : b + 2].sum() for b in spoke_bins]
    ax.set_rlabel_position(float(spokes[int(np.argmin(spoke_mass))]))
    ax.tick_params(axis="y", labelsize=7, colors="0.45", pad=1)
    for lbl in ax.get_yticklabels():
        lbl.set_path_effects(
            [path_effects.withStroke(linewidth=2.0, foreground="white")]
        )

    ax.set_title("Escape Direction Distribution", fontweight="bold", pad=22)

    if wallraff is not None and "Escape" in mu_by_name and "PreWalk" in mu_by_name:
        mu_esc = mu_by_name["Escape"]
        mu_pw = mu_by_name["PreWalk"]
        delta_mu = abs(np.degrees(mu_esc) - np.degrees(mu_pw))
        delta_mu = min(delta_mu, 360 - delta_mu)
        ns_ww = " (ns)" if p_ww > 0.05 else ""

        # Wallraff: if p<0.05, equal-kappa assumption of Watson-Williams is violated
        # (ponytail: WW is robust to moderate kappa violations; switch to
        # Watson-Wheeler only if Wallraff p<0.01 and ΔR>0.2)
        wr_p = wallraff["p"]
        wr_sig = "ns" if wr_p > 0.05 else "**" if wr_p < 0.01 else "*"
        conc_text = (
            f"Wallraff U={wallraff['U']:.0f}, p={wr_p:.3g}{wr_sig} | "
            f"ΔR̄={dR_obs:.2f}, 95%CI [{dR_lo:.2f}, {dR_hi:.2f}] (bootstrap 10k)"
        )
        ww_text = (
            f"Watson-Williams: Δμ={delta_mu:.0f}°, F={F_ww:.2f}, p={p_ww:.2g}{ns_ww}"
        )
        ax.text(
            0.5, -0.16, ww_text, transform=ax.transAxes,
            ha="center", fontsize=7.5, color="0.35",
        )
        ax.text(
            0.5, -0.22, conc_text, transform=ax.transAxes,
            ha="center", fontsize=7.5, color="0.35",
        )

    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
        frameon=False,
        fontsize=8,
        handlelength=1.4,
        labelspacing=0.6,
    )

    return fig


def _plot_comparison_polar(
    df: pd.DataFrame,
    figsize: tuple[float, float],
    bins: int,
    modalities: list[str],
) -> plt.Figure:
    """Cross-modal comparison polar plot using unified angular_peak preset."""
    # Determine modality column
    mod_col = "modality" if "modality" in df.columns else (
        "type" if "type" in df.columns else "response_type"
    )
    present = [m for m in modalities if m in df[mod_col].unique()]
    if not present:
        log.warning("No requested modalities found in column '%s'", mod_col)
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No matching modalities", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="0.5")
        return fig

    group_cols = (
        ["subject_id", "global_trial_id"]
        if "subject_id" in df.columns
        else ["global_trial_id"]
    )

    # Modality colors
    import matplotlib.colors as mcolors
    mod_colors = {"V": "#ED0000", "M": "#00468B", "VM": "#009E73"}
    for m in modalities:
        if m not in mod_colors:
            mod_colors[m] = "#E69F00"

    angles_by_mod: dict[str, list[float]] = {m: [] for m in present}
    n_by_mod: dict[str, int] = {m: 0 for m in present}
    escape_n_by_mod: dict[str, int] = {m: 0 for m in present}

    for mod in present:
        sub = df[df[mod_col] == mod]
        for keys, grp in sub.groupby(group_cols):
            grp = grp.sort_values("t_rel")
            _onset_ms = (
                grp["interval_onset_ms"].iloc[0]
                if "interval_onset_ms" in grp.columns
                else np.nan
            )
            _offset_ms = (
                grp["interval_offset_ms"].iloc[0]
                if "interval_offset_ms" in grp.columns
                else np.nan
            )
            n_by_mod[mod] += 1
            # Check if this is an Escape trial
            rt = grp["response_type"].iloc[0] if "response_type" in grp.columns else ""
            if rt == "Escape":
                escape_n_by_mod[mod] += 1
            else:
                continue  # only escape trials for direction analysis

            # Use unified angular_peak preset: mask_z = angular_peak, macro_yaw_override=None
            t_vals = grp["t_rel"].values
            if pd.notna(_onset_ms) and pd.notna(_offset_ms):
                onset_idx = int(np.argmin(np.abs(t_vals - _onset_ms)))
                offset_idx = int(np.argmin(np.abs(t_vals - _offset_ms)))
                if onset_idx >= offset_idx:
                    offset_idx = min(onset_idx + 1, len(t_vals) - 1)

                # Build angular_peak mask
                av = (grp["angular_velocity"].values
                      if "angular_velocity" in grp.columns else None)
                if av is not None:
                    mask_z = build_angular_peak_dz_mask(av, onset_idx, offset_idx, len(t_vals))
                else:
                    mask_z = np.zeros(len(t_vals), dtype=bool)
                    mask_z[onset_idx:offset_idx] = True

                mask_xy = np.zeros(len(t_vals), dtype=bool)
                mask_xy[onset_idx:offset_idx] = True
            else:
                continue

            result = body_to_traj(
                grp, mask_xy, use_z=True, use_rigid_rotation=True,
                mask_z=mask_z, macro_yaw_override=None,
            )
            if result is None:
                continue
            traj_x, traj_y = result
            if traj_x is None or len(traj_x) < 2:
                continue

            ss = _get_unified_side(grp)
            if ss == "left":
                traj_x = -traj_x

            angle_rad = float(np.arctan2(traj_x[-1], traj_y[-1]))
            angles_by_mod[mod].append(angle_rad)

    # Filter out empty modalities
    angles_by_mod = {m: np.asarray(a) for m, a in angles_by_mod.items() if a}
    if not angles_by_mod:
        log.warning("No valid escape angles for any modality.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No valid escape angles",
                ha="center", va="center", transform=ax.transAxes, fontsize=12, color="0.5")
        return fig

    # ── Plot ──
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetagrids([0, 90, 180, 270], ["0°", "90°\nIpsi", "±180°", "−90°\nContra"], fontsize=9, color="0.2")
    ax.tick_params(axis="x", pad=6)
    ax.grid(True, color="0.88", linewidth=0.7, linestyle="-")
    ax.set_axisbelow(True)

    bin_edges = np.linspace(-np.pi, np.pi, bins + 1)
    bin_width = 2 * np.pi / bins

    radii_by_mod: dict[str, np.ndarray] = {}
    for mod, angles in angles_by_mod.items():
        counts, _ = np.histogram(angles, bins=bin_edges)
        radii_by_mod[mod] = counts / angles.size * 100.0
    rmax = max(r.max() for r in radii_by_mod.values()) * 1.30

    legend_handles: list[Patch] = []
    mu_by_mod: dict[str, float] = {}
    for mod in present:
        if mod not in angles_by_mod:
            continue
        angles = angles_by_mod[mod]
        color = mod_colors.get(mod, "#E69F00")

        ax.bar(bin_edges[:-1], radii_by_mod[mod], width=bin_width, align="edge",
               facecolor=mcolors.to_rgba(color, 0.30), edgecolor="white", linewidth=0.7, zorder=2)

        if angles.size > 1 and np.std(angles) > 1e-6:
            kde = gaussian_kde(angles, bw_method="scott")
            x_kde = np.linspace(-np.pi, np.pi, 361)
            dens = kde(x_kde) + kde(x_kde - 2 * np.pi) + kde(x_kde + 2 * np.pi)
            dens[0] = dens[-1] = 0.5 * (dens[0] + dens[-1])
            ax.plot(x_kde, dens * bin_width * 100.0, color=color, lw=2.0, zorder=4)

        mu = float(circmean(angles, high=np.pi, low=-np.pi))
        mu_by_mod[mod] = mu
        resultant = float(np.abs(np.exp(1j * angles).mean()))
        p_ray = rayleigh_p(resultant, angles.size)

        arrow = ax.annotate("", xy=(mu, resultant * rmax), xytext=(mu, 0.03 * rmax),
                            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4, mutation_scale=10,
                                            shrinkA=0, shrinkB=0, connectionstyle="arc3,rad=0"), zorder=5)
        arrow.arrow_patch.set_path_effects([path_effects.withStroke(linewidth=2.6, foreground="white")])

        p_str = "p < 1e-15" if p_ray < 1e-15 else f"p = {p_ray:.2e}"
        legend_handles.append(Patch(facecolor=mcolors.to_rgba(color, 0.30), edgecolor=color, linewidth=1.2,
                                    label=f"{mod} (n={angles.size}) μ={np.degrees(mu):.0f}° R={resultant:.2f} {p_str}"))

    ax.set_ylim(0, rmax)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="lower"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}%"))

    # Label position
    combined = np.sum(np.stack(list(radii_by_mod.values())), axis=0)
    spokes = np.arange(0, 360, 45)
    spoke_bins = [min(int(((np.deg2rad(s) + np.pi) % (2 * np.pi)) / bin_width), bins - 1) for s in spokes]
    spoke_mass = [combined[max(b - 1, 0): b + 2].sum() for b in spoke_bins]
    ax.set_rlabel_position(float(spokes[int(np.argmin(spoke_mass))]))
    ax.tick_params(axis="y", labelsize=7, colors="0.45", pad=1)
    for lbl in ax.get_yticklabels():
        lbl.set_path_effects([path_effects.withStroke(linewidth=2.0, foreground="white")])

    ax.set_title("Cross-Modal Escape Direction", fontweight="bold", pad=22)

    # Pairwise stats: V vs VM, M vs VM
    pairwise_lines = []
    for ref_mod, target_mod in [("V", "VM"), ("M", "VM")]:
        if ref_mod not in angles_by_mod or target_mod not in angles_by_mod:
            continue
        a1 = angles_by_mod[ref_mod]
        a2 = angles_by_mod[target_mod]
        if len(a1) < 2 or len(a2) < 2:
            continue
        F_ww, p_ww = watson_williams_test(a1, a2)
        wr = wallraff_test(a1, a2)
        dR_obs, dR_lo, dR_hi = bootstrap_dR_ci(a1, a2, n_boot=10000)
        delta_mu = abs(np.degrees(mu_by_mod[ref_mod]) - np.degrees(mu_by_mod[target_mod]))
        delta_mu = min(delta_mu, 360 - delta_mu)
        ns_ww = "ns" if p_ww > 0.05 else "*"
        wr_sig = "ns" if wr["p"] > 0.05 else "**" if wr["p"] < 0.01 else "*"
        pairwise_lines.append(
            f"{ref_mod} vs {target_mod}: Δμ={delta_mu:.0f}° F={F_ww:.2f} p={p_ww:.2g}{ns_ww} | "
            f"Wallraff U={wr['U']:.0f} p={wr['p']:.3g}{wr_sig} ΔR̄={dR_obs:.2f} 95%CI[{dR_lo:.2f},{dR_hi:.2f}]"
        )

    # MSI
    if "V" in angles_by_mod and "M" in angles_by_mod and "VM" in angles_by_mod:
        R_V = float(np.abs(np.exp(1j * angles_by_mod["V"]).mean()))
        R_M = float(np.abs(np.exp(1j * angles_by_mod["M"]).mean()))
        R_VM = float(np.abs(np.exp(1j * angles_by_mod["VM"]).mean()))
        MSI_R = (R_VM - max(R_V, R_M)) / max(R_V, R_M) * 100

        P_V = escape_n_by_mod.get("V", 0) / max(n_by_mod.get("V", 1), 1)
        P_M = escape_n_by_mod.get("M", 0) / max(n_by_mod.get("M", 1), 1)
        P_VM = escape_n_by_mod.get("VM", 0) / max(n_by_mod.get("VM", 1), 1)
        MSI_prob = (P_VM - max(P_V, P_M)) / max(P_V, P_M) * 100

        msi_line = f"MSI_R={MSI_R:.0f}%  MSI_prob={MSI_prob:.0f}%"
        pairwise_lines.append(msi_line)

    if pairwise_lines:
        ax.text(0.5, -0.18, "\n".join(pairwise_lines), transform=ax.transAxes,
                ha="center", fontsize=7, color="0.35", va="top")

    # Legend inside plot
    fig.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.99, 0.99),
               frameon=False, fontsize=8, handlelength=1.4, labelspacing=0.6)

    return fig
