"""
Cercus Framework — Multisensory Overview & Enhancement Plots
============================================================
Two figures drawn from the *pre-aggregated* multisensory summary tables:

* :func:`plot_ms_overview`    — per-condition animal escape probability,
  selected timing, and escape-interval distance (one point per animal +
  cohort mean/SD or median/IQR).
* :func:`plot_ms_enhancement` — per-comparison ``delta_best`` /
  ``delta_independence`` with asymmetric animal-bootstrap CIs.

The analysis side owns every derived quantity (probabilities, timing, distance,
deltas). This module only draws.

Cohorts are the full analysis condition (paradigm, type, TTC, l/v and initial
angle), never a paradigm-level pool of conditions — a per-condition animal
median is reported, not a mean of
condition medians passed off as an animal-level median. One point per animal.

Style is read strictly from ``visualization.multisensory`` in YAML (separate
overview/enhancement layout + footer-y keys) — no Python geometry fallbacks.
Footers are drawn inside the reserved layout band, so ``bbox_inches="tight"``
never inflates the figure.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cercus.config import get_visualization
from cercus.constants.colors import NPG_PALETTE
from cercus.constants.response_types import RESPONSE_COLORS

log = logging.getLogger(__name__)

#: Coverage statuses worth annotating on the overview (available is silent).
_ANNOTATED_STATUSES = {"empty", "missing", "partial", "failed", "unsupported"}


# ══════════════════════════════════════════════════════════════════════
# Config / table helpers
# ══════════════════════════════════════════════════════════════════════


def _ms():
    """The ``visualization.multisensory`` config proxy (strict; YAML keys only)."""
    return get_visualization().multisensory


def _as_dict(obj) -> dict:
    """Plain dict from a ConfigProxy / dict (nested layout block)."""
    if isinstance(obj, dict):
        return dict(obj)
    return obj.to_dict()


def _num(frame: pd.DataFrame, col: str) -> np.ndarray:
    if col in frame.columns:
        return pd.to_numeric(frame[col], errors="coerce").to_numpy(float)
    return np.full(len(frame), np.nan)


def _paradigm_order(coverage, subjects) -> list[str]:
    for frame in (coverage, subjects):
        if frame is not None and "paradigm" in getattr(frame, "columns", []):
            return list(dict.fromkeys(frame["paradigm"].astype(str)))
    return []


def _status_map(coverage) -> dict[str, str]:
    if coverage is None or "status" not in getattr(coverage, "columns", []):
        return {}
    return dict(zip(coverage["paradigm"].astype(str), coverage["status"].astype(str)))


#: Columns that together define an analysis condition.
_CONDITION_COLS = ("type", "target_ttc_ms", "lv_ratio_ms", "init_half_angle_deg")


def _condition_display(rows: pd.DataFrame, vary_cols: tuple[str, ...]) -> str:
    """Label for one condition from the columns that actually vary in its
    paradigm — so same-TTC conditions differing in lv/angle stay distinct."""
    parts = []
    for c in vary_cols:
        if c not in rows.columns or not len(rows):
            continue
        v = rows[c].iloc[0]
        if pd.isna(v):
            continue
        number = str(float(v)).removesuffix(".0") if c != "type" else str(v)
        if c == "target_ttc_ms":
            parts.append(f"{number} ms")
        elif c == "lv_ratio_ms":
            parts.append(f"lv {number}")
        elif c == "init_half_angle_deg":
            parts.append(f"{number}°")
        else:
            parts.append(str(v))
    return " / ".join(parts)


def _animal_values(grp: pd.DataFrame, col: str) -> np.ndarray:
    """One value per animal within one condition (first row if duplicated)."""
    if "subject_id" in grp.columns:
        grp = grp.drop_duplicates("subject_id")
    vals = _num(grp, col)
    return vals[np.isfinite(vals)]


def _summary(vals: np.ndarray, kind: str) -> tuple[float, float, float] | None:
    """(center, low, high) for the cohort marker — mean/SD or median/IQR."""
    vals = np.asarray([v for v in vals if np.isfinite(v)], float)
    if vals.size == 0:
        return None
    if kind == "mean":
        center = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        return center, center - sd, center + sd
    q1, med, q3 = np.percentile(vals, [25.0, 50.0, 75.0])
    return float(med), float(q1), float(q3)


def _ylim_from(values: np.ndarray, pad: float) -> tuple[float, float]:
    """Data-driven limits; a flat/zero/all-NaN cohort still yields a usable span."""
    ms = _ms()
    vals = np.asarray([v for v in values if np.isfinite(v)], float)
    if vals.size == 0:
        return 0.0, float(ms.min_axis_span)
    lo, hi = float(vals.min()), float(vals.max())
    if hi == lo:
        span = max(abs(hi) * float(ms.flat_axis_fraction), float(ms.min_axis_span))
        lo, hi = lo - span, hi + span
    margin = pad * (hi - lo)
    return lo - margin, hi + margin


def _pack(entries: list[str], max_chars: int) -> list[str]:
    """Pack compact entries into lines under a character budget (no overflow)."""
    lines, cur = [], ""
    for entry in entries:
        add = entry if not cur else "  " + entry
        if cur and len(cur) + len(add) > max_chars:
            lines.append(cur)
            cur = entry
        else:
            cur += add
    if cur:
        lines.append(cur)
    return lines


# ══════════════════════════════════════════════════════════════════════
# Overview figure
# ══════════════════════════════════════════════════════════════════════


def _overview_categories(subjects: pd.DataFrame, paradigms: list[str]) -> list[dict]:
    """One category per (paradigm, condition); paradigms absent from *subjects*
    still yield a single empty category so their coverage status is shown."""
    cats: list[dict] = []
    if "paradigm" not in subjects.columns:
        return [{"paradigm": p, "cond": None, "label": p, "rows": subjects.iloc[0:0]}
                for p in paradigms]
    cols = [column for column in _CONDITION_COLS if column in subjects.columns]
    for p in paradigms:
        psub = subjects[subjects["paradigm"].astype(str) == p]
        if psub.empty:
            cats.append({"paradigm": p, "cond": None, "label": p, "rows": psub})
            continue
        # Group original values; display rounding must never change membership.
        conditions = list(psub.groupby(cols, dropna=False, sort=False)) if cols else [(None, psub)]
        vary = tuple(column for column in cols if psub[column].nunique(dropna=False) > 1)
        for key, rows in conditions:
            label = f"{p}\n{_condition_display(rows, vary)}" if len(conditions) > 1 else p
            cats.append({"paradigm": p, "cond": key, "label": label, "rows": rows})
    return cats


def plot_ms_overview(
    subjects: pd.DataFrame,
    coverage: pd.DataFrame,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """3-panel multisensory overview: escape probability / timing / distance.

    x categories are (paradigm, condition) in coverage order. Each animal
    contributes one point; the cohort marker is mean±SD (probability) or
    median±IQR (timing, distance). Empty/absent categories keep their coverage
    status; a category with rows but no usable timing is marked ``RT
    unavailable`` — never a fabricated zero point.
    """
    ms = _ms()
    if figsize is None:
        figsize = tuple(ms.overview_figsize)
    layout = _as_dict(ms.overview_layout)
    footer_y = float(ms.overview_footer_y)
    pack_chars = int(ms.footer_pack_chars)
    seed = int(ms.jitter_seed)
    rot = float(ms.tick_rotation)
    psize = float(ms.point_size)
    afs = float(ms.annotation_fontsize)

    subjects = subjects if subjects is not None else pd.DataFrame()
    paradigms = _paradigm_order(coverage, subjects)
    statuses = _status_map(coverage)
    cats = _overview_categories(subjects, paradigms)
    mod_col = next((c for c in ("modality", "type") if c in subjects.columns), None)
    modalities = (
        list(dict.fromkeys(subjects[mod_col].astype(str))) if mod_col else []
    )
    mod_color = {m: NPG_PALETTE[i % len(NPG_PALETTE)] for i, m in enumerate(modalities)}

    panels = (
        ("escape_probability", "Escape probability", "P(derived Escape)", "mean",
         RESPONSE_COLORS["Escape"]),
        ("rt_median_ms", "Selected timing", "Selected timing endpoint (ms)", "median",
         NPG_PALETTE[0]),
        ("distance_median_mm", "Escape-interval distance", "distance (mm)", "median",
         NPG_PALETTE[2]),
    )

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.subplots_adjust(**layout)
    rng = np.random.default_rng(seed)
    n = len(cats)

    for ax, (col, title, ylab, kind, base_color) in zip(axes, panels):
        ax.set_title(title, loc="left", fontsize=float(ms.title_fontsize),
                     fontweight="bold", pad=float(ms.title_pad))
        collected: list[np.ndarray] = []
        for i, cat in enumerate(cats):
            rows = cat["rows"]
            vals = _animal_values(rows, col) if not rows.empty else np.array([])
            if vals.size == 0:
                continue
            if mod_col:
                modality = str(rows[mod_col].iloc[0])
                color = mod_color.get(modality, base_color)
            else:
                color = base_color
            jitter = float(ms.jitter_width)
            xs = i + rng.uniform(-jitter, jitter, vals.size)
            ax.scatter(xs, vals, s=psize, c=color, edgecolors=ms.point_edge_color,
                       linewidths=float(ms.point_edge_width), alpha=float(ms.point_alpha), zorder=4)
            summary = _summary(vals, kind)
            if summary is not None:
                cen, lo, hi = summary
                ax.errorbar([i], [cen], yerr=[[cen - lo], [hi - cen]], fmt="s",
                            color=ms.summary_color, ms=float(ms.summary_marker_size),
                            capsize=float(ms.summary_cap_size), lw=float(ms.summary_line_width),
                            markeredgecolor=ms.point_edge_color,
                            markeredgewidth=float(ms.summary_edge_width), zorder=6)
            collected.append(vals)

        # Per-category vertical annotation for empty / RT-missing conditions.
        for i, cat in enumerate(cats):
            status = statuses.get(cat["paradigm"], "")
            rows = cat["rows"]
            has_values = (not rows.empty and col in rows.columns
                          and rows[col].notna().any())
            if has_values:
                continue
            if rows.empty:
                label = status if status in _ANNOTATED_STATUSES else "n=0"
            elif col == "escape_probability":
                label = "n=0"
            elif col == "rt_median_ms":
                label = "RT unavailable"
            else:
                label = "distance unavailable"
            ax.text(i, float(ms.annotation_y), label, transform=ax.get_xaxis_transform(),
                    rotation=90, ha="center", va="bottom",
                    fontsize=afs, color=ms.annotation_color)

        ax.set_xticks(range(n))
        ax.set_xticklabels([c["label"] for c in cats], rotation=rot, ha="right", fontsize=float(ms.tick_fontsize))
        if n:
            ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylabel(ylab, fontsize=float(ms.label_fontsize))
        ax.tick_params(axis="y", labelsize=float(ms.tick_fontsize))
        ax.grid(axis="y", color=ms.grid_color, lw=float(ms.grid_line_width), alpha=float(ms.grid_alpha))
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if col == "escape_probability":
            ax.set_ylim(0.0, 1.0)
        else:
            flat = np.concatenate(collected) if collected else np.array([0.0])
            lo, hi = _ylim_from(flat, pad=float(ms.timing_axis_padding))
            ax.set_ylim(lo, hi)
            ax.axhline(0, color=ms.zero_color, ls="--", lw=float(ms.zero_line_width), zorder=1)

    if len(modalities) > 1:
        handles = [
            plt.Line2D([], [], marker="o", ls="", color=mod_color[m],
                       markeredgecolor=ms.point_edge_color,
                       markersize=float(ms.legend_marker_size), label=m)
            for m in modalities
        ]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
                   ncol=len(modalities), fontsize=float(ms.legend_fontsize), frameon=False,
                   columnspacing=float(ms.legend_columnspacing))

    footer = ["Timing endpoints: BV trials use the TTC anchor; wind trials use the "
              "calibrated airflow-onset reference. a=animals; RT=observed/missing."]
    if "paradigm" in subjects.columns:
        entries = []
        for p in paradigms:
            sub = subjects[subjects["paradigm"].astype(str) == p]
            animals = sub["subject_id"].astype(str).nunique() if "subject_id" in sub.columns else len(sub)
            obs = int(np.nansum(_num(sub, "rt_observed"))) if "rt_observed" in sub.columns else None
            miss = int(np.nansum(_num(sub, "rt_missing"))) if "rt_missing" in sub.columns else None
            rt = f"RT={obs}/{miss}" if obs is not None else "RT=-/-"
            entries.append(f"{p}: a={animals}; {rt}")
        footer += _pack(entries, pack_chars)
    fig.text(float(layout["left"]), footer_y, "\n".join(footer),
             ha="left", va="bottom", fontsize=afs, color=ms.footer_color,
             linespacing=float(ms.footer_linespacing))
    return fig


# ══════════════════════════════════════════════════════════════════════
# Enhancement figure
# ══════════════════════════════════════════════════════════════════════


def _comparison_label(row: pd.Series, keys: tuple[str, ...]) -> str:
    condition = _condition_display(row.to_frame().T, keys[1:])
    paradigm = str(row.get("paradigm", ""))
    return f"{paradigm}\n{condition}" if condition else paradigm


def _comparison_keys(comparisons: pd.DataFrame) -> tuple[str, ...]:
    """Show fields that vary within a paradigm, including l/v and initial angle."""
    keys = ["paradigm"]
    if "paradigm" in comparisons.columns:
        groups = comparisons.groupby("paradigm", dropna=False)
        keys += [
            column for column in _CONDITION_COLS
            if column in comparisons.columns
            and groups[column].nunique(dropna=False).gt(1).any()
        ]
    return tuple(keys)


def plot_ms_enhancement(
    comparisons: pd.DataFrame,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """2-panel delta_best / delta_independence with asymmetric bootstrap CIs.

    One x position per comparison row; rows with no drawable value keep their
    tick (gray ``n/a``). A row with a point but no CI is labelled
    ``CI unavailable``. The CI is drawn as its own vertical span + endcaps,
    independent of the point — an animal bootstrap of a nonlinear delta need
    not straddle the point, so the interval never goes through ``yerr``.
    """
    ms = _ms()
    if figsize is None:
        figsize = tuple(ms.enhancement_figsize)
    layout = _as_dict(ms.enhancement_layout)
    footer_y = float(ms.enhancement_footer_y)
    rot = float(ms.tick_rotation)
    psize = float(ms.point_size)
    afs = float(ms.annotation_fontsize)

    comparisons = comparisons if comparisons is not None else pd.DataFrame()
    n = len(comparisons)
    paradigms = (
        list(dict.fromkeys(comparisons["paradigm"].astype(str)))
        if "paradigm" in comparisons.columns else []
    )
    colors = {p: NPG_PALETTE[i % len(NPG_PALETTE)] for i, p in enumerate(paradigms)}
    para = comparisons["paradigm"].astype(str).to_numpy() if "paradigm" in comparisons.columns else np.array([""] * n)
    keys = _comparison_keys(comparisons)
    labels = [_comparison_label(row, keys) for _, row in comparisons.iterrows()]

    panels = (
        ("delta_best", "delta_best_ci_low", "delta_best_ci_high",
         "Δ best unimodal (any burst)"),
        ("delta_independence", "delta_independence_ci_low",
         "delta_independence_ci_high", "Δ probability independence"),
    )

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.subplots_adjust(**layout)
    for ax, (dcol, lcol, hcol, title) in zip(axes, panels):
        delta = _num(comparisons, dcol)
        ci_lo = _num(comparisons, lcol)
        ci_hi = _num(comparisons, hcol)
        ax.set_title(title, loc="left", fontsize=float(ms.title_fontsize),
                     fontweight="bold", pad=float(ms.title_pad))
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=rot, ha="right", fontsize=float(ms.tick_fontsize))
        if n:
            ax.set_xlim(-0.5, n - 0.5)
        ax.axhline(0, color=ms.zero_color, ls="--", lw=float(ms.zero_line_width), zorder=1)

        span_vals = []
        for i in range(n):
            color = colors.get(para[i], NPG_PALETTE[0])
            has_ci = np.isfinite(ci_lo[i]) and np.isfinite(ci_hi[i])
            has_point = np.isfinite(delta[i])
            if not (has_ci or has_point):
                ax.text(i, float(ms.unavailable_y), "n/a", transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=afs, color=ms.annotation_color)
                continue
            # CI drawn as an independent span + endcaps: an animal bootstrap
            # of a nonlinear delta need not straddle the point.
            if has_ci:
                cap = float(ms.ci_cap_width)
                ax.vlines(i, ci_lo[i], ci_hi[i], color=color, lw=float(ms.summary_line_width), zorder=3)
                ax.plot([i - cap, i + cap], [ci_lo[i]] * 2, color=color,
                        lw=float(ms.summary_line_width), zorder=3)
                ax.plot([i - cap, i + cap], [ci_hi[i]] * 2, color=color,
                        lw=float(ms.summary_line_width), zorder=3)
                span_vals.extend([ci_lo[i], ci_hi[i]])
            if has_point:
                ax.scatter([i], [delta[i]], c=[color], s=psize,
                           edgecolors=ms.point_edge_color,
                           linewidths=float(ms.summary_edge_width), zorder=4)
                span_vals.append(delta[i])
            if has_point and not has_ci:
                ax.text(i, float(ms.unavailable_y), "CI unavailable", transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=afs, color=ms.annotation_color)

        lo_y, hi_y = _ylim_from(np.asarray(span_vals, float), pad=float(ms.enhancement_axis_padding))
        ax.set_ylim(lo_y, hi_y)
        ax.set_ylabel("Δ probability", fontsize=float(ms.label_fontsize))
        ax.tick_params(axis="y", labelsize=float(ms.tick_fontsize))
        ax.grid(axis="y", color=ms.grid_color, lw=float(ms.grid_line_width), alpha=float(ms.grid_alpha))
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if n == 0:
        for ax in axes:
            ax.text(0.5, 0.5, "no comparisons", transform=ax.transAxes,
                    ha="center", va="center", fontsize=float(ms.title_fontsize), color=ms.annotation_color)
        note = ["No comparisons (no eligible rows or no matched baseline)."]
    else:
        note = [
            "Any-burst probability; animal bootstrap CI.",
            "Probability-independence reference (not an RT race model).",
        ]
        if "visual_match_status" in comparisons.columns:
            degraded = set(comparisons["visual_match_status"].astype(str)) - {"lv_ratio_and_angle"}
            if degraded:
                note.append("Visual baseline: initial angle unrecorded.")
    fig.text(float(layout["left"]), footer_y, "\n".join(note),
             ha="left", va="bottom", fontsize=afs, color=ms.footer_color,
             linespacing=float(ms.footer_linespacing))
    return fig


__all__ = ["plot_ms_overview", "plot_ms_enhancement"]
