"""
Cercus Framework — Reaction Time & Escape Distance
==================================================
Pure physics: stimulus-anchored reaction time and path distance from the
speed trace.  **No matplotlib dependency.**

反应时与逃逸行程——纯运动学测量，不做 matplotlib 导入。

Definitions (see CONTEXT.md):
- ``reaction_time_ms``: ``interval_onset_ms − anchor`` on the ``t_rel`` axis.
  Multimodal wind trials anchor at wind onset (``target_ttc_ms``); all other
  paradigms anchor at ``t_rel = 0`` (TTC or wind onset, whichever the axis was
  aligned to).  Negative ⇒ the animal started *before* the trigger stimulus
  (pure-vision PreEscape lead time).
- ``distance_mm``: trapezoidal integral of ``speed`` (mm/s) over the escape
  interval ``[interval_onset_ms, interval_offset_ms]`` — "one escape, how far".
- ``distance_500ms_mm``: same integral clipped to ``[onset, onset + window]``.
"""

from __future__ import annotations

import numpy as np

_DEFAULT_POST_ONSET_WINDOW_MS = 500.0  # ms — second distance window


def compute_reaction_and_distance(
    t_rel: np.ndarray,
    speed: np.ndarray,
    interval_onset_ms: float,
    interval_offset_ms: float,
    anchor_ms: float,
    post_onset_window_ms: float = _DEFAULT_POST_ONSET_WINDOW_MS,
) -> dict[str, float]:
    """Reaction time + escape distance from a trial's speed trace.

    NaN in ⇒ NaN out for every affected metric (NoResponse convention).
    """
    out = {"reaction_time_ms": np.nan, "distance_mm": np.nan,
           "distance_500ms_mm": np.nan}

    if not (np.isfinite(interval_onset_ms) and np.isfinite(anchor_ms)):
        return out
    out["reaction_time_ms"] = float(interval_onset_ms - anchor_ms)

    t = np.asarray(t_rel, dtype=float)
    s = np.asarray(speed, dtype=float)
    keep = np.isfinite(t) & np.isfinite(s)
    t, s = t[keep], s[keep]
    if len(t) < 2:
        return out

    def _integrate(lo: float, hi: float) -> float:
        # 梯形积分：mm/s × ms / 1000 → mm；窗边界线性插值补齐
        # （np.interp 会钳位，窗完全在数据外时必须显式 NaN）
        if hi <= t[0] or lo >= t[-1]:
            return np.nan
        m = (t > lo) & (t < hi)
        tt = np.concatenate(([lo], t[m], [hi]))
        ss = np.concatenate((
            [np.interp(lo, t, s)], s[m], [np.interp(hi, t, s)],
        ))
        return float(np.trapezoid(ss, tt) / 1000.0)

    if np.isfinite(interval_offset_ms) and interval_offset_ms > interval_onset_ms:
        out["distance_mm"] = _integrate(interval_onset_ms, interval_offset_ms)
    out["distance_500ms_mm"] = _integrate(
        interval_onset_ms, interval_onset_ms + post_onset_window_ms
    )
    return out


if __name__ == "__main__":
    # 手算对照: 恒速 100 mm/s × 500 ms → 50 mm
    t = np.arange(-1000, 2000, 10.0)
    v = np.where((t >= 0) & (t <= 1000), 100.0, 0.0)
    r = compute_reaction_and_distance(t, v, 0.0, 1000.0, -373.0)
    assert abs(r["reaction_time_ms"] - 373.0) < 1e-9
    assert abs(r["distance_mm"] - 100.0) < 1e-6          # 100mm/s × 1s
    assert abs(r["distance_500ms_mm"] - 50.0) < 1e-6     # 100mm/s × 0.5s
    # NaN 传播
    r2 = compute_reaction_and_distance(t, v, np.nan, np.nan, -373.0)
    assert all(np.isnan(list(r2.values())))
    print("OK")
