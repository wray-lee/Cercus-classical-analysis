"""
Cercus Framework — Threshold Constants
=======================================
Physical thresholds for escape detection and classification.

⚠️  These values are now loaded from config/defaults/thresholds.yaml
    and can be overridden in the project root config.yaml.
"""

from __future__ import annotations

from cercus.config import get_thresholds

# ═══════════════════════════════════════════════════════════════════════
# Escape Detection Thresholds
# ═══════════════════════════════════════════════════════════════════════

_ESCAPE_START_THRESHOLD = get_thresholds().escape.start_threshold
ESCAPE_START_THRESHOLD: float = float(_ESCAPE_START_THRESHOLD)

_ESCAPE_VMAX_THRESHOLD = get_thresholds().escape.vmax_threshold
ESCAPE_VMAX_THRESHOLD: float = float(_ESCAPE_VMAX_THRESHOLD)

_ESCAPE_WINDOW_MS = get_thresholds().escape.window_ms
ESCAPE_WINDOW_MS: float = float(_ESCAPE_WINDOW_MS)

# ═══════════════════════════════════════════════════════════════════════
# PreWalk Detection Thresholds
# ═══════════════════════════════════════════════════════════════════════

_PREWALK_THRESHOLD = get_thresholds().prewalk.threshold
PREWALK_THRESHOLD: float = float(_PREWALK_THRESHOLD)

_PREWALK_WINDOW_MS = get_thresholds().prewalk.window_ms
PREWALK_WINDOW_MS: float = float(_PREWALK_WINDOW_MS)

# ═══════════════════════════════════════════════════════════════════════
# Buffer Settings
# ═══════════════════════════════════════════════════════════════════════

_POST_STIM_BUFFER_MS = get_thresholds().post_stim.buffer_ms
POST_STIM_BUFFER_MS: float = float(_POST_STIM_BUFFER_MS)

# ═══════════════════════════════════════════════════════════════════════
# Backward Compatibility Aliases
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    "ESCAPE_START_THRESHOLD",
    "ESCAPE_VMAX_THRESHOLD",
    "ESCAPE_WINDOW_MS",
    "PREWALK_THRESHOLD",
    "PREWALK_WINDOW_MS",
    "POST_STIM_BUFFER_MS",
]
