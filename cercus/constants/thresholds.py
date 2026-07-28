"""
Cercus Framework — Threshold Constants
=======================================
Physical thresholds for escape detection and classification.
"""

# ── Physical-Threshold Constants ──
ESCAPE_VMAX_THRESHOLD: float = 50.0    # mm/s — burst floor for valid escape
ESCAPE_START_THRESHOLD: float = 10.0   # mm/s — latency onset anchor
PREWALK_THRESHOLD: float = 10.0        # mm/s — pre-stimulus spontaneous activity
PREWALK_WINDOW_MS: float = 1000.0      # ms — pre-stimulus validation window
POST_STIM_BUFFER_MS: float = 50.0      # ms — post-stimulus tail buffer
ESCAPE_WINDOW_MS: float = 250.0        # ms — post-stimulus burst detection window