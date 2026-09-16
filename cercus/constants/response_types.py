"""
Cercus Framework — Response Type Registry
=========================================
Single source of truth for behavior categories and their colors.
When PreEscape is disabled (``classification.use_preescape: false``) it is
excluded from the tuple, so every downstream figure/metric follows the switch
without per-file branches.
"""

from __future__ import annotations

from cercus.constants.colors import (
    COLOR_ESCAPE,
    COLOR_NO_RESPONSE,
    COLOR_PRE_ESCAPE,
    COLOR_PREWALK,
)
from cercus.constants.thresholds import USE_PRE_ESCAPE

_ESCAPE_TYPES = ("Escape", "PreEscape", "PreWalk", "NoResponse")
_TERNARY_TYPES = ("Escape", "PreWalk", "NoResponse")

#: Display/priority order used by all population figures.
RESPONSE_TYPES: tuple[str, ...] = _ESCAPE_TYPES if USE_PRE_ESCAPE else _TERNARY_TYPES

RESPONSE_COLORS: dict[str, str] = {
    "Escape": COLOR_ESCAPE,
    "PreEscape": COLOR_PRE_ESCAPE,
    "PreWalk": COLOR_PREWALK,
    "NoResponse": COLOR_NO_RESPONSE,
}

__all__ = ["RESPONSE_TYPES", "RESPONSE_COLORS", "COLOR_PRE_ESCAPE"]
