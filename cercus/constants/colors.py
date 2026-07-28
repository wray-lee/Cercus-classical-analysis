"""
Cercus Framework — Color Constants
===================================
Centralized color palette (Lancet / Cell / NPG style).

⚠️  These values are now loaded from config/defaults/colors.yaml
    and can be overridden in the project root config.yaml.
"""

from __future__ import annotations

from cercus.config import get_colors

# ═══════════════════════════════════════════════════════════════════════
# Per-Stimulus Side
# ═══════════════════════════════════════════════════════════════════════

_COLOR_LEFT = get_colors().left
COLOR_LEFT: str = str(_COLOR_LEFT)

_COLOR_RIGHT = get_colors().right
COLOR_RIGHT: str = str(_COLOR_RIGHT)

_COLOR_CONTROL = get_colors().control
COLOR_CONTROL: str = str(_COLOR_CONTROL)

# ═══════════════════════════════════════════════════════════════════════
# Stimulus Waveform Channels
# ═══════════════════════════════════════════════════════════════════════

_COLOR_OSCI_VIS = get_colors().osci_vis
COLOR_OSCI_VIS: str = str(_COLOR_OSCI_VIS)

_COLOR_OSCI_HW = get_colors().osci_hw
COLOR_OSCI_HW: str = str(_COLOR_OSCI_HW)

# ═══════════════════════════════════════════════════════════════════════
# Response Type
# ═══════════════════════════════════════════════════════════════════════

_COLOR_ESCAPE = get_colors().escape
COLOR_ESCAPE: str = str(_COLOR_ESCAPE)

_COLOR_PREWALK = get_colors().prewalk
COLOR_PREWALK: str = str(_COLOR_PREWALK)

_COLOR_NO_RESPONSE = get_colors().no_response
COLOR_NO_RESPONSE: str = str(_COLOR_NO_RESPONSE)

# ═══════════════════════════════════════════════════════════════════════
# Stillness Binary Pair
# ═══════════════════════════════════════════════════════════════════════

_COLOR_WITH_STILLNESS = get_colors().with_stillness
COLOR_WITH_STILLNESS: str = str(_COLOR_WITH_STILLNESS)

_COLOR_NO_STILLNESS = get_colors().no_stillness
COLOR_NO_STILLNESS: str = str(_COLOR_NO_STILLNESS)

# ═══════════════════════════════════════════════════════════════════════
# NPG (Nature Publishing Group) Colour Palette
# ═══════════════════════════════════════════════════════════════════════

_NPG_PALETTE = get_colors().npg_palette
NPG_PALETTE: list[str] = list(_NPG_PALETTE)

# ═══════════════════════════════════════════════════════════════════════
# Backward Compatibility Aliases
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    "COLOR_LEFT",
    "COLOR_RIGHT",
    "COLOR_CONTROL",
    "COLOR_OSCI_VIS",
    "COLOR_OSCI_HW",
    "COLOR_ESCAPE",
    "COLOR_PREWALK",
    "COLOR_NO_RESPONSE",
    "COLOR_WITH_STILLNESS",
    "COLOR_NO_STILLNESS",
    "NPG_PALETTE",
]
