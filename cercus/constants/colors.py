"""
Cercus Framework — Color Constants
===================================
Centralized color palette (Lancet / Cell / NPG style).
"""

# ── Per-Stimulus Side ──
COLOR_LEFT: str = "#00468B"          # Navy Blue
COLOR_RIGHT: str = "#ED0000"         # Crimson Red
COLOR_CONTROL: str = "#7C878E"       # Slate Grey

# ── Stimulus Waveform Channels ──
COLOR_OSCI_VIS: str = "#ADB6B6"      # Cool Grey (visual stimulus background)
COLOR_OSCI_HW: str = "#E69F00"       # Sand Orange (hardware stimulus background)

# ── Response Type ──
COLOR_ESCAPE: str = "#ED0000"
COLOR_PREWALK: str = "#00468B"
COLOR_NO_RESPONSE: str = "#7C878E"

# ── Stillness Binary Pair ──
COLOR_WITH_STILLNESS: str = COLOR_PREWALK      # deep navy — the signal state
COLOR_NO_STILLNESS: str = "#5B6770"            # neutral rock-grey — absence

# ── NPG (Nature Publishing Group) Colour Palette ──
# Colour-blind friendly, widely used in top-tier journals.
NPG_PALETTE: list[str] = [
    "#E64B35",   # Red
    "#4DBBD5",   # Cyan
    "#00A087",   # Teal
    "#3C5488",   # Navy Blue
    "#F39B7F",   # Salmon
    "#8491B4",   # Slate Blue
    "#91D1C2",   # Mint
    "#DC0000",   # Dark Red
]