"""Modern dark scientific theme for Cercus Analysis.

Centralizes all styling: QSS, color palette, typography, and
Matplotlib rcParams for seamless dark UI integration.
"""

from typing import Dict

# =============================================================================
# Color Palette
# =============================================================================

# Backgrounds (darkest → lightest)
BG_PRIMARY = "#1E1E1E"      # Main window background
BG_CARD = "#2D2D30"         # Card / panel backgrounds
BG_INPUT = "#3C3C3C"        # Input field backgrounds
BG_HOVER = "#404040"        # Hover state
BG_PRESSED = "#4A4A4A"      # Pressed state

# Accent
ACCENT = "#0A84FF"          # Primary accent (buttons, highlights)
ACCENT_HOVER = "#409CFF"    # Accent hover
ACCENT_PRESSED = "#0060CC"  # Accent pressed

# Text
TEXT_PRIMARY = "#FFFFFF"     # Primary text (white)
TEXT_SECONDARY = "#A0A0A5"  # Secondary / muted text
TEXT_DISABLED = "#5A5A5E"   # Disabled text
TEXT_ON_ACCENT = "#FFFFFF"  # Text on accent buttons

# Borders
BORDER_SUBTLE = "#3E3E42"   # Subtle borders
BORDER_FOCUS = ACCENT       # Focus ring

# Status
STATUS_SUCCESS = "#4ADE80"
STATUS_WARNING = "#FBBF24"
STATUS_ERROR = "#F87171"

# Matplotlib plot colors
MPL_FACE_FIG = BG_CARD
MPL_FACE_AX = BG_PRIMARY
MPL_GRID = "#3A3A3E"
MPL_SPINE = "#4A4A4E"
MPL_TICK = "#A0A0A5"
MPL_LABEL = "#D4D4D4"

# =============================================================================
# QSS Stylesheet
# =============================================================================

MAIN_STYLESHEET = f"""
/* ---- Global ---- */
QMainWindow, QWidget {{
    background-color: {BG_PRIMARY};
    color: {TEXT_PRIMARY};
    font-family: "Segoe UI", "Inter", "Roboto", sans-serif;
    font-size: 13px;
}}

/* ---- Frames / Cards ---- */
QFrame[card="true"] {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 8px;
    padding: 16px;
}}

/* ---- Labels ---- */
QLabel {{
    color: {TEXT_PRIMARY};
    background: transparent;
}}
QLabel[secondary="true"] {{
    color: {TEXT_SECONDARY};
    font-size: 11px;
}}
QLabel[title="true"] {{
    font-size: 14px;
    font-weight: bold;
    color: {TEXT_PRIMARY};
    padding-bottom: 4px;
}}
QLabel[card_title="true"] {{
    font-size: 13px;
    font-weight: bold;
    color: {TEXT_PRIMARY};
    padding: 0px;
    margin-bottom: 6px;
}}
QLabel[status="true"] {{
    color: {TEXT_SECONDARY};
    font-size: 11px;
    padding: 4px;
}}

/* ---- Group Box (legacy fallback) ---- */
QGroupBox {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 16px;
    font-weight: bold;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 8px;
    color: {TEXT_PRIMARY};
}}

/* ---- Spin Boxes ---- */
QDoubleSpinBox, QSpinBox {{
    background-color: {BG_INPUT};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 4px;
    padding: 6px 8px;
    min-height: 20px;
    selection-background-color: {ACCENT};
}}
QDoubleSpinBox:focus, QSpinBox:focus {{
    border: 1px solid {BORDER_FOCUS};
}}
QDoubleSpinBox:hover, QSpinBox:hover {{
    border: 1px solid {TEXT_SECONDARY};
}}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button,
QSpinBox::up-button, QSpinBox::down-button {{
    background-color: {BG_INPUT};
    border: none;
    width: 16px;
}}
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover,
QSpinBox::up-button:hover, QSpinBox::down-button:hover {{
    background-color: {BG_HOVER};
}}
QDoubleSpinBox::up-arrow, QDoubleSpinBox::down-arrow,
QSpinBox::up-arrow, QSpinBox::down-arrow {{
    width: 8px;
    height: 8px;
}}

/* ---- Combo Box ---- */
QComboBox {{
    background-color: {BG_INPUT};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 4px;
    padding: 6px 8px;
    min-height: 20px;
}}
QComboBox:focus {{
    border: 1px solid {BORDER_FOCUS};
}}
QComboBox:hover {{
    border: 1px solid {TEXT_SECONDARY};
}}
QComboBox::drop-down {{
    border: none;
    width: 24px;
}}
QComboBox QAbstractItemView {{
    background-color: {BG_CARD};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_SUBTLE};
    selection-background-color: {ACCENT};
    selection-color: {TEXT_PRIMARY};
}}

/* ---- Line Edit ---- */
QLineEdit {{
    background-color: {BG_INPUT};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 4px;
    padding: 6px 8px;
}}
QLineEdit:focus {{
    border: 1px solid {BORDER_FOCUS};
}}

/* ---- Buttons ---- */
QPushButton {{
    background-color: {BG_INPUT};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 500;
}}
QPushButton:hover {{
    background-color: {BG_HOVER};
    border: 1px solid {TEXT_SECONDARY};
}}
QPushButton:pressed {{
    background-color: {BG_PRESSED};
}}
QPushButton:disabled {{
    background-color: {BG_INPUT};
    color: {TEXT_DISABLED};
    border: 1px solid {BORDER_SUBTLE};
}}

/* ---- Primary Accent Button ---- */
QPushButton[accent="true"] {{
    background-color: {ACCENT};
    color: {TEXT_ON_ACCENT};
    border: none;
    border-radius: 6px;
    padding: 10px 20px;
    font-weight: bold;
    font-size: 13px;
}}
QPushButton[accent="true"]:hover {{
    background-color: {ACCENT_HOVER};
}}
QPushButton[accent="true"]:pressed {{
    background-color: {ACCENT_PRESSED};
}}
QPushButton[accent="true"]:disabled {{
    background-color: {BG_INPUT};
    color: {TEXT_DISABLED};
}}

/* ---- Progress Bar ---- */
QProgressBar {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 4px;
    text-align: center;
    color: {TEXT_PRIMARY};
    min-height: 18px;
    font-size: 11px;
}}
QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 3px;
}}

/* ---- Tab Widget ---- */
QTabWidget::pane {{
    background-color: {BG_PRIMARY};
    border: 1px solid {BORDER_SUBTLE};
    border-top-left-radius: 0px;
    border-top-right-radius: 0px;
    border-bottom-left-radius: 8px;
    border-bottom-right-radius: 8px;
}}
QTabBar::tab {{
    background-color: {BG_CARD};
    color: {TEXT_SECONDARY};
    border: 1px solid {BORDER_SUBTLE};
    border-bottom: none;
    padding: 10px 20px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
}}
QTabBar::tab:selected {{
    background-color: {BG_PRIMARY};
    color: {TEXT_PRIMARY};
    border-bottom: 2px solid {ACCENT};
}}
QTabBar::tab:hover:!selected {{
    background-color: {BG_HOVER};
    color: {TEXT_PRIMARY};
}}

/* ---- Scroll Bar ---- */
QScrollBar:vertical {{
    background-color: {BG_PRIMARY};
    width: 8px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background-color: {BG_INPUT};
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background-color: {TEXT_SECONDARY};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: none;
}}
QScrollBar:horizontal {{
    background-color: {BG_PRIMARY};
    height: 8px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background-color: {BG_INPUT};
    border-radius: 4px;
    min-width: 30px;
}}
QScrollBar::handle:horizontal:hover {{
    background-color: {TEXT_SECONDARY};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
}}

/* ---- Check Box ---- */
QCheckBox {{
    color: {TEXT_PRIMARY};
    spacing: 8px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 3px;
    background-color: {BG_INPUT};
}}
QCheckBox::indicator:checked {{
    background-color: {ACCENT};
    border: 1px solid {ACCENT};
}}
QCheckBox::indicator:hover {{
    border: 1px solid {TEXT_SECONDARY};
}}

/* ---- Dialog ---- */
QDialog {{
    background-color: {BG_PRIMARY};
}}

/* ---- Message Box ---- */
QMessageBox {{
    background-color: {BG_PRIMARY};
}}
QMessageBox QLabel {{
    color: {TEXT_PRIMARY};
}}
QMessageBox QPushButton {{
    min-width: 80px;
}}

/* ---- Tool Tip ---- */
QToolTip {{
    background-color: {BG_CARD};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 4px;
    padding: 4px 8px;
}}
"""


# =============================================================================
# Matplotlib Configuration
# =============================================================================

MPL_RC_PARAMS: Dict = {
    # Figure
    "figure.facecolor": MPL_FACE_FIG,
    "figure.edgecolor": MPL_FACE_FIG,
    "savefig.facecolor": MPL_FACE_FIG,
    "savefig.edgecolor": MPL_FACE_FIG,
    "savefig.dpi": 150,

    # Axes
    "axes.facecolor": MPL_FACE_AX,
    "axes.edgecolor": MPL_SPINE,
    "axes.labelcolor": MPL_LABEL,
    "axes.titlecolor": MPL_LABEL,
    "axes.grid": True,
    "axes.linewidth": 0.8,
    "axes.titlesize": 12,
    "axes.labelsize": 11,

    # Ticks
    "xtick.color": MPL_TICK,
    "ytick.color": MPL_TICK,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.direction": "out",
    "ytick.direction": "out",

    # Grid
    "grid.color": MPL_GRID,
    "grid.alpha": 0.4,
    "grid.linewidth": 0.5,

    # Lines
    "lines.linewidth": 0.8,
    "lines.antialiased": True,

    # Font
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Inter", "Roboto", "DejaVu Sans"],
    "font.size": 11,

    # Legend
    "legend.facecolor": BG_CARD,
    "legend.edgecolor": BORDER_SUBTLE,
    "legend.fontsize": 9,
    "legend.labelcolor": TEXT_PRIMARY,
    "legend.framealpha": 0.9,

    # Spines
    "axes.spines.top": False,
    "axes.spines.right": False,

    # Patch (histogram bars, etc.)
    "patch.facecolor": ACCENT,
    "patch.edgecolor": MPL_FACE_AX,
    "patch.linewidth": 0.3,
}


def apply_mpl_theme() -> None:
    """Apply the dark Matplotlib theme globally."""
    import matplotlib.pyplot as plt
    plt.rcParams.update(MPL_RC_PARAMS)
