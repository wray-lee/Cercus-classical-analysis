"""Calibration Visualizer GUI: Standalone hardware validation window.

Independent from the main ethological analysis pipeline. Uses
CalibrationProcessor for all kinematics math — no DataProcessor,
no multiprocessing, no queues.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.processors.calibration_core import CalibrationProcessor, CalibrationResult
from src.visualization.calibration import (
    create_crosstalk_panel,
    create_yaw_drift_panel,
    create_jitter_histogram,
    create_steering_sine_panel,
    export_calibration_report,
)
from src.gui.theme import MAIN_STYLESHEET, apply_mpl_theme

logger = logging.getLogger(__name__)


# =============================================================================
# Helper: Card Frame
# =============================================================================

def _make_card(title: str, parent: Optional[QWidget] = None) -> tuple:
    """Create a styled card frame with a bold title label."""
    frame = QFrame(parent)
    frame.setProperty("card", True)
    frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

    layout = QVBoxLayout(frame)
    layout.setContentsMargins(16, 14, 16, 14)
    layout.setSpacing(10)

    title_label = QLabel(title)
    title_label.setProperty("card_title", True)
    layout.addWidget(title_label)

    return frame, layout


def _make_spin(
    spin_type: type,
    min_val: float,
    max_val: float,
    default: float,
    step: float,
) -> Any:
    """Create a configured spin box."""
    spin = spin_type()
    spin.setRange(min_val, max_val)
    spin.setValue(default)
    spin.setSingleStep(step)
    spin.setMinimumHeight(30)
    return spin


def _add_unit_label(widget: QWidget, unit: str) -> QWidget:
    """Wrap a widget with an optional unit label."""
    if not unit:
        return widget
    container = QWidget()
    row = QHBoxLayout(container)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(6)
    row.addWidget(widget)
    unit_lbl = QLabel(unit)
    unit_lbl.setProperty("secondary", True)
    unit_lbl.setFixedWidth(36)
    row.addWidget(unit_lbl)
    return container


# =============================================================================
# Calibration Canvas
# =============================================================================

class CalibrationCanvas(QWidget):
    """Tabbed matplotlib canvas for calibration figures."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._current_figures: Dict[str, Figure] = {}
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._tabs = QTabWidget()
        layout.addWidget(self._tabs)

        self._tab_canvases: Dict[str, FigureCanvas] = {}
        tab_names = {
            "cross_talk": "Cross-talk Inspection",
            "yaw_drift": "Yaw Drift",
            "jitter_histogram": "Jitter Histogram",
            "steering_sine": "Steering Angle",
        }
        for key, title in tab_names.items():
            canvas = FigureCanvas(Figure(figsize=(10, 6)))
            canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            toolbar = NavigationToolbar(canvas, self)
            tab_widget = QWidget()
            tab_layout = QVBoxLayout(tab_widget)
            tab_layout.setContentsMargins(2, 2, 2, 2)
            tab_layout.setSpacing(2)
            tab_layout.addWidget(toolbar)
            tab_layout.addWidget(canvas, stretch=1)
            self._tabs.addTab(tab_widget, title)
            self._tab_canvases[key] = canvas

    def update_figures(self, figures: Dict[str, Figure]) -> None:
        """Replace canvas contents with new figures."""
        for fig in self._current_figures.values():
            plt.close(fig)
        self._current_figures.clear()

        for key, fig in figures.items():
            canvas = self._tab_canvases.get(key)
            if canvas is None:
                continue

            old_fig = canvas.figure
            canvas.figure = fig
            fig.set_canvas(canvas)
            canvas.draw()
            plt.close(old_fig)
            self._current_figures[key] = fig

    def get_figures(self) -> Dict[str, Figure]:
        """Return current figures."""
        return dict(self._current_figures)


# =============================================================================
# Calibration Window
# =============================================================================

class CalibrationWindow(QMainWindow):
    """Standalone calibration visualizer.

    Uses CalibrationProcessor for all kinematics math.
    No multiprocessing. No queues. No dependency on main pipeline.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._session_id: str = ""
        self._current_figures: Dict[str, Figure] = {}
        self._result: Optional[CalibrationResult] = None

        # Apply theme
        self.setStyleSheet(MAIN_STYLESHEET)
        apply_mpl_theme()

        self._init_ui()

    def _init_ui(self) -> None:
        self.setWindowTitle("Cercus Calibration — Hardware Validation")
        self.setMinimumSize(1400, 900)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(10)

        # Left panel: controls (fixed width)
        ctrl_panel = QWidget()
        ctrl_panel.setMinimumWidth(280)
        ctrl_panel.setMaximumWidth(340)
        ctrl_layout = QVBoxLayout(ctrl_panel)
        ctrl_layout.setContentsMargins(0, 0, 0, 0)
        ctrl_layout.setSpacing(10)

        # --- Card: Kinematics File ---
        file_card, file_layout = _make_card("Kinematics File")

        self._file_label = QLabel("No file selected")
        self._file_label.setProperty("secondary", True)
        self._file_label.setWordWrap(True)
        file_layout.addWidget(self._file_label)

        self._file_btn = QPushButton("Browse...")
        self._file_btn.setMinimumHeight(34)
        self._file_btn.clicked.connect(self._select_file)
        file_layout.addWidget(self._file_btn)

        ctrl_layout.addWidget(file_card)

        # --- Card: Signal Processing ---
        signal_card, signal_layout = _make_card("Signal Processing")
        signal_form = QFormLayout()
        signal_form.setLabelAlignment(Qt.AlignRight)
        signal_form.setFormAlignment(Qt.AlignLeft)
        signal_form.setSpacing(8)

        self._filter_cutoff = _make_spin(QDoubleSpinBox, 0.1, 50.0, 10.0, 0.5)
        signal_form.addRow("Low-pass Cutoff:", _add_unit_label(self._filter_cutoff, "Hz"))

        self._filter_order = _make_spin(QSpinBox, 1, 10, 4, 1)
        signal_form.addRow("Filter Order:", _add_unit_label(self._filter_order, ""))

        self._resample_freq = _make_spin(QDoubleSpinBox, 10.0, 1000.0, 100.0, 10.0)
        signal_form.addRow("Resample Freq:", _add_unit_label(self._resample_freq, "Hz"))

        signal_layout.addLayout(signal_form)
        ctrl_layout.addWidget(signal_card)

        # --- Card: Actions ---
        actions_card, actions_layout = _make_card("Actions")

        self._generate_btn = QPushButton("Generate Report")
        self._generate_btn.setProperty("accent", True)
        self._generate_btn.setMinimumHeight(42)
        self._generate_btn.setEnabled(False)
        self._generate_btn.clicked.connect(self._on_generate)
        actions_layout.addWidget(self._generate_btn)

        self._export_btn = QPushButton("Export PDF")
        self._export_btn.setMinimumHeight(36)
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._on_export)
        actions_layout.addWidget(self._export_btn)

        ctrl_layout.addWidget(actions_card)

        # --- Card: File Info ---
        info_card, info_layout = _make_card("File Info")

        self._info_label = QLabel("Load a kinematics CSV to begin.")
        self._info_label.setProperty("secondary", True)
        self._info_label.setWordWrap(True)
        info_layout.addWidget(self._info_label)

        ctrl_layout.addWidget(info_card)

        ctrl_layout.addStretch()
        main_layout.addWidget(ctrl_panel)

        # Right panel: visualization (expanding)
        self._canvas = CalibrationCanvas()
        main_layout.addWidget(self._canvas, stretch=1)

    def _select_file(self) -> None:
        """Open file dialog for kinematics CSV."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Kinematics CSV",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not filepath:
            return

        path = Path(filepath)
        self._file_label.setText(path.name)
        self._file_label.setProperty("secondary", False)
        self._file_label.style().unpolish(self._file_label)
        self._file_label.style().polish(self._file_label)

        # Quick validation: try loading with processor params
        try:
            processor = self._build_processor()
            df = processor.load_csv(path)
            self._session_id = (
                path.stem
                .replace("_kinematics", "")
                .replace("-kinematics", "")
            )

            # Update info
            n_samples = len(df)
            duration = df["sys_time"].iloc[-1] - df["sys_time"].iloc[0]
            dt = float(df["sys_time"].diff().median())
            self._info_label.setText(
                f"File: {path.name}\n"
                f"Samples: {n_samples:,}\n"
                f"Duration: {duration:.2f} s\n"
                f"Median dt: {dt:.4f} s ({1/dt:.1f} Hz)"
            )
            self._generate_btn.setEnabled(True)
            self._loaded_filepath = path

        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load file:\n{e}")
            self._generate_btn.setEnabled(False)
            self._loaded_filepath = None

    def _build_processor(self) -> CalibrationProcessor:
        """Build a CalibrationProcessor from current UI parameters."""
        return CalibrationProcessor(
            filter_cutoff_hz=self._filter_cutoff.value(),
            filter_order=self._filter_order.value(),
            resample_freq_hz=self._resample_freq.value(),
        )

    def _on_generate(self) -> None:
        """Generate calibration report using CalibrationProcessor."""
        if not hasattr(self, "_loaded_filepath") or self._loaded_filepath is None:
            return

        try:
            processor = self._build_processor()
            result = processor.process(self._loaded_filepath, self._session_id)
            self._result = result

            # Generate figures
            figures = {
                "cross_talk": create_crosstalk_panel(
                    result.time, result.dx_filtered, result.dy_filtered,
                    result.dz_filtered, result.session_id,
                ),
                "yaw_drift": create_yaw_drift_panel(
                    result.time, result.dz_filtered, result.session_id,
                ),
                "jitter_histogram": create_jitter_histogram(
                    result.speed, result.session_id,
                ),
                "steering_sine": create_steering_sine_panel(
                    result.time, result.dz_filtered, result.session_id,
                ),
            }

            self._current_figures = figures
            self._canvas.update_figures(figures)
            self._export_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(
                self, "Processing Error", f"Failed to generate report:\n{e}"
            )

    def _on_export(self) -> None:
        """Export calibration figures as PDF."""
        if not self._current_figures:
            return

        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory"
        )
        if not output_dir:
            return

        try:
            filepath = export_calibration_report(
                self._current_figures,
                Path(output_dir),
                self._session_id,
            )
            QMessageBox.information(
                self,
                "Export Complete",
                f"Calibration report saved to:\n{filepath}",
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Export Error", f"Failed to export:\n{e}"
            )

    def closeEvent(self, event: Any) -> None:
        """Clean up figures on close."""
        plt.close("all")
        event.accept()
