"""Main GUI window for Cercus Analysis.

Runs in the main process. Communicates with DataProcessor exclusively
via multiprocessing.Queue objects.
"""

import logging
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.models import (
    Command,
    CommandAction,
    SessionResults,
    Telemetry,
    TelemetryStatus,
)
from src.visualization.visualizer import generate_all_panels, export_figures
from src.gui.theme import MAIN_STYLESHEET, apply_mpl_theme
from src.gui.group_analysis_window import GroupAnalysisWindow

from PyQt5.QtCore import QTimer

logger = logging.getLogger(__name__)


# =============================================================================
# Helper: Card Frame
# =============================================================================


def _make_card(title: str, parent: Optional[QWidget] = None) -> tuple:
    """Create a styled card frame with a bold title label.

    Returns (QFrame, QVBoxLayout) — caller adds content to the layout.
    """
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


# =============================================================================
# Parameter Panel
# =============================================================================


class ParameterPanel(QWidget):
    """I/O selection and processing parameter inputs."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._selected_dir: Optional[str] = None
        self._init_ui()

    def _init_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(10)

        # Scrollable area for parameters
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        scroll_widget = QWidget()
        layout = QVBoxLayout(scroll_widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # --- Card: Input / Output ---
        io_card, io_layout = _make_card("Input / Output")
        io_form = QFormLayout()
        io_form.setLabelAlignment(Qt.AlignRight)
        io_form.setFormAlignment(Qt.AlignLeft)
        io_form.setSpacing(8)

        self._dir_label = QLabel("No directory selected")
        self._dir_label.setProperty("secondary", True)
        self._dir_label.setWordWrap(True)
        self._dir_btn = QPushButton("Browse...")
        self._dir_btn.setMaximumWidth(100)
        self._dir_btn.clicked.connect(self._select_directory)

        dir_row = QHBoxLayout()
        dir_row.addWidget(self._dir_label, stretch=1)
        dir_row.addWidget(self._dir_btn)
        io_form.addRow("Directory:", dir_row)

        io_layout.addLayout(io_form)
        layout.addWidget(io_card)

        # --- Card: Signal Processing ---
        signal_card, signal_layout = _make_card("Signal Processing")
        signal_form = QFormLayout()
        signal_form.setLabelAlignment(Qt.AlignRight)
        signal_form.setFormAlignment(Qt.AlignLeft)
        signal_form.setSpacing(8)

        self._filter_cutoff = self._make_spin(QDoubleSpinBox, 0.1, 50.0, 10.0, 0.5)
        signal_form.addRow(
            "Low-pass Cutoff:", self._add_unit_label(self._filter_cutoff, "Hz")
        )

        self._resample_freq = self._make_spin(QDoubleSpinBox, 10.0, 1000.0, 100.0, 10.0)
        signal_form.addRow(
            "Resample Freq:", self._add_unit_label(self._resample_freq, "Hz")
        )

        signal_layout.addLayout(signal_form)
        layout.addWidget(signal_card)

        # --- Card: Ethological Detection ---
        etho_card, etho_layout = _make_card("Ethological Detection")
        etho_form = QFormLayout()
        etho_form.setLabelAlignment(Qt.AlignRight)
        etho_form.setFormAlignment(Qt.AlignLeft)
        etho_form.setSpacing(8)

        self._freezing_threshold = self._make_spin(QDoubleSpinBox, 0.01, 10.0, 0.5, 0.1)
        etho_form.addRow(
            "Freezing Thresh:", self._add_unit_label(self._freezing_threshold, "mm/s")
        )

        self._freezing_duration = self._make_spin(
            QDoubleSpinBox, 100.0, 10000.0, 1000.0, 100.0
        )
        etho_form.addRow(
            "Min Duration:", self._add_unit_label(self._freezing_duration, "ms")
        )

        self._escape_threshold = self._make_spin(QDoubleSpinBox, 0.1, 50.0, 2.0, 0.5)
        etho_form.addRow(
            "Escape Accel:", self._add_unit_label(self._escape_threshold, "")
        )

        etho_layout.addLayout(etho_form)
        layout.addWidget(etho_card)

        # --- Card: PSTH Window ---
        psth_card, psth_layout = _make_card("PSTH Window")
        psth_form = QFormLayout()
        psth_form.setLabelAlignment(Qt.AlignRight)
        psth_form.setFormAlignment(Qt.AlignLeft)
        psth_form.setSpacing(8)

        self._psth_pre = self._make_spin(QDoubleSpinBox, -30.0, 0.0, -2.0, 0.5)
        psth_form.addRow("Pre-stimulus:", self._add_unit_label(self._psth_pre, "s"))

        self._psth_post = self._make_spin(QDoubleSpinBox, 0.0, 60.0, 5.0, 0.5)
        psth_form.addRow("Post-stimulus:", self._add_unit_label(self._psth_post, "s"))

        self._stimulus_offset = self._make_spin(QDoubleSpinBox, 0.0, 30.0, 0.0, 0.1)
        psth_form.addRow("Stimulus Offset:", self._add_unit_label(self._stimulus_offset, "s"))

        psth_layout.addLayout(psth_form)
        layout.addWidget(psth_card)

        layout.addStretch()
        scroll.setWidget(scroll_widget)
        root_layout.addWidget(scroll, stretch=1)

        # --- Actions (fixed at bottom, outside scroll) ---
        actions_card, actions_layout = _make_card("Actions")

        self._process_btn = QPushButton("Process Batch")
        self._process_btn.setProperty("accent", True)
        self._process_btn.setMinimumHeight(42)
        self._process_btn.setEnabled(False)
        actions_layout.addWidget(self._process_btn)

        export_row = QHBoxLayout()
        export_row.setSpacing(8)
        self._export_btn = QPushButton("Export CSV")
        self._export_btn.setEnabled(False)
        export_row.addWidget(self._export_btn)
        self._export_fig_btn = QPushButton("Export Figures")
        self._export_fig_btn.setEnabled(False)
        export_row.addWidget(self._export_fig_btn)
        actions_layout.addLayout(export_row)

        self._group_analysis_btn = QPushButton("Launch Group Analysis (基线对比)")
        self._group_analysis_btn.setEnabled(True)
        actions_layout.addWidget(self._group_analysis_btn)

        root_layout.addWidget(actions_card)

        # --- Progress & Status ---
        self._progress = QProgressBar()
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        root_layout.addWidget(self._progress)

        self._status_label = QLabel("Ready")
        self._status_label.setProperty("status", True)
        root_layout.addWidget(self._status_label)

    def _make_spin(
        self,
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

    def _add_unit_label(self, widget: QWidget, unit: str) -> QWidget:
        """Wrap a widget with an optional unit label in a horizontal layout."""
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

    def _select_directory(self) -> None:
        """Open directory selection dialog."""
        path = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if path:
            self._selected_dir = path
            self._dir_label.setText(path)
            self._dir_label.setProperty("secondary", False)
            self._dir_label.style().unpolish(self._dir_label)
            self._dir_label.style().polish(self._dir_label)
            self._process_btn.setEnabled(True)
            self._status_label.setText(f"Selected: {Path(path).name}")

    def get_params(self) -> Dict[str, Any]:
        """Return current parameter values."""
        return {
            "input_dir": self._selected_dir or "",
            "output_dir": str(Path(self._selected_dir or ".") / "output"),
            "filter_cutoff_hz": self._filter_cutoff.value(),
            "resample_freq_hz": self._resample_freq.value(),
            "freezing_threshold_mm_s": self._freezing_threshold.value(),
            "freezing_min_duration_ms": self._freezing_duration.value(),
            "escape_acceleration_threshold": self._escape_threshold.value(),
            "psth_window_pre_s": self._psth_pre.value(),
            "psth_window_post_s": self._psth_post.value(),
            "stimulus_offset_s": self._stimulus_offset.value(),
            "filter_order": 4,
        }

    def set_progress(self, value: float) -> None:
        """Update progress bar (0.0 to 1.0)."""
        self._progress.setValue(int(value * 100))

    def set_status(self, text: str) -> None:
        """Update status label."""
        self._status_label.setText(text)

    def set_processing(self, busy: bool) -> None:
        """Enable/disable controls during processing."""
        self._process_btn.setEnabled(not busy and self._selected_dir is not None)
        self._export_btn.setEnabled(not busy)
        self._export_fig_btn.setEnabled(not busy)
        self._group_analysis_btn.setEnabled(not busy)


# =============================================================================
# Visualization Canvas
# =============================================================================


class VisualizationCanvas(QWidget):
    """Tabbed matplotlib canvas for displaying analysis figures."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._current_figures: Dict[str, Figure] = {}
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Session selector row
        selector_row = QHBoxLayout()
        selector_row.setContentsMargins(4, 4, 4, 0)
        selector_row.setSpacing(8)
        session_lbl = QLabel("Session:")
        session_lbl.setProperty("secondary", True)
        selector_row.addWidget(session_lbl)
        self._session_combo = QComboBox()
        self._session_combo.setMinimumHeight(30)
        self._session_combo.currentIndexChanged.connect(self._on_session_changed)
        selector_row.addWidget(self._session_combo, stretch=1)
        layout.addLayout(selector_row)

        # Tab widget
        self._tabs = QTabWidget()
        layout.addWidget(self._tabs, stretch=1)

        # Pre-create tabs
        self._tab_canvases: Dict[str, FigureCanvas] = {}
        self._tab_toolbars: Dict[str, NavigationToolbar] = {}
        self._tab_widgets: Dict[str, QWidget] = {}
        tab_names = {
            "trial_trajectory": "Trial Trajectory",
            "trajectory_heatmap": "Trajectory + Heatmap",
            "speed_timeseries": "Speed vs Time",
            "psth": "PSTH Response",
            "angular_velocity_histogram": "Angular Velocity",
            "multimodal_psth": "Multimodal PSTH",
        }
        for key, title in tab_names.items():
            canvas = FigureCanvas(Figure(figsize=(8, 6)))
            canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            toolbar = NavigationToolbar(canvas, self)
            tab_widget = QWidget()
            tab_layout = QVBoxLayout(tab_widget)
            tab_layout.setContentsMargins(2, 2, 2, 2)
            tab_layout.setSpacing(2)
            tab_layout.addWidget(toolbar)
            tab_layout.addWidget(canvas, stretch=1)
            tab_idx = self._tabs.addTab(tab_widget, title)
            self._tab_canvases[key] = canvas
            self._tab_toolbars[key] = toolbar
            self._tab_widgets[key] = tab_widget
            # Hide data-dependent tabs by default
            if key in ("trial_trajectory", "multimodal_psth"):
                self._tabs.setTabVisible(tab_idx, False)

        self._results: List[SessionResults] = []
        self._current_session_idx: int = -1

    def set_results(self, results: List[SessionResults]) -> None:
        """Store results and populate session selector."""
        self._results = results
        self._session_combo.blockSignals(True)
        self._session_combo.clear()
        for res in results:
            self._session_combo.addItem(res.session_id)
        self._session_combo.blockSignals(False)

        if results:
            self._session_combo.setCurrentIndex(0)
            self._update_figures(0)

    def _on_session_changed(self, index: int) -> None:
        """Handle session selection change."""
        if 0 <= index < len(self._results):
            self._update_figures(index)

    def _update_figures(self, session_idx: int) -> None:
        """Render figures for the selected session into tab canvases."""
        # Close previous figures to free memory
        for fig in self._current_figures.values():
            plt.close(fig)
        self._current_figures.clear()

        if session_idx < 0 or session_idx >= len(self._results):
            # Clear all canvases with a placeholder so the tab layout stays intact
            for key, canvas in self._tab_canvases.items():
                canvas.figure.clear()
                ax = canvas.figure.add_subplot(111)
                ax.text(
                    0.5, 0.5, "No data loaded",
                    ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="gray",
                )
                ax.set_axis_off()
                canvas.draw()
            return

        self._current_session_idx = session_idx
        result = self._results[session_idx]

        # Generate fresh figures
        figures = generate_all_panels(result)

        # Render each panel into its canvas
        for key, canvas in self._tab_canvases.items():
            if key in figures:
                fig = figures[key]
                old_fig = canvas.figure
                canvas.figure = fig
                fig.set_canvas(canvas)

                toolbar = self._tab_toolbars.get(key)
                if toolbar:
                    toolbar.update()

                canvas.draw()
                plt.close(old_fig)
                self._current_figures[key] = fig
            else:
                # Clear canvases that have no data this session
                canvas.figure.clear()
                ax = canvas.figure.add_subplot(111)
                ax.text(
                    0.5, 0.5, "Not available for this session",
                    ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="gray",
                )
                ax.set_axis_off()
                canvas.draw()

        # Toggle visibility of data-dependent tabs
        data_dependent_keys = ("trial_trajectory", "multimodal_psth")
        for key in data_dependent_keys:
            tab_widget = self._tab_widgets.get(key)
            if tab_widget is None:
                continue
            tab_idx = self._tabs.indexOf(tab_widget)
            if tab_idx < 0:
                continue

            if key in figures:
                self._tabs.setTabVisible(tab_idx, True)
            else:
                self._tabs.setTabVisible(tab_idx, False)

    def get_current_figures(self) -> Dict[str, Figure]:
        """Return figures for the currently selected session."""
        return dict(self._current_figures)


# =============================================================================
# Export Dialog
# =============================================================================


class ExportDialog(QDialog):
    """Dialog for selecting export formats."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export Figures")
        self.setMinimumWidth(320)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        header = QLabel("Select output formats:")
        header.setProperty("title", True)
        layout.addWidget(header)

        self._png_check = QCheckBox("PNG (raster, 150 DPI)")
        self._png_check.setChecked(True)
        layout.addWidget(self._png_check)

        self._pdf_check = QCheckBox("PDF (vector)")
        self._pdf_check.setChecked(True)
        layout.addWidget(self._pdf_check)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_formats(self) -> tuple:
        """Return selected formats as a tuple of strings."""
        formats = []
        if self._png_check.isChecked():
            formats.append("png")
        if self._pdf_check.isChecked():
            formats.append("pdf")
        return tuple(formats)


# =============================================================================
# Main Window
# =============================================================================


class MainWindow(QMainWindow):
    """Main application window. Communicates via queues only."""

    def __init__(
        self,
        cmd_queue: mp.Queue,
        telemetry_queue: mp.Queue,
    ) -> None:
        super().__init__()
        self._cmd_queue: mp.Queue = cmd_queue
        self._telemetry_queue: mp.Queue = telemetry_queue
        self._pending_requests: Dict[str, str] = {}
        self._results: List[SessionResults] = []
        self._cached_results: List[SessionResults] = []

        # Apply theme
        self.setStyleSheet(MAIN_STYLESHEET)
        apply_mpl_theme()

        self._init_ui()
        self._start_telemetry_timer()

    def _init_ui(self) -> None:
        self.setWindowTitle("Cercus Analysis — Locomotor Tracking")
        self.setMinimumSize(1400, 900)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(10)

        # Left panel: parameters (fixed width)
        self._param_panel = ParameterPanel()
        self._param_panel.setMinimumWidth(300)
        self._param_panel.setMaximumWidth(360)
        self._param_panel._process_btn.clicked.connect(self._on_process)
        self._param_panel._export_btn.clicked.connect(self._on_export_csv)
        self._param_panel._export_fig_btn.clicked.connect(self._on_export_figures)
        self._param_panel._group_analysis_btn.clicked.connect(
            self._on_launch_group_analysis
        )
        main_layout.addWidget(self._param_panel)

        # Right panel: visualization (expanding)
        self._viz_canvas = VisualizationCanvas()
        main_layout.addWidget(self._viz_canvas, stretch=1)

    def _start_telemetry_timer(self) -> None:
        """Poll telemetry queue on a timer to avoid blocking UI."""
        self._telemetry_timer = QTimer(self)
        self._telemetry_timer.timeout.connect(self._poll_telemetry)
        self._telemetry_timer.start(50)

    def _poll_telemetry(self) -> None:
        """Non-blocking check of telemetry queue."""
        while not self._telemetry_queue.empty():
            try:
                msg: Telemetry = self._telemetry_queue.get_nowait()
                self._handle_telemetry(msg)
            except Exception:
                break

    def _handle_telemetry(self, msg: Telemetry) -> None:
        """Process incoming telemetry from DataProcessor."""
        if msg.status == TelemetryStatus.PROGRESS:
            self._param_panel.set_progress(msg.progress)
            self._param_panel.set_status(f"Processing... {msg.progress:.0%}")

        elif msg.status == TelemetryStatus.COMPLETE:
            self._pending_requests.pop(msg.request_id, None)

            if msg.data and "results" in msg.data:
                self._results = msg.data["results"]
                self._cached_results = self._results
                self._viz_canvas.set_results(self._results)
                self._param_panel._export_btn.setEnabled(True)
                self._param_panel._export_fig_btn.setEnabled(True)
                self._param_panel.set_status(
                    f"Complete — {len(self._results)} session(s)"
                )
            elif msg.data and "summary_path" in msg.data:
                self._param_panel.set_status(f"Exported: {msg.data['summary_path']}")
            else:
                self._param_panel.set_status("Complete")

            self._param_panel.set_progress(1.0)
            self._param_panel.set_processing(False)

        elif msg.status == TelemetryStatus.ERROR:
            self._pending_requests.pop(msg.request_id, None)
            self._param_panel.set_processing(False)
            self._param_panel.set_progress(0)
            self._param_panel.set_status("Error")
            QMessageBox.critical(
                self,
                "Processing Error",
                msg.error or "Unknown error occurred",
            )

    def _on_process(self) -> None:
        """Send process_batch command to DataProcessor."""
        params = self._param_panel.get_params()
        if not params["input_dir"]:
            QMessageBox.warning(self, "No Input", "Please select an input directory.")
            return

        cmd = Command(
            action=CommandAction.PROCESS_BATCH,
            params=params,
        )
        self._pending_requests[cmd.request_id] = "process_batch"
        self._cmd_queue.put(cmd)

        self._param_panel.set_processing(True)
        self._param_panel.set_progress(0)
        self._param_panel.set_status("Processing...")

    def _on_export_csv(self) -> None:
        """Send export_results command to DataProcessor."""
        if not self._results:
            QMessageBox.information(self, "No Data", "Process a batch first.")
            return

        params = self._param_panel.get_params()
        cmd = Command(
            action=CommandAction.EXPORT_RESULTS,
            params={
                "results": self._results,
                "output_dir": params["output_dir"],
            },
        )
        self._pending_requests[cmd.request_id] = "export_csv"
        self._cmd_queue.put(cmd)
        self._param_panel.set_status("Exporting CSV...")

    def _on_export_figures(self) -> None:
        """Export figures directly from GUI process."""
        if not self._results:
            QMessageBox.information(self, "No Data", "Process a batch first.")
            return

        dialog = ExportDialog(self)
        if dialog.exec_() != QDialog.Accepted:
            return

        formats = dialog.get_formats()
        if not formats:
            QMessageBox.warning(self, "No Format", "Select at least one format.")
            return

        figures = self._viz_canvas.get_current_figures()
        if not figures:
            return

        params = self._param_panel.get_params()
        output_dir = Path(params["output_dir"]) / "figures"
        session_id = (
            self._results[self._viz_canvas._current_session_idx].session_id
            if self._results
            else "unknown"
        )

        saved = export_figures(figures, output_dir, session_id, formats)
        self._param_panel.set_status(f"Exported {len(saved)} figure(s)")
        QMessageBox.information(
            self,
            "Export Complete",
            f"Saved {len(saved)} figure(s) to:\n{output_dir}",
        )

    def _on_launch_group_analysis(self) -> None:
        """Open the group analysis window as an independent workspace."""
        dlg = GroupAnalysisWindow(
            cmd_queue=self._cmd_queue,
            telemetry_queue=self._telemetry_queue,
            speed_threshold_mm_s=self._param_panel._escape_threshold.value(),
            parent=self,
        )
        dlg.exec_()

    def closeEvent(self, event: Any) -> None:
        """Shutdown DataProcessor on window close."""
        cmd = Command(action=CommandAction.SHUTDOWN)
        self._cmd_queue.put(cmd)
        plt.close("all")
        event.accept()
