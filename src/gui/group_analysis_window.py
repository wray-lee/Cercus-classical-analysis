"""Group analysis window for cross-session baseline comparison.

Operates as an independent workspace: the user selects multiple data
directories (treatment, visual baseline, wind baseline), dispatches a
``PROCESS_GROUP_BATCH`` command to the shared ``DataProcessor``, and
renders group-level comparison plots once telemetry reports completion.
"""

import logging
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.models import (
    Command,
    CommandAction,
    PSTHResult,
    SessionResults,
    Telemetry,
    TelemetryStatus,
)
from src.gui.theme import apply_mpl_theme

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Plot color constants (matching visualizer.py dark-theme palette)
# ---------------------------------------------------------------------------
_C_LOOMING = "#FF6B6B"
_C_AIRPUFF = "#4A9EFF"
_C_BASE_VIS = "#FFB86C"
_C_BASE_WIND = "#8BE9FD"
_C_STIMULUS = "#FBBF24"
_C_PRIMARY = "#4A9EFF"
_C_ACCENT = "#4ADE80"
_GRID_ALPHA = 0.3

# group_label keyword -> plot color (substring match)
_GROUP_COLOR_RULES: List[Tuple[str, str]] = [
    ("looming", _C_LOOMING),
    ("air_puff", _C_AIRPUFF),
    ("base_visual", _C_BASE_VIS),
    ("base_wind", _C_BASE_WIND),
    ("treatment", _C_PRIMARY),
]


def _resolve_group_color(label: str, fallback: str = _C_PRIMARY) -> str:
    """Match a group_label to a plot color via substring."""
    lower = label.lower()
    for keyword, color in _GROUP_COLOR_RULES:
        if keyword in lower:
            return color
    return fallback


# ---------------------------------------------------------------------------
# Pure computation functions (no side-effects, no file I/O)
# ---------------------------------------------------------------------------


def _compute_response_probability(
    sessions: List[SessionResults],
    speed_threshold_mm_s: float = 10.0,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Compute response probability time-series per event type.

    For each event type, at every PSTH time bin, the response probability
    is the fraction of trials whose speed exceeds ``speed_threshold_mm_s``.

    Returns
    -------
    dict
        Mapping of event_type -> (time_axis, probability_array).
        Only returns keys present in at least one session.
    """
    event_type_sessions: Dict[str, List[PSTHResult]] = {}
    for sess in sessions:
        for evt_type, psth in sess.psth_results.items():
            if psth.speed_trials is not None and psth.n_trials > 0:
                event_type_sessions.setdefault(evt_type, []).append(psth)

    result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for evt_type, psth_list in event_type_sessions.items():
        time_axis = psth_list[0].time_axis
        all_trials = np.concatenate(
            [p.speed_trials for p in psth_list if p.speed_trials is not None],
            axis=0,
        )
        prob = np.nanmean(all_trials > speed_threshold_mm_s, axis=0)
        result[evt_type] = (time_axis, prob)

    return result


def _extract_peak_velocities(
    sessions: List[SessionResults],
    post_stimulus_window_s: float = 2.0,
) -> Dict[str, List[float]]:
    """Extract per-trial peak velocity after stimulus onset for each event type.

    Returns
    -------
    dict
        Mapping of event_type -> list of peak velocities (one per trial).
    """
    result: Dict[str, List[float]] = {}
    for sess in sessions:
        for evt_type, psth in sess.psth_results.items():
            if psth.speed_trials is None or psth.n_trials == 0:
                continue

            time_axis = psth.time_axis
            post_mask = (time_axis >= 0) & (time_axis <= post_stimulus_window_s)
            if not np.any(post_mask):
                continue

            post_speed = psth.speed_trials[:, post_mask]
            with np.errstate(all="ignore"):
                peaks = np.nanmax(post_speed, axis=1)
            valid_peaks = peaks[~np.isnan(peaks)].tolist()
            result.setdefault(evt_type, []).extend(valid_peaks)

    return result


# ---------------------------------------------------------------------------
# Figure generators (return Figure objects, no I/O)
# ---------------------------------------------------------------------------


def _find_matching_key(keyword: str, data_keys: List[str]) -> Optional[str]:
    """Return the first key in ``data_keys`` whose lowercase contains ``keyword``."""
    for k in data_keys:
        if keyword in k.lower():
            return k
    return None


def _create_response_probability_figure(
    prob_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    escape_threshold: float,
    group_labels: Optional[List[str]] = None,
) -> Figure:
    """Response probability comparison plot.

    Two subplots:
      - Looming vs Base Visual
      - Air-puff vs Base Wind

    When *group_labels* contains more than one unique label, each
    stimulus keyword is split into separate treatment / baseline traces
    so the user can visually compare across groups in the same axis.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle(
        f"Response Probability (speed > {escape_threshold:.1f} mm/s)",
        fontsize=13, fontweight="bold",
    )

    data_keys = list(prob_data.keys())

    pairs: List[Tuple[str, str, str, str]] = [
        ("looming", _C_LOOMING, "base_visual", _C_BASE_VIS),
        ("air_puff", _C_AIRPUFF, "base_wind", _C_BASE_WIND),
    ]
    pair_titles = ["Visual: Looming vs Baseline", "Mechanical: Air-puff vs Baseline"]

    for ax, (exp_kw, exp_c, ctrl_kw, ctrl_c), title in zip(axes, pairs, pair_titles):
        has_data = False
        for keyword, color in [(exp_kw, exp_c), (ctrl_kw, ctrl_c)]:
            matched_key = _find_matching_key(keyword, data_keys)
            if matched_key is not None:
                t, p = prob_data[matched_key]
                label = matched_key.replace("_", " ").title()
                ax.plot(t, p, linewidth=1.8, color=color, label=label)
                ax.fill_between(t, 0, p, alpha=0.15, color=color)
                has_data = True

        if not has_data:
            ax.text(
                0.5, 0.5, "No data",
                ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="gray",
            )

        ax.axvline(-0.5, color=_C_LOOMING, linestyle="--", linewidth=1.2, alpha=0.7, label="Looming Onset")
        ax.axvline(0.0, color=_C_STIMULUS, linestyle="--", linewidth=1.2, alpha=0.7, label="Air-puff / Wind Onset")
        ax.set_xlabel("Time relative to stimulus (s)")
        ax.set_ylabel("Response Probability")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
        ax.grid(True, alpha=_GRID_ALPHA)

    fig.tight_layout()
    return fig


def _create_group_psth_figure(
    sessions: List[SessionResults],
) -> Figure:
    """Overlay PSTH mean speed traces split by group_label.

    For each event-type keyword (looming, air_puff, base_visual,
    base_wind), all sessions whose ``group_label`` contains that keyword
    are pooled and their PSTH traces overlaid with the group's color.
    This produces a single figure that visually compares treatment
    sessions against their matched baselines in the same metric space.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle(
        "PSTH Comparison by Group",
        fontsize=13, fontweight="bold",
    )

    stimulus_keywords = [
        ("looming", "base_visual", "Visual: Looming vs Baseline"),
        ("air_puff", "base_wind", "Mechanical: Air-puff vs Baseline"),
    ]

    for ax, (treat_kw, base_kw, title) in zip(axes, stimulus_keywords):
        has_data = False

        for keyword, fallback_color in [
            (treat_kw, _C_LOOMING if treat_kw == "looming" else _C_AIRPUFF),
            (base_kw, _C_BASE_VIS if base_kw == "base_visual" else _C_BASE_WIND),
        ]:
            # Pool PSTH results from sessions whose group_label matches
            pooled_means: List[np.ndarray] = []
            pooled_sems: List[np.ndarray] = []
            time_axis = None
            label_tag = keyword

            for sess in sessions:
                gl = sess.group_label.lower()
                # Check if this session belongs to the right group
                is_match = keyword in gl
                # Also match sessions whose event types contain the keyword
                if not is_match:
                    continue

                for evt_type, psth in sess.psth_results.items():
                    if keyword in evt_type.lower() and psth.n_trials > 0:
                        pooled_means.append(psth.mean_speed)
                        pooled_sems.append(psth.sem_speed)
                        if time_axis is None:
                            time_axis = psth.time_axis
                        label_tag = f"{sess.group_label}: {evt_type}"

            # Also try direct event-type matching across all sessions
            if not pooled_means:
                for sess in sessions:
                    for evt_type, psth in sess.psth_results.items():
                        if keyword in evt_type.lower() and psth.n_trials > 0:
                            pooled_means.append(psth.mean_speed)
                            pooled_sems.append(psth.sem_speed)
                            if time_axis is None:
                                time_axis = psth.time_axis

            if not pooled_means or time_axis is None:
                continue

            grand_mean = np.nanmean(pooled_means, axis=0)
            # Pool SEMs in quadrature / sqrt(N)
            grand_sem = np.sqrt(np.nanmean(np.array(pooled_sems) ** 2, axis=0))
            color = _resolve_group_color(keyword, fallback_color)
            ax.plot(
                time_axis, grand_mean,
                linewidth=1.8, color=color,
                label=keyword.replace("_", " ").title(),
            )
            ax.fill_between(
                time_axis,
                grand_mean - grand_sem,
                grand_mean + grand_sem,
                alpha=0.15, color=color,
            )
            has_data = True

        if not has_data:
            ax.text(
                0.5, 0.5, "No data",
                ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="gray",
            )

        ax.axvline(-0.5, color=_C_LOOMING, linestyle="--", linewidth=1.2, alpha=0.7, label="Looming Onset")
        ax.axvline(0.0, color=_C_STIMULUS, linestyle="--", linewidth=1.2, alpha=0.7, label="Air-puff / Wind Onset")
        ax.set_xlabel("Time relative to stimulus (s)")
        ax.set_ylabel("Speed (mm/s)")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
        ax.grid(True, alpha=_GRID_ALPHA)

    fig.tight_layout()
    return fig


def _create_peak_velocity_figure(
    peak_data: Dict[str, List[float]],
) -> Figure:
    """Peak velocity boxplot + stripplot comparison across all four conditions."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.suptitle(
        "Post-Stimulus Peak Velocity Distribution",
        fontsize=13, fontweight="bold",
    )

    order_keywords = ["looming", "air_puff", "base_visual", "base_wind"]
    fallback_colors = [_C_LOOMING, _C_AIRPUFF, _C_BASE_VIS, _C_BASE_WIND]

    data_keys = list(peak_data.keys())
    plot_data = []
    plot_colors = []
    plot_labels = []
    for keyword, fallback_color in zip(order_keywords, fallback_colors):
        matched_key = _find_matching_key(keyword, data_keys)
        if matched_key is not None and peak_data[matched_key]:
            plot_data.append(peak_data[matched_key])
            plot_colors.append(fallback_color)
            plot_labels.append(matched_key.replace("_", " ").title())

    if not plot_data:
        ax.text(
            0.5, 0.5, "No peak velocity data available",
            ha="center", va="center",
            transform=ax.transAxes, fontsize=11, color="gray",
        )
        ax.set_axis_off()
        return fig

    bp = ax.boxplot(
        plot_data,
        labels=plot_labels,
        patch_artist=True,
        widths=0.5,
        showfliers=False,
        medianprops=dict(color="white", linewidth=1.5),
    )
    for patch, color in zip(bp["boxes"], plot_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
        patch.set_edgecolor("white")
        patch.set_linewidth(0.8)
    for whisker in bp["whiskers"]:
        whisker.set_color("#A0A0A5")
        whisker.set_linewidth(0.8)
    for cap in bp["caps"]:
        cap.set_color("#A0A0A5")
        cap.set_linewidth(0.8)

    for i, (data, color) in enumerate(zip(plot_data, plot_colors)):
        jitter = np.random.normal(0, 0.04, size=len(data))
        ax.scatter(
            np.full(len(data), i + 1) + jitter,
            data,
            s=18,
            alpha=0.45,
            color=color,
            edgecolors="white",
            linewidths=0.3,
            zorder=3,
        )

    ax.set_ylabel("Peak Velocity (mm/s)")
    ax.set_xlabel("Stimulus Condition")
    ax.grid(True, alpha=_GRID_ALPHA, axis="y")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Helper: styled card
# ---------------------------------------------------------------------------


def _make_card(title: str, parent: Optional[QWidget] = None) -> Tuple[QFrame, QVBoxLayout]:
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


# ---------------------------------------------------------------------------
# Folder selection row widget
# ---------------------------------------------------------------------------


class _FolderRow(QWidget):
    """A labelled line-edit + browse button for selecting a directory."""

    def __init__(self, label: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._path: Optional[str] = None
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        lbl = QLabel(label)
        lbl.setFixedWidth(170)
        row.addWidget(lbl)

        self._line = QLineEdit()
        self._line.setReadOnly(True)
        self._line.setPlaceholderText("Select folder...")
        row.addWidget(self._line, stretch=1)

        btn = QPushButton("Browse...")
        btn.setFixedWidth(90)
        btn.clicked.connect(self._browse)
        row.addWidget(btn)

    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if path:
            self._path = path
            self._line.setText(path)

    def get_path(self) -> Optional[str]:
        return self._path


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class GroupAnalysisWindow(QDialog):
    """Cross-session group analysis panel.

    Operates as an independent workspace that dispatches
    ``PROCESS_GROUP_BATCH`` to the shared ``DataProcessor`` via the
    multiprocessing command queue.  Telemetry is polled on a QTimer so
    the GUI remains responsive.

    Parameters
    ----------
    cmd_queue : mp.Queue
        Outbound command queue to ``DataProcessor``.
    telemetry_queue : mp.Queue
        Inbound telemetry queue from ``DataProcessor``.
    speed_threshold_mm_s : float
        Escape velocity threshold for response probability computation.
    parent : QWidget or None
        Parent widget.
    """

    def __init__(
        self,
        cmd_queue: mp.Queue,
        telemetry_queue: mp.Queue,
        speed_threshold_mm_s: float = 10.0,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._cmd_queue = cmd_queue
        self._telemetry_queue = telemetry_queue
        self._speed_threshold = speed_threshold_mm_s
        self._sessions: List[SessionResults] = []
        self._figures: Dict[str, Figure] = {}

        self.setWindowTitle("Group Analysis — Baseline Comparison")
        self.setMinimumSize(1200, 700)

        apply_mpl_theme()
        self._init_ui()
        self._start_telemetry_timer()

    # ---- UI construction ---------------------------------------------------

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        # --- I/O card with folder selectors ---
        io_card, io_layout = _make_card("Data Directories")

        self._treatment_row = _FolderRow("Treatment Data Root:")
        io_layout.addWidget(self._treatment_row)

        self._visual_row = _FolderRow("Visual Baseline Data:")
        io_layout.addWidget(self._visual_row)

        self._wind_row = _FolderRow("Wind Baseline Data:")
        io_layout.addWidget(self._wind_row)

        # Run button
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._run_btn = QPushButton("Run Group Analysis")
        self._run_btn.setProperty("accent", True)
        self._run_btn.setMinimumHeight(38)
        self._run_btn.setMinimumWidth(200)
        self._run_btn.clicked.connect(self._on_run)
        btn_row.addWidget(self._run_btn)
        btn_row.addStretch()
        io_layout.addLayout(btn_row)

        root.addWidget(io_card)

        # --- Tab widget for plots ---
        self._tabs = QTabWidget()
        self._tab_canvases: Dict[str, FigureCanvas] = {}
        self._tab_toolbars: Dict[str, NavigationToolbar] = {}

        tab_names = {
            "response_probability": "Response Probability",
            "group_psth": "Group PSTH Comparison",
            "peak_velocity": "Peak Velocity Distribution",
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
            self._tabs.addTab(tab_widget, title)
            self._tab_canvases[key] = canvas
            self._tab_toolbars[key] = toolbar

            # Draw placeholder so tabs render correctly before data arrives
            ax = canvas.figure.add_subplot(111)
            ax.text(
                0.5, 0.5, "No data loaded",
                ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray",
            )
            ax.set_axis_off()
            canvas.draw()

        root.addWidget(self._tabs, stretch=1)

        # --- Status bar ---
        self._progress = QProgressBar()
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        root.addWidget(self._progress)

        self._status_label = QLabel("Select directories and click Run")
        self._status_label.setProperty("status", True)
        root.addWidget(self._status_label)

    # ---- Telemetry polling -------------------------------------------------

    def _start_telemetry_timer(self) -> None:
        self._telemetry_timer = QTimer(self)
        self._telemetry_timer.timeout.connect(self._poll_telemetry)
        self._telemetry_timer.start(50)

    def _poll_telemetry(self) -> None:
        while not self._telemetry_queue.empty():
            try:
                msg: Telemetry = self._telemetry_queue.get_nowait()
                self._handle_telemetry(msg)
            except Exception:
                break

    def _handle_telemetry(self, msg: Telemetry) -> None:
        if msg.status == TelemetryStatus.PROGRESS:
            self._progress.setValue(int(msg.progress * 100))
            self._status_label.setText(f"Processing... {msg.progress:.0%}")

        elif msg.status == TelemetryStatus.COMPLETE:
            self._progress.setValue(100)
            self._run_btn.setEnabled(True)

            if msg.data and "results" in msg.data:
                self._sessions = msg.data["results"]
                self._status_label.setText(
                    f"Complete — {len(self._sessions)} session(s) loaded"
                )
                self._render_figures()
            else:
                self._status_label.setText("Complete")

        elif msg.status == TelemetryStatus.ERROR:
            self._run_btn.setEnabled(True)
            self._progress.setValue(0)
            self._status_label.setText(f"Error: {msg.error or 'Unknown'}")

    # ---- Command dispatch --------------------------------------------------

    def _on_run(self) -> None:
        """Dispatch PROCESS_GROUP_BATCH to DataProcessor."""
        groups: Dict[str, str] = {}
        treatment_path = self._treatment_row.get_path()
        visual_path = self._visual_row.get_path()
        wind_path = self._wind_row.get_path()

        if treatment_path:
            groups["treatment"] = treatment_path
        if visual_path:
            groups["base_visual"] = visual_path
        if wind_path:
            groups["base_wind"] = wind_path

        if not groups:
            self._status_label.setText("Select at least one directory")
            return

        # Use a default ProcessingParams dict; the handler extracts fields
        cmd = Command(
            action=CommandAction.PROCESS_GROUP_BATCH,
            params={
                "processing": {
                    "input_dir": treatment_path or list(groups.values())[0],
                    "output_dir": str(
                        Path(treatment_path or list(groups.values())[0]) / "output"
                    ),
                },
                "groups": groups,
            },
        )
        self._cmd_queue.put(cmd)
        self._run_btn.setEnabled(False)
        self._progress.setValue(0)
        self._status_label.setText("Dispatching group analysis...")

    # ---- Figure rendering --------------------------------------------------

    def _render_figures(self) -> None:
        """Generate and display figures from the loaded sessions."""
        # Close previous figures
        for fig in self._figures.values():
            plt.close(fig)
        self._figures.clear()

        if not self._sessions:
            self._status_label.setText("No sessions to plot")
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

        # Response probability
        prob_data = _compute_response_probability(
            self._sessions, self._speed_threshold,
        )
        prob_fig = _create_response_probability_figure(
            prob_data, self._speed_threshold,
        )
        self._figures["response_probability"] = prob_fig

        # Group PSTH comparison (treatment vs baseline overlay)
        psth_fig = _create_group_psth_figure(self._sessions)
        self._figures["group_psth"] = psth_fig

        # Peak velocity distribution
        peak_data = _extract_peak_velocities(self._sessions)
        peak_fig = _create_peak_velocity_figure(peak_data)
        self._figures["peak_velocity"] = peak_fig

        # Render into tab canvases
        for key, fig in self._figures.items():
            canvas = self._tab_canvases.get(key)
            if canvas is None:
                continue
            old_fig = canvas.figure
            canvas.figure = fig
            fig.set_canvas(canvas)
            toolbar = self._tab_toolbars.get(key)
            if toolbar:
                toolbar.update()
            canvas.draw()
            plt.close(old_fig)

        self._status_label.setText(
            f"Rendered {len(self._figures)} figure(s) from "
            f"{len(self._sessions)} session(s)"
        )

    # ---- Public API --------------------------------------------------------

    def get_figures(self) -> Dict[str, Figure]:
        """Return rendered figures for export."""
        return dict(self._figures)
