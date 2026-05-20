"""Entry point for Cercus Analysis application.

Two modes:
  python main.py              — Launch main ethological analysis (GUI + DataProcessor)
  python main.py --calibrate  — Launch standalone calibration visualizer

Main mode spawns DataProcessor in an isolated process. GUI runs in main process.
Communication exclusively via multiprocessing.Queue.

Calibration mode runs entirely in the main process (no heavy computation needed).
"""

import argparse
import logging
import multiprocessing as mp
import sys

from PyQt5.QtWidgets import QApplication

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _run_processor(cmd_queue: mp.Queue, telemetry_queue: mp.Queue) -> None:
    """Target function for DataProcessor process."""
    from src.processors.data_processor import DataProcessor
    processor = DataProcessor(cmd_queue, telemetry_queue)
    processor.run()


def run_main_app() -> int:
    """Launch main ethological analysis: GUI + DataProcessor in separate processes."""
    from src.gui.main_window import MainWindow

    # IPC queues — the only legal cross-process channel
    cmd_queue: mp.Queue = mp.Queue()
    telemetry_queue: mp.Queue = mp.Queue()

    # Spawn DataProcessor in isolated process
    processor_proc = mp.Process(
        target=_run_processor,
        args=(cmd_queue, telemetry_queue),
        name="DataProcessor",
        daemon=True,
    )
    processor_proc.start()
    logger.info("DataProcessor process started (PID=%d)", processor_proc.pid)

    # Run GUI in main process
    app = QApplication(sys.argv)
    window = MainWindow(cmd_queue, telemetry_queue)
    window.show()
    exit_code = app.exec_()

    # Cleanup
    if processor_proc.is_alive():
        processor_proc.join(timeout=5.0)
    if processor_proc.is_alive():
        processor_proc.terminate()
    logger.info("Application exit (code=%d)", exit_code)
    return exit_code


def run_calibration_app() -> int:
    """Launch standalone calibration visualizer. No multiprocessing needed."""
    from src.gui.calibration_window import CalibrationWindow

    app = QApplication(sys.argv)
    window = CalibrationWindow()
    window.show()
    exit_code = app.exec_()
    logger.info("Calibration exit (code=%d)", exit_code)
    return exit_code


def main() -> int:
    """Parse arguments and launch appropriate mode."""
    parser = argparse.ArgumentParser(
        description="Cercus Analysis — Locomotor Tracking from Spherical Treadmill",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Launch standalone calibration visualizer (hardware validation)",
    )
    args = parser.parse_args()

    if args.calibrate:
        return run_calibration_app()
    else:
        return run_main_app()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    sys.exit(main())
