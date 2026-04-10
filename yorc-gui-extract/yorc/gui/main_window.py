"""
Main window for YORC GUI.

Provides the main application window with file selection, 3D visualization,
and workflow controls.
"""

import numpy as np
from PyQt6.QtCore import QObject, QRunnable, Qt, QThread, QThreadPool, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .file_panel import FilePanel
from .viewer_3d import Viewer3D
from .workflow_panel import WorkflowPanel


class MeshLoaderSignals(QObject):
    """Signals for mesh loader."""

    finished = pyqtSignal(str, object, str)  # file_type, pyvista_data, error_msg
    progress = pyqtSignal(str)  # status message


class MeshLoader(QRunnable):
    """Background worker for loading mesh files."""

    def __init__(self, file_type, file_path):
        super().__init__()
        self.file_type = file_type
        self.file_path = file_path
        self.signals = MeshLoaderSignals()

    def run(self):
        import time

        try:
            import pyvista as pv

            from ..core.io_utils import load_mesh_for_display

            print(f"[MeshLoader] START {self.file_type}")
            self.signals.progress.emit(f"Loading {self.file_type}...")

            # Load mesh in background with progress updates
            t0 = time.time()
            self.signals.progress.emit(f"Reading {self.file_type} file...")
            o3d_cloud, _ = load_mesh_for_display(self.file_path)
            t1 = time.time()
            n_points = len(o3d_cloud.points)
            print(f"[MeshLoader] Loaded {n_points:,} points in {t1 - t0:.2f}s")

            # Convert to PyVista in background
            t0 = time.time()
            self.signals.progress.emit(f"Converting {n_points:,} points to display format...")
            points = np.asarray(o3d_cloud.points)
            pv_data = pv.PolyData(points)
            t1 = time.time()
            print(f"[MeshLoader] Created PyVista in {t1 - t0:.2f}s")

            self.signals.progress.emit(f"Loaded {self.file_type}: {n_points:,} points")
            print("[MeshLoader] Emitting signal")
            self.signals.finished.emit(self.file_type, pv_data, "")
            print("[MeshLoader] DONE")
        except Exception as e:
            print(f"[MeshLoader] ERROR: {e}")
            import traceback

            traceback.print_exc()
            self.signals.finished.emit(self.file_type, None, str(e))


class RegistrationWorker(QThread):
    """Worker thread for running registration pipeline."""

    progress = pyqtSignal(str, int)  # message, percent
    step_completed = pyqtSignal(int, bool, str)  # step_index, success, message
    finished = pyqtSignal(bool, str)  # success, message
    error = pyqtSignal(str)

    def __init__(self, pipeline, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            self.pipeline.run(
                progress_callback=self._on_progress,
                step_callback=self._on_step_completed,
                cancel_check=lambda: self._is_cancelled,
            )
            if not self._is_cancelled:
                self.finished.emit(True, "Registration completed successfully")
        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit(False, str(e))

    def _on_progress(self, message, percent):
        self.progress.emit(message, percent)

    def _on_step_completed(self, step_index, success, message):
        self.step_completed.emit(step_index, success, message)


class MainWindow(QMainWindow):
    """Main application window for YORC GUI."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("YORC - York OPM Registration Code")
        self.setMinimumSize(1200, 800)

        # State
        self.pipeline = None
        self.worker = None
        self.thread_pool = QThreadPool()
        self.loaded_clouds = {}  # Store loaded point clouds
        self.loaded_data = {
            "outside_mesh": None,
            "inside_mesh": None,
            "mri_scalp": None,
            "meg_data": [],
            "subjects_dir": None,
            "subject": None,
        }

        self._setup_ui()
        self._setup_menu()
        self._connect_signals()

    def _setup_ui(self):
        """Set up the main UI layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel: File selection + Workflow
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.file_panel = FilePanel()
        self.workflow_panel = WorkflowPanel()

        left_layout.addWidget(self.file_panel)
        left_layout.addWidget(self.workflow_panel)
        left_layout.addStretch()

        # Right panel: 3D Viewer
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.viewer = Viewer3D()
        right_layout.addWidget(self.viewer)

        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([350, 850])

        main_layout.addWidget(splitter)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label)

    def _setup_menu(self):
        """Set up the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        load_action = QAction("&Load Files...", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self._on_load_files)
        file_menu.addAction(load_action)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        reset_view_action = QAction("&Reset View", self)
        reset_view_action.setShortcut("R")
        reset_view_action.triggered.connect(self._on_reset_view)
        view_menu.addAction(reset_view_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About YORC", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    def _connect_signals(self):
        """Connect signals between components."""
        # File panel signals
        self.file_panel.file_loaded.connect(self._on_file_loaded)
        self.file_panel.file_cleared.connect(self._on_file_cleared)

        # Workflow panel signals
        self.workflow_panel.run_automatic.connect(self._on_run_automatic)
        self.workflow_panel.run_step.connect(self._on_run_step)
        self.workflow_panel.cancel_requested.connect(self._on_cancel)

        # Viewer signals
        self.viewer.point_picked.connect(self._on_point_picked)

    def _on_file_loaded(self, file_type, file_path):
        """Handle file loaded event."""
        self.loaded_data[file_type] = file_path

        # Load and display mesh in viewer (asynchronously in background thread)
        if file_type in ["outside_mesh", "inside_mesh", "mri_scalp"]:
            self._update_status(f"Loading {file_type}...")

            # Use background thread to avoid blocking GUI with large meshes
            loader = MeshLoader(file_type, file_path)
            loader.signals.progress.connect(self._update_status)
            loader.signals.finished.connect(self._on_mesh_loaded)
            self.thread_pool.start(loader)
        else:
            # Update workflow panel state for non-mesh files
            self._update_workflow_state()

    def _on_mesh_loaded(self, file_type, pv_data, error_msg):
        """Handle mesh loaded event from background thread."""
        print(f"[MainWindow] _on_mesh_loaded {file_type}")

        if error_msg:
            print(f"[MainWindow] ERROR: {error_msg}")
            self._show_error(f"Failed to load {file_type}: {error_msg}")
            self.loaded_data[file_type] = None
        else:
            print("[MainWindow] Storing cloud, scheduling viewer update")
            self.loaded_clouds[file_type] = pv_data
            # Defer the entire add_points call to the main thread
            from PyQt6.QtCore import QTimer

            self._update_status(f"Adding {file_type} to viewer ({pv_data.n_points} points)...")
            print("[MainWindow] QTimer.singleShot scheduling _add_mesh_to_viewer")
            QTimer.singleShot(0, lambda: self._add_mesh_to_viewer(file_type, pv_data))
            print("[MainWindow] QTimer scheduled")

        # Update workflow panel state after loading completes
        self._update_workflow_state()
        print("[MainWindow] _on_mesh_loaded DONE")

    def _add_mesh_to_viewer(self, file_type, pv_data):
        """Add mesh to viewer on main thread."""
        import time

        print(f"[MainWindow] _add_mesh_to_viewer {file_type} START")
        t0 = time.time()
        self.viewer.add_pyvista_points(pv_data, name=file_type)
        t1 = time.time()
        print(f"[MainWindow] add_pyvista_points returned in {t1 - t0:.2f}s")
        self._update_status(f"Displayed: {file_type}")
        print("[MainWindow] _add_mesh_to_viewer DONE")

    def _on_file_cleared(self, file_type):
        """Handle file cleared event."""
        self.loaded_data[file_type] = None
        self.viewer.remove_geometry(file_type)
        self._update_workflow_state()

    def _on_run_automatic(self):
        """Run the full automatic registration pipeline."""
        if not self._validate_inputs():
            return

        self._start_pipeline(automatic=True)

    def _on_run_step(self, step_index):
        """Run a specific workflow step."""
        if not self._validate_inputs(step_index):
            return

        self._start_pipeline(automatic=False, step=step_index)

    def _on_cancel(self):
        """Cancel the current operation."""
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self._update_status("Cancelling...")

    def _start_pipeline(self, automatic=True, step=None):
        """Start the registration pipeline."""
        from .pipeline import RegistrationPipeline

        self.pipeline = RegistrationPipeline(
            outside_mesh=self.loaded_data["outside_mesh"],
            inside_mesh=self.loaded_data["inside_mesh"],
            mri_scalp=self.loaded_data["mri_scalp"],
            meg_data=self.loaded_data["meg_data"],
            subjects_dir=self.loaded_data.get("subjects_dir"),
            subject=self.loaded_data.get("subject"),
            automatic=automatic,
            step=step,
        )

        self.worker = RegistrationWorker(self.pipeline)
        self.worker.progress.connect(self._on_progress)
        self.worker.step_completed.connect(self._on_step_completed)
        self.worker.finished.connect(self._on_pipeline_finished)
        self.worker.error.connect(self._on_pipeline_error)

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.workflow_panel.set_running(True)

        self.worker.start()

    def _on_progress(self, message, percent):
        """Handle progress update."""
        self.progress_bar.setValue(percent)
        self._update_status(message)

    def _on_step_completed(self, step_index, success, message):
        """Handle step completion."""
        self.workflow_panel.set_step_status(step_index, success, message)

        # Update viewer with intermediate results if available
        if self.pipeline and hasattr(self.pipeline, "get_step_result"):
            result = self.pipeline.get_step_result(step_index)
            if result is not None:
                self._display_step_result(step_index, result)

    def _on_pipeline_finished(self, success, message):
        """Handle pipeline completion."""
        self.progress_bar.setVisible(False)
        self.workflow_panel.set_running(False)

        if success:
            self._update_status("Registration completed successfully")
            QMessageBox.information(self, "Success", message)
        else:
            self._update_status(f"Registration failed: {message}")

    def _on_pipeline_error(self, error_message):
        """Handle pipeline error."""
        self.progress_bar.setVisible(False)
        self.workflow_panel.set_running(False)
        self._show_error(error_message)

    def _on_point_picked(self, point, point_index):
        """Handle point picked in viewer."""
        self._update_status(f"Picked point: ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})")

    def _display_step_result(self, step_index, result):
        """Display result of a registration step in the viewer."""
        # Implementation depends on what each step returns
        pass

    def _validate_inputs(self, step=None):
        """Validate that required inputs are loaded."""
        required = ["outside_mesh", "inside_mesh", "mri_scalp"]

        missing = []
        for key in required:
            if not self.loaded_data.get(key):
                missing.append(key.replace("_", " "))

        if not self.loaded_data.get("meg_data"):
            missing.append("MEG data")

        if missing:
            self._show_error("Missing required files:\n- " + "\n- ".join(missing))
            return False

        return True

    def _update_workflow_state(self):
        """Update workflow panel based on loaded files."""
        has_required = all(
            [
                self.loaded_data.get("outside_mesh"),
                self.loaded_data.get("inside_mesh"),
                self.loaded_data.get("mri_scalp"),
                self.loaded_data.get("meg_data"),
            ]
        )
        self.workflow_panel.set_enabled(has_required)

    def _update_status(self, message):
        """Update status bar message."""
        self.status_label.setText(message)

    def _show_error(self, message):
        """Show error dialog."""
        QMessageBox.critical(self, "Error", message)

    def _on_load_files(self):
        """Open file browser for loading files."""
        # Delegate to file panel
        self.file_panel.browse_all()

    def _on_reset_view(self):
        """Reset the 3D viewer."""
        self.viewer.reset_camera()

    def _on_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About YORC",
            "<h3>YORC - York OPM Registration Code</h3>"
            "<p>Version 0.2.0</p>"
            "<p>A tool for co-registering OPM-MEG sensor positions "
            "with MRI brain scans.</p>"
            "<p>Developed by York Neuroimaging Centre, University of York</p>"
            "<p>Authors: Richard Aveyard, Alex Wade, Joe Lyons</p>",
        )

    def closeEvent(self, event):
        """Handle window close."""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Confirm Exit",
                "Registration is still running. Are you sure you want to exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return

            self.worker.cancel()
            self.worker.wait(5000)

        # Clean up viewer
        self.viewer.close()
        event.accept()
