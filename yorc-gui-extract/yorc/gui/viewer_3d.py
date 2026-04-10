"""
3D Viewer widget for YORC GUI.

Provides embedded PyVista visualization with point picking support.
"""

import sys
import tempfile
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget
from vtkmodules.vtkRenderingCore import vtkCellPicker

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor

    # Configure PyVista for macOS
    if sys.platform == "darwin":
        # Suppress deprecation warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="pyvista")

        # Desktop Qt app: do not force any Jupyter backend.
        # Setting a notebook backend here can trigger optional IPython/trame
        # dependencies even though this is not a notebook workflow.

        # Optimize for interactive use
        pv.global_theme.interactive = True
        pv.global_theme.smooth_shading = False  # Faster rendering
        pv.global_theme.show_edges = False

        # Set anti-aliasing for better quality without too much overhead
        pv.global_theme.multi_samples = 4

        # Use VTK's native rendering on macOS (Cocoa backend)
        import os

        os.environ["VTK_DEFAULT_RENDER_WINDOW_TYPE"] = "cocoa"

    PYVISTA_AVAILABLE = True
    PYVISTA_IMPORT_ERROR = ""
except ImportError as e:
    PYVISTA_AVAILABLE = False
    PYVISTA_IMPORT_ERROR = str(e)


DEBUG_LOG_PATH = Path(tempfile.gettempdir()) / "yorc_tripanel_debug.log"


def _viewer_debug_log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    line = f"[{timestamp}] [Viewer3D] {message}"
    print(line)
    try:
        with DEBUG_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(f"{line}\n")
    except OSError:
        pass


class Viewer3D(QWidget):
    """3D visualization widget with point picking support."""

    point_picked = pyqtSignal(object, int)  # point coordinates, point index
    points_confirmed = pyqtSignal(list)  # list of picked points

    def __init__(self, parent=None):
        super().__init__(parent)
        self.geometries = {}  # name -> actor mapping
        self.picked_points = []
        self.pick_markers = []
        self.picking_enabled = False
        self.expected_points = 0
        self.picking_mode = None
        self.pick_clear_callback = None
        self.plotter = None  # Lazy initialization
        self._plotter_container = None
        self._pending_add = []  # Queue for deferred mesh additions
        self._processing_pending = False  # Flag to prevent concurrent processing
        self._vtk_picker = None
        self._left_click_observer = None
        self._right_click_observer = None

        self._setup_ui()

    def _setup_ui(self):
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        if not PYVISTA_AVAILABLE:
            error_text = (
                "PyVista not available for 3D viewer.\n\n"
                f"Python executable: {sys.executable}\n"
                f"Import error: {PYVISTA_IMPORT_ERROR}\n\n"
                "Install in this environment with:\n"
                "  uv pip install --python .venv/bin/python pyvista pyvistaqt vtk\n"
                "Then launch with:\n"
                "  source .venv/bin/activate && uv run yorc-tripanel-gui"
            )
            label = QLabel(error_text)
            label.setWordWrap(True)
            self._layout.addWidget(label)
            _viewer_debug_log(
                f"{self.objectName() or 'unnamed'} PyVista unavailable: {PYVISTA_IMPORT_ERROR}"
            )
            return

        # Placeholder label - plotter will be created lazily
        self._placeholder = QLabel("3D Viewer - Load a mesh to initialize")
        self._placeholder.setMinimumSize(400, 300)
        self._placeholder.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        self._layout.addWidget(self._placeholder)

        # Control bar (created now, but plotter initialized later)
        control_layout = QHBoxLayout()

        self.pick_mode_combo = QComboBox()
        self.pick_mode_combo.addItems(
            [
                "View Only",
                "Pick Anatomical Points",
                "Pick Helmet Landmarks",
                "Pick Facial Landmarks",
            ]
        )
        self.pick_mode_combo.currentIndexChanged.connect(self._on_pick_mode_changed)
        control_layout.addWidget(QLabel("Mode:"))
        control_layout.addWidget(self.pick_mode_combo)

        control_layout.addStretch()

        self.reset_btn = QPushButton("Reset View")
        self.reset_btn.clicked.connect(self.reset_camera)
        control_layout.addWidget(self.reset_btn)

        self.clear_picks_btn = QPushButton("Clear Picks")
        self.clear_picks_btn.clicked.connect(lambda: self.clear_picked_points(notify=True))
        self.clear_picks_btn.setEnabled(False)
        control_layout.addWidget(self.clear_picks_btn)

        self.confirm_btn = QPushButton("Confirm Points")
        self.confirm_btn.clicked.connect(self._on_confirm_points)
        self.confirm_btn.setEnabled(False)
        control_layout.addWidget(self.confirm_btn)

        self._layout.addLayout(control_layout)

        # Status label for picking
        self.pick_status = QLabel("")
        self._layout.addWidget(self.pick_status)

    def _init_plotter(self):
        """Initialize PyVista plotter lazily."""
        if self.plotter is not None:
            return True

        if not PYVISTA_AVAILABLE:
            _viewer_debug_log(
                f"{self.objectName() or 'unnamed'} _init_plotter aborted: PyVista unavailable"
            )
            return False

        _viewer_debug_log(f"{self.objectName() or 'unnamed'} _init_plotter start")

        # Remove placeholder
        if self._placeholder is not None:
            self._layout.removeWidget(self._placeholder)
            self._placeholder.deleteLater()
            self._placeholder = None

        # Create plotter with macOS-optimized settings
        self.plotter = QtInteractor(self)
        self.plotter.set_background("#101215")
        self.plotter.add_axes()
        self.plotter.enable_trackball_style()

        # macOS-specific optimizations
        if sys.platform == "darwin":
            # Disable problematic features on macOS
            self.plotter.enable_render_on_set_window_size_event = False

            # Set reasonable initial window size to avoid crashes
            self.plotter.setMinimumSize(400, 300)

            # Use faster rendering mode
            if hasattr(self.plotter.ren_win, "SetMultiSamples"):
                self.plotter.ren_win.SetMultiSamples(0)  # Disable for speed
        else:
            # Disable auto-rendering to prevent blocking on large meshes
            self.plotter.enable_render_on_set_window_size_event = False

        # Insert at position 0 (before controls)
        self._layout.insertWidget(0, self.plotter)
        self._vtk_picker = vtkCellPicker()
        self._vtk_picker.SetTolerance(0.002)

        interactor = self.plotter.iren if hasattr(self.plotter, "iren") else None
        if interactor is not None:
            if self._left_click_observer is None:
                self._left_click_observer = interactor.add_observer(
                    "LeftButtonPressEvent", self._on_left_click
                )
            if self._right_click_observer is None:
                self._right_click_observer = interactor.add_observer(
                    "RightButtonPressEvent", self._on_right_click
                )

        _viewer_debug_log(f"{self.objectName() or 'unnamed'} _init_plotter done")

        return True

    def ensure_plotter_initialized(self) -> bool:
        """Public wrapper to ensure QtInteractor is created and placeholder removed."""
        ok = self._init_plotter()
        _viewer_debug_log(f"{self.objectName() or 'unnamed'} ensure_plotter_initialized -> {ok}")
        return ok

    def add_debug_anchor(self, point, name="debug_anchor", radius=6.0, color="tomato"):
        if not self._init_plotter():
            return

        import pyvista as pv

        try:
            center = np.asarray(point, dtype=float).reshape(3)
            sphere = pv.Sphere(radius=radius, center=center)
            self.remove_geometry(name)
            actor = self.plotter.add_mesh(sphere, color=color, name=name, render=False)
            self.geometries[name] = {"actor": actor, "data": sphere}
            _viewer_debug_log(
                f"{self.objectName() or 'unnamed'} added debug anchor {name} at {center.tolist()}"
            )
        except Exception as exc:
            _viewer_debug_log(
                f"{self.objectName() or 'unnamed'} failed to add debug anchor {name}: {exc}"
            )

    def debug_state(self, label: str = "state") -> None:
        name = self.objectName() or "unnamed"
        plotter_ready = self.plotter is not None
        _viewer_debug_log(
            f"{name} {label}: visible={self.isVisible()} size={self.width()}x{self.height()} "
            f"plotter_ready={plotter_ready} geometries={list(self.geometries.keys())}"
        )

        if not plotter_ready:
            return

        for geom_name, geom in self.geometries.items():
            data = geom.get("data")
            n_points = getattr(data, "n_points", "?")
            bounds = getattr(data, "bounds", None)
            actor = geom.get("actor")
            actor_visible = None
            if actor is not None and hasattr(actor, "GetVisibility"):
                actor_visible = actor.GetVisibility()
            _viewer_debug_log(
                f"{name} {label}: geom={geom_name} points={n_points} bounds={bounds} actor_visible={actor_visible}"
            )

        try:
            _viewer_debug_log(f"{name} {label}: camera_position={self.plotter.camera_position}")
        except Exception as exc:
            _viewer_debug_log(f"{name} {label}: camera_position unavailable: {exc}")

    def add_pyvista_points(
        self, pv_data, name="cloud", color=None, point_size=4.0, auto_render=True
    ):
        """Add a PyVista PolyData point cloud to the viewer (already converted)."""
        import time

        _viewer_debug_log(f"{self.objectName() or 'unnamed'} add_pyvista_points {name} START")
        t0 = time.time()

        if not self._init_plotter():
            _viewer_debug_log(
                f"{self.objectName() or 'unnamed'} add_pyvista_points {name} aborted: init failed"
            )
            return

        # Remove existing geometry with same name
        self.remove_geometry(name)

        # Intelligently decimate large meshes to prevent crashes
        n_points = pv_data.n_points
        _viewer_debug_log(f"{self.objectName() or 'unnamed'} {name} original_points={n_points:,}")

        # Adaptive decimation based on point count
        if n_points > 100000:  # 100k points
            target_points = 50000  # Aggressive decimation for very large meshes
            _viewer_debug_log(
                f"{self.objectName() or 'unnamed'} {name} decimating to {target_points:,} points"
            )
            decimation_ratio = target_points / n_points

            # Use PyVista's decimation for point clouds
            if decimation_ratio < 1.0:
                import random

                random.seed(42)  # Reproducible sampling
                indices = random.sample(range(n_points), target_points)
                points = pv_data.points[indices]
                pv_data = pv.PolyData(points)
                _viewer_debug_log(
                    f"{self.objectName() or 'unnamed'} {name} decimated_points={pv_data.n_points:,}"
                )

        # Assign distinct colors per mesh type
        color_map = {
            "outside_mesh": "lightblue",
            "inside_mesh": "lightgreen",
            "mri_scalp": "lightyellow",
        }
        mesh_color = color if color else color_map.get(name, "gray")

        # Add points directly (we're already on main thread from signal handler)
        _viewer_debug_log(
            f"{self.objectName() or 'unnamed'} adding {name} color={mesh_color} point_size={point_size} bounds={pv_data.bounds}"
        )
        actor = self.plotter.add_points(
            pv_data,
            color=mesh_color,
            point_size=point_size,
            render_points_as_spheres=True,
            name=name,
            render=False,
        )
        self.geometries[name] = {"actor": actor, "data": pv_data}

        # Don't render immediately - will render after all meshes added
        if auto_render:
            _viewer_debug_log(
                f"{self.objectName() or 'unnamed'} {name} added; scheduling deferred render"
            )
            # Cancel any pending render timer and schedule a new one
            # This way multiple meshes arriving quickly will only render once
            if hasattr(self, "_render_timer"):
                self._render_timer.stop()
            self._render_timer = QTimer()
            self._render_timer.setSingleShot(True)
            self._render_timer.timeout.connect(self._do_final_render)
            self._render_timer.start(500)  # Render 500ms after last mesh
        else:
            _viewer_debug_log(
                f"{self.objectName() or 'unnamed'} {name} added; auto_render disabled"
            )

        t1 = time.time()
        _viewer_debug_log(
            f"{self.objectName() or 'unnamed'} add_pyvista_points {name} DONE in {t1 - t0:.2f}s"
        )
        self.debug_state("post-add")

    def add_point_cloud(
        self, o3d_cloud, name="cloud", color=None, point_size=4.0, auto_render=True
    ):
        """
        Add an Open3D point cloud to the viewer.

        Args:
            o3d_cloud: Open3D PointCloud object
            name: Name for the geometry
            color: Optional RGB color (0-1 scale)
            point_size: Point rendering size
        """
        if not self._init_plotter():
            return

        import pyvista as pv

        # Simple conversion - just points, skip colors for speed
        points = np.asarray(o3d_cloud.points)
        pv_data = pv.PolyData(points)

        # Use the optimized method
        self.add_pyvista_points(
            pv_data, name=name, color=color, point_size=point_size, auto_render=auto_render
        )

    def add_mesh(self, o3d_mesh, name="mesh", color=None, opacity=1.0, auto_render=True):
        """
        Add an Open3D mesh to the viewer.

        Args:
            o3d_mesh: Open3D TriangleMesh object
            name: Name for the geometry
            color: Optional RGB color (0-1 scale)
            opacity: Mesh opacity (0-1)
        """
        if not self._init_plotter():
            return

        from ..utils.geometry import o3d_to_pyvista

        pv_data = o3d_to_pyvista(o3d_mesh)

        # Remove existing geometry with same name
        self.remove_geometry(name)

        # Add to plotter
        if color is not None:
            actor = self.plotter.add_mesh(
                pv_data, color=color, opacity=opacity, smooth_shading=True, name=name, render=False
            )
        elif "RGB" in pv_data.array_names:
            actor = self.plotter.add_mesh(
                pv_data,
                scalars="RGB",
                rgb=True,
                opacity=opacity,
                smooth_shading=True,
                name=name,
                render=False,
            )
        else:
            actor = self.plotter.add_mesh(
                pv_data, color="gray", opacity=opacity, smooth_shading=True, name=name, render=False
            )

        self.geometries[name] = {"actor": actor, "data": pv_data}

        # Defer rendering to avoid blocking
        if auto_render:
            QTimer.singleShot(0, self._deferred_update)

    def remove_geometry(self, name):
        """Remove a geometry by name."""
        if self.plotter is None:
            return

        if name in self.geometries:
            self.plotter.remove_actor(self.geometries[name]["actor"])
            del self.geometries[name]

    def clear_all(self):
        """Remove all geometries."""
        if self.plotter is None:
            return

        for name in list(self.geometries.keys()):
            self.remove_geometry(name)
        self.clear_picked_points(notify=False)

    def _deferred_update(self):
        """Perform rendering and camera reset in a deferred manner."""
        import time

        print("[Viewer3D] _deferred_update START")

        if self.plotter is None:
            print("[Viewer3D] Plotter is None")
            return

        print("[Viewer3D] Calling reset_camera")
        t0 = time.time()
        self.plotter.reset_camera()
        t1 = time.time()
        print(f"[Viewer3D] reset_camera returned in {t1 - t0:.2f}s")

        print("[Viewer3D] Calling render")
        t0 = time.time()
        self.plotter.render()
        t1 = time.time()
        print(f"[Viewer3D] render returned in {t1 - t0:.2f}s")
        print("[Viewer3D] _deferred_update DONE")

    def _process_pending_queue(self):
        """Process one item from the pending queue."""
        import time

        print("[Viewer3D] _process_pending_queue START")
        print(f"[Viewer3D] Queue length: {len(self._pending_add)}")

        if not self._pending_add or self.plotter is None:
            print("[Viewer3D] Queue empty or plotter is None")
            self._processing_pending = False
            return

        self._processing_pending = True
        data = self._pending_add.pop(0)  # Get first item from queue
        print(f"[Viewer3D] Processing {data['name']}, {len(self._pending_add)} remaining")

        print(f"[Viewer3D] Calling plotter.add_points for {data['name']}")
        t0 = time.time()
        actor = self.plotter.add_points(
            data["pv_data"],
            color=data["mesh_color"],
            point_size=data["point_size"],
            render_points_as_spheres=False,
            name=data["name"],
            render=False,
        )
        t1 = time.time()
        print(f"[Viewer3D] add_points returned in {t1 - t0:.2f}s")

        self.geometries[data["name"]] = {"actor": actor, "data": data["pv_data"]}

        # Schedule rendering, then continue with queue
        print("[Viewer3D] Scheduling _deferred_update")
        QTimer.singleShot(0, self._deferred_update_and_continue)
        print("[Viewer3D] _process_pending_queue DONE")

    def _deferred_update_and_continue(self):
        """Perform rendering and then continue processing queue."""
        print("[Viewer3D] _deferred_update_and_continue START")
        self._deferred_update()

        # Mark as not processing so new items can be scheduled
        self._processing_pending = False

        # Continue processing queue if there are more items
        if self._pending_add:
            print(f"[Viewer3D] {len(self._pending_add)} items still in queue, scheduling next")
            self._processing_pending = True  # Set again before scheduling
            QTimer.singleShot(0, self._process_pending_queue)
        else:
            print("[Viewer3D] Queue empty, processing complete")
        print("[Viewer3D] _deferred_update_and_continue DONE")

    def _do_final_render(self):
        """Perform final render after all meshes are added."""
        _viewer_debug_log(f"{self.objectName() or 'unnamed'} _do_final_render START")
        if self.plotter is not None:
            _viewer_debug_log(f"{self.objectName() or 'unnamed'} _do_final_render reset_camera")
            self.plotter.reset_camera()
            _viewer_debug_log(f"{self.objectName() or 'unnamed'} _do_final_render render")
            self.plotter.render()
            _viewer_debug_log(f"{self.objectName() or 'unnamed'} _do_final_render DONE")
            self.debug_state("post-final-render")

    def reset_camera(self):
        """Reset camera to fit all geometries."""
        if self.plotter is None:
            return
        _viewer_debug_log(f"{self.objectName() or 'unnamed'} reset_camera start")
        self.plotter.reset_camera()
        self.plotter.render()
        _viewer_debug_log(f"{self.objectName() or 'unnamed'} reset_camera done")

    def render(self):
        """Render current scene without camera reset."""
        if self.plotter is None:
            return
        self.plotter.render()

    def enable_picking(self, mode, num_points, callback=None, clear_callback=None):
        """
        Enable point picking mode.

        Args:
            mode: Picking mode name
            num_points: Expected number of points to pick
            callback: Optional callback when all points are picked
        """
        if not self._init_plotter():
            return

        self.picking_mode = mode
        self.expected_points = num_points
        self.picking_enabled = True
        self.pick_callback = callback
        self.pick_clear_callback = clear_callback
        self.picked_points = []
        self._clear_pick_markers()

        self._update_pick_status()
        self.clear_picks_btn.setEnabled(True)
        self.confirm_btn.setEnabled(False)

    def disable_picking(self):
        """Disable point picking mode."""
        if self.plotter is None:
            return

        self.picking_enabled = False
        self.pick_status.setText("")

    def _on_left_click(self, obj, event):
        if not self.picking_enabled:
            return
        if obj is None or obj.GetShiftKey() != 1:
            return
        if self._vtk_picker is None or self.plotter is None:
            return

        click_pos = obj.GetEventPosition()
        renderer = self.plotter.renderer
        picked = self._vtk_picker.Pick(click_pos[0], click_pos[1], 0, renderer)
        if not picked:
            return

        point = np.array(self._vtk_picker.GetPickPosition(), dtype=float)
        self.picked_points.append(np.array(point))

        # Add visual marker
        sphere = pv.Sphere(radius=3.0, center=point)
        actor = self.plotter.add_mesh(
            sphere, color="yellow", name=f"pick_marker_{len(self.picked_points)}"
        )
        self.pick_markers.append(actor)

        self._update_pick_status()

        # Emit signal
        self.point_picked.emit(point, len(self.picked_points) - 1)

        # Check if we have enough points
        if len(self.picked_points) >= self.expected_points:
            self.confirm_btn.setEnabled(True)

    def _on_right_click(self, obj, event):
        if not self.picking_enabled:
            return
        if obj is None or obj.GetShiftKey() != 1:
            return
        self._remove_last_pick()

    def _remove_last_pick(self):
        if not self.picked_points:
            return

        idx = len(self.picked_points)
        self.picked_points.pop()
        if self.plotter is not None:
            try:
                self.plotter.remove_actor(f"pick_marker_{idx}")
            except Exception:
                pass
        if self.pick_markers:
            self.pick_markers.pop()
        self._update_pick_status()
        self.confirm_btn.setEnabled(len(self.picked_points) >= self.expected_points)

    def _on_pick_mode_changed(self, index):
        """Handle picking mode change."""
        if index == 0:  # View Only
            self.disable_picking()
        elif index == 1:  # Anatomical Points
            self.enable_picking("anatomical", 3)
        elif index == 2:  # Helmet Landmarks
            self.enable_picking("helmet", 7)
        elif index == 3:  # Facial Landmarks
            self.enable_picking("facial", 3)

    def _on_confirm_points(self):
        """Handle confirm points button."""
        if len(self.picked_points) > 0:
            self.points_confirmed.emit(self.picked_points.copy())
            if self.pick_callback:
                self.pick_callback(self.picked_points.copy())

    def _update_pick_status(self):
        """Update the picking status label."""
        if self.picking_enabled:
            self.pick_status.setText(
                f"Picking {self.picking_mode} points: "
                f"{len(self.picked_points)}/{self.expected_points}"
            )
        else:
            self.pick_status.setText("")

    def clear_picked_points(self, notify=True):
        """Clear all picked points."""
        self.picked_points = []
        self._clear_pick_markers()
        self._update_pick_status()
        self.confirm_btn.setEnabled(False)
        if notify and self.pick_clear_callback is not None:
            self.pick_clear_callback()

    def _clear_pick_markers(self):
        """Remove all pick marker actors."""
        if self.plotter is None:
            return

        for i, _ in enumerate(self.pick_markers):
            try:
                self.plotter.remove_actor(f"pick_marker_{i + 1}")
            except Exception:
                pass
        self.pick_markers = []

    def add_sphere_markers(self, points, radius=3.0, color="red", name="markers"):
        """
        Add sphere markers at specified points.

        Args:
            points: Nx3 array of positions
            radius: Sphere radius
            color: Marker color
            name: Name prefix for markers
        """
        if not self._init_plotter():
            return

        for i, point in enumerate(points):
            sphere = pv.Sphere(radius=radius, center=point)
            self.plotter.add_mesh(sphere, color=color, name=f"{name}_{i}")

    def highlight_registration_result(
        self, source_cloud, target_cloud, source_color="blue", target_color="red"
    ):
        """
        Display two clouds for registration comparison.

        Args:
            source_cloud: Transformed source cloud
            target_cloud: Target cloud
            source_color: Color for source
            target_color: Color for target
        """
        self.add_point_cloud(source_cloud, name="source_result", color=source_color)
        self.add_point_cloud(target_cloud, name="target_result", color=target_color)

    def close(self):
        """Clean up resources."""
        if PYVISTA_AVAILABLE and hasattr(self, "plotter"):
            self.plotter.close()
