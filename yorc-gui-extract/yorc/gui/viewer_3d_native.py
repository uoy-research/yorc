"""Native VTK 3D viewer widget for YORC tri-panel GUI.

Provides a rendering path that avoids pyvistaqt while preserving
the minimal Viewer3D API used by the registration workflow.
"""

from __future__ import annotations

import tempfile
import time
from datetime import datetime

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkCommonCore import vtkPoints, vtkUnsignedCharArray
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.vtkRenderingCore import vtkActor, vtkCellPicker, vtkPolyDataMapper, vtkRenderer

DEBUG_LOG_PATH = tempfile.gettempdir() + "/yorc_tripanel_debug.log"


class _NoAutoPaintVTKWidget(QVTKRenderWindowInteractor):
    """QVTKRenderWindowInteractor with the macOS display-link paint loop throttled.

    QVTKRenderWindowInteractor sets WA_PaintOnScreen in its __init__, which causes
    macOS's CADisplayLink to drive paintEvent() at 60 Hz per panel. With three panels
    this saturates the CPU (~100%) and starves the Qt event loop so QTimer callbacks
    never fire. We throttle paintEvent() to ≤20 fps per panel to keep the display
    responsive while capping idle-render CPU. CreateTimer/DestroyTimer are overridden
    to suppress VTK's 10 ms idle timer (the trackball camera style, UseTimers=0, does
    not need this timer for mouse-driven rotation).
    """

    _PAINT_INTERVAL = 0.05  # 20 fps max

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_paint_time = 0.0

    def paintEvent(self, ev):
        now = time.monotonic()
        if now - self._last_paint_time < self._PAINT_INTERVAL:
            return
        self._last_paint_time = now
        super().paintEvent(ev)

    def CreateTimer(self, obj, evt):
        pass  # suppress VTK's 10 ms idle animation timer

    def DestroyTimer(self, obj, evt):
        return 1  # report success so VTK's state machine stays consistent


def _viewer_debug_log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    line = f"[{timestamp}] [Viewer3DNative] {message}"
    print(line)
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(f"{line}\n")
    except OSError:
        pass


class Viewer3DNative(QWidget):
    """Native VTK visualization widget with basic point picking."""

    point_picked = pyqtSignal(object, int)
    points_confirmed = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.geometries = {}
        self.picked_points = []
        self.pick_markers = []
        self.picking_enabled = False
        self.expected_points = 0
        self.picking_mode = None
        self.pick_callback = None
        self.pick_clear_callback = None

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.vtk_widget = _NoAutoPaintVTKWidget(self)
        self.vtk_widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        layout.addWidget(self.vtk_widget)

        self.renderer = vtkRenderer()
        self.renderer.SetBackground(0.07, 0.08, 0.09)
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)

        axes = vtkAxesActor()
        axes.SetTotalLength(30.0, 30.0, 30.0)
        self.renderer.AddActor(axes)

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

        layout.addLayout(control_layout)

        self.pick_status = QLabel("")
        layout.addWidget(self.pick_status)

        self._picker = vtkCellPicker()
        self._picker.SetTolerance(0.002)
        self._left_click_observer = None
        self._right_click_observer = None
        self._initialized = False

    def ensure_plotter_initialized(self) -> bool:
        if self._initialized:
            return True

        self.vtk_widget.Initialize()
        self.vtk_widget.Start()

        interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        if interactor is not None:
            interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
            if self._left_click_observer is None:
                self._left_click_observer = interactor.AddObserver(
                    "LeftButtonPressEvent", self._on_left_click
                )
            if self._right_click_observer is None:
                self._right_click_observer = interactor.AddObserver(
                    "RightButtonPressEvent", self._on_right_click
                )

        self._initialized = True
        _viewer_debug_log(f"{self.objectName() or 'unnamed'} initialized")
        return True

    def _polydata_from_points(self, points_xyz: np.ndarray) -> vtkPolyData:
        points_xyz = np.asarray(points_xyz, dtype=float)
        vtk_points = vtkPoints()
        for row in points_xyz:
            vtk_points.InsertNextPoint(float(row[0]), float(row[1]), float(row[2]))

        polydata = vtkPolyData()
        polydata.SetPoints(vtk_points)

        verts = vtkCellArray()
        for index in range(len(points_xyz)):
            verts.InsertNextCell(1)
            verts.InsertCellPoint(index)
        polydata.SetVerts(verts)
        return polydata

    def _add_polydata_actor(
        self,
        polydata: vtkPolyData,
        name: str,
        color=(0.7, 0.7, 0.7),
        point_size: float = 4.0,
        as_points: bool = True,
        opacity: float = 1.0,
    ) -> vtkActor:
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(polydata)

        actor = vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetColor(float(color[0]), float(color[1]), float(color[2]))
        prop.SetOpacity(float(opacity))
        if as_points:
            prop.SetPointSize(point_size)
            # Keep point clouds bright/consistent regardless of light direction.
            prop.SetAmbient(1.0)
            prop.SetDiffuse(0.0)
            prop.SetSpecular(0.0)
        else:
            # Mesh surfaces should be shaded for depth perception.
            prop.SetAmbient(0.2)
            prop.SetDiffuse(0.8)
            prop.SetSpecular(0.1)
            prop.SetSpecularPower(20.0)
        self.renderer.AddActor(actor)

        self.geometries[name] = {
            "actor": actor,
            "polydata": polydata,
        }
        return actor

    @staticmethod
    def _resolve_color(name: str, color):
        if isinstance(color, (tuple, list)) and len(color) == 3:
            return tuple(float(c) for c in color)

        color_map = {
            "outside_mesh": (0.6, 0.8, 1.0),
            "inside_mesh": (0.5, 0.95, 0.6),
            "mri_scalp": (1.0, 0.8, 0.2),
            "mri_registered": (1.0, 0.9, 0.4),
            "sensors": (0.2, 0.5, 1.0),
        }

        if isinstance(color, str):
            named = {
                "lightblue": (0.6, 0.8, 1.0),
                "lightgreen": (0.5, 0.95, 0.6),
                "gold": (1.0, 0.8, 0.2),
                "lightyellow": (1.0, 0.9, 0.4),
                "blue": (0.2, 0.5, 1.0),
                "red": (1.0, 0.2, 0.2),
                "orange": (1.0, 0.55, 0.2),
                "magenta": (1.0, 0.0, 1.0),
                "purple": (0.7, 0.3, 0.9),
                "cyan": (0.2, 0.9, 0.9),
                "tomato": (1.0, 0.39, 0.28),
            }
            if color in named:
                return named[color]

        return color_map.get(name, (0.7, 0.7, 0.7))

    def add_point_cloud(
        self, o3d_cloud, name="cloud", color=None, point_size=4.0, auto_render=True
    ):
        self.ensure_plotter_initialized()
        self.remove_geometry(name)

        points = np.asarray(o3d_cloud.points)
        if len(points) == 0:
            return

        if len(points) > 100000:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(points), size=50000, replace=False)
            points = points[idx]

        polydata = self._polydata_from_points(points)
        resolved = self._resolve_color(name, color)
        self._add_polydata_actor(
            polydata, name=name, color=resolved, point_size=point_size, as_points=True
        )

        if auto_render:
            self.reset_camera()

    def add_mesh(self, o3d_mesh, name="mesh", color=None, opacity=1.0, auto_render=True):
        self.ensure_plotter_initialized()
        self.remove_geometry(name)

        vertices = np.asarray(o3d_mesh.vertices)
        triangles = np.asarray(o3d_mesh.triangles)
        if len(vertices) == 0 or len(triangles) == 0:
            return

        if not o3d_mesh.has_vertex_normals():
            o3d_mesh.compute_vertex_normals()

        vtk_points = vtkPoints()
        for row in vertices:
            vtk_points.InsertNextPoint(float(row[0]), float(row[1]), float(row[2]))

        polys = vtkCellArray()
        for tri in triangles:
            polys.InsertNextCell(3)
            polys.InsertCellPoint(int(tri[0]))
            polys.InsertCellPoint(int(tri[1]))
            polys.InsertCellPoint(int(tri[2]))

        polydata = vtkPolyData()
        polydata.SetPoints(vtk_points)
        polydata.SetPolys(polys)

        use_vertex_colors = color is None and o3d_mesh.has_vertex_colors()
        if use_vertex_colors:
            rgb = (np.asarray(o3d_mesh.vertex_colors) * 255.0).clip(0, 255).astype(np.uint8)
            vtk_colors = vtkUnsignedCharArray()
            vtk_colors.SetName("RGB")
            vtk_colors.SetNumberOfComponents(3)
            for row in rgb:
                vtk_colors.InsertNextTuple3(int(row[0]), int(row[1]), int(row[2]))
            polydata.GetPointData().SetScalars(vtk_colors)

            mapper = vtkPolyDataMapper()
            mapper.SetInputData(polydata)
            mapper.SetScalarModeToUsePointData()
            mapper.SetColorModeToDirectScalars()
            mapper.ScalarVisibilityOn()

            actor = vtkActor()
            actor.SetMapper(mapper)
            prop = actor.GetProperty()
            prop.SetOpacity(float(opacity))
            prop.SetAmbient(0.2)
            prop.SetDiffuse(0.8)
            prop.SetSpecular(0.1)
            prop.SetSpecularPower(20.0)
            self.renderer.AddActor(actor)
            self.geometries[name] = {"actor": actor, "polydata": polydata}
        else:
            resolved = self._resolve_color(name, color)
            self._add_polydata_actor(
                polydata, name=name, color=resolved, as_points=False, opacity=opacity
            )

        if auto_render:
            self.reset_camera()

    def add_debug_anchor(self, point, name="debug_anchor", radius=6.0, color="tomato"):
        self.ensure_plotter_initialized()
        self.remove_geometry(name)

        center = np.asarray(point, dtype=float).reshape(3)
        sphere = vtkSphereSource()
        sphere.SetCenter(float(center[0]), float(center[1]), float(center[2]))
        sphere.SetRadius(float(radius))
        sphere.SetThetaResolution(20)
        sphere.SetPhiResolution(20)
        sphere.Update()

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())

        actor = vtkActor()
        actor.SetMapper(mapper)
        resolved = self._resolve_color(name, color)
        actor.GetProperty().SetColor(*resolved)
        self.renderer.AddActor(actor)
        self.geometries[name] = {"actor": actor, "polydata": sphere.GetOutput()}

    def add_sphere_markers(self, points, radius=3.0, color="red", name="markers"):
        for i, point in enumerate(points):
            marker_name = f"{name}_{i}"
            self.add_debug_anchor(point, name=marker_name, radius=radius, color=color)
        self.reset_camera()

    def remove_geometry(self, name):
        if name in self.geometries:
            actor = self.geometries[name].get("actor")
            if actor is not None:
                self.renderer.RemoveActor(actor)
            del self.geometries[name]

    def clear_all(self):
        for name in list(self.geometries.keys()):
            self.remove_geometry(name)
        self.clear_picked_points(notify=False)
        self.render()

    def reset_camera(self):
        self.renderer.ResetCamera()
        self.render()

    def render(self):
        self.vtk_widget.GetRenderWindow().Render()

    def debug_state(self, label: str = "state") -> None:
        _viewer_debug_log(
            f"{self.objectName() or 'unnamed'} {label}: visible={self.isVisible()} "
            f"size={self.width()}x{self.height()} geometries={list(self.geometries.keys())}"
        )

    def enable_picking(self, mode, num_points, callback=None, clear_callback=None):
        self.ensure_plotter_initialized()
        self.picking_mode = mode
        self.expected_points = num_points
        self.pick_callback = callback
        self.pick_clear_callback = clear_callback
        self.picking_enabled = True
        self.picked_points = []
        self._clear_pick_markers()
        self._update_pick_status()
        self.clear_picks_btn.setEnabled(True)
        self.confirm_btn.setEnabled(False)

    def disable_picking(self):
        self.picking_enabled = False
        self.pick_status.setText("")

    def _on_left_click(self, obj, event):
        # obj is vtkGenericRenderWindowInteractor (C++ object); VTK's interactor
        # style handles camera rotation through its own observer on this same event,
        # so we only need to handle the picking logic here.
        if not self.picking_enabled:
            return
        if obj is None or obj.GetShiftKey() != 1:
            return

        click_pos = obj.GetEventPosition()
        picked = self._picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
        if picked:
            point = np.array(self._picker.GetPickPosition(), dtype=float)
            self.picked_points.append(point)
            self._add_pick_marker(point)
            self._update_pick_status()
            self.point_picked.emit(point, len(self.picked_points) - 1)

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

        self.picked_points.pop()
        if self.pick_markers:
            actor = self.pick_markers.pop()
            self.renderer.RemoveActor(actor)
        self._update_pick_status()
        self.confirm_btn.setEnabled(len(self.picked_points) >= self.expected_points)
        self.render()

    def _add_pick_marker(self, point: np.ndarray) -> None:
        sphere = vtkSphereSource()
        sphere.SetCenter(float(point[0]), float(point[1]), float(point[2]))
        sphere.SetRadius(3.0)
        sphere.SetThetaResolution(16)
        sphere.SetPhiResolution(16)
        sphere.Update()

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())

        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 1.0, 0.0)
        self.renderer.AddActor(actor)
        self.pick_markers.append(actor)
        self.render()

    def _on_pick_mode_changed(self, index):
        if index == 0:
            self.disable_picking()
        elif index == 1:
            self.enable_picking("anatomical", 3)
        elif index == 2:
            self.enable_picking("helmet", 7)
        elif index == 3:
            self.enable_picking("facial", 3)

    def _on_confirm_points(self):
        if len(self.picked_points) > 0:
            picked_copy = [np.array(p) for p in self.picked_points]
            self.points_confirmed.emit(picked_copy)
            if self.pick_callback:
                self.pick_callback(picked_copy)

    def _update_pick_status(self):
        if self.picking_enabled:
            self.pick_status.setText(
                f"Picking {self.picking_mode} points: "
                f"{len(self.picked_points)}/{self.expected_points}"
            )
        else:
            self.pick_status.setText("")

    def _clear_pick_markers(self):
        for actor in self.pick_markers:
            self.renderer.RemoveActor(actor)
        self.pick_markers = []

    def clear_picked_points(self, notify=True):
        self.picked_points = []
        self._clear_pick_markers()
        self._update_pick_status()
        self.confirm_btn.setEnabled(False)
        if notify and self.pick_clear_callback is not None:
            self.pick_clear_callback()
        self.render()

    def close(self):
        super().close()
