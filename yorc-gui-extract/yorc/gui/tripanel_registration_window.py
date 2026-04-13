"""Three-panel registration GUI for landmark-driven alignment.

Panels:
- Inside LIDAR scan
- Outside LIDAR scan
- MRI scalp surface

Workflow:
1. Load all datasets
2. Pick landmarks in each panel
3. Compute transforms
4. Preview sensors on scalp
5. Apply transforms to FIF
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import mne
import numpy as np
import open3d as o3d
from PyQt6.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from yorc.core.io_utils import (
    load_freesurfer_bem,
    load_mesh,
    load_mesh_for_display,
    load_point_cloud,
)
from yorc.core.bids_integration import export_bids_fiducials
from yorc.core.landmark_detection import detect_landmark_candidates, find_landmarks
from yorc.core.registration import (
    HELMET_STICKER_POSITIONS,
    extract_meg_sensor_contact_and_detector_points,
    extract_meg_sensor_detector_points,
    head_to_helmet,
    head_to_head,
    head_to_mri,
    head_to_standard,
    project_to_rigid_transform,
    write_output,
)
from yorc.gui.viewer_3d import Viewer3D
from yorc.gui.viewer_3d_native import Viewer3DNative

DEBUG_LOG_PATH = Path(tempfile.gettempdir()) / "yorc_tripanel_debug.log"


def _tripanel_debug_log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    line = f"[{timestamp}] [Tripanel] {message}"
    print(line)
    try:
        with DEBUG_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(f"{line}\n")
    except OSError:
        pass


def _surface_vertices_to_mm(vertices: np.ndarray) -> np.ndarray:
    """Convert FreeSurfer surface vertices to mm, handling both m and mm inputs."""
    verts = np.asarray(vertices, dtype=float)
    if verts.size == 0:
        return verts
    # Heuristic: coordinates from MNE BEM/FS can be meters (~0.1) or mm (~100).
    max_abs = float(np.max(np.abs(verts)))
    scale = 1000.0 if max_abs < 1.0 else 1.0
    return verts * scale


def _load_any_cloud(path: str, sample_points: int = 50000) -> o3d.geometry.PointCloud:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".fif":
        cloud, _ = load_freesurfer_bem(path)
        return cloud

    if ext == ".surf":
        verts_m, _ = mne.read_surface(path, return_dict=False, read_metadata=False)
        verts_mm = _surface_vertices_to_mm(verts_m)
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(verts_mm)
        return cloud

    if ext in {".ply", ".pcd"}:
        try:
            cloud = load_point_cloud(path)
            if not cloud.is_empty():
                return cloud
        except Exception:
            pass

    try:
        cloud, _ = load_mesh_for_display(path, max_points=50000)
        if not cloud.is_empty():
            return cloud
    except Exception:
        pass

    cloud, _ = load_mesh(path, n_points=sample_points, fast=True)
    return cloud


def _load_any_geometry(
    path: str, sample_points: int = 50000
) -> tuple[o3d.geometry.PointCloud, Optional[o3d.geometry.TriangleMesh]]:
    """Load input as cloud + optional mesh for richer surface rendering."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".fif":
        cloud, mesh = load_freesurfer_bem(path)
        return cloud, mesh

    if ext == ".surf":
        verts_m, faces = mne.read_surface(path, return_dict=False, read_metadata=False)
        verts_mm = _surface_vertices_to_mm(verts_m)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts_mm)
        mesh.triangles = o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32))
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(verts_mm)
        return cloud, mesh

    if ext in {".ply", ".stl", ".obj"}:
        try:
            mesh = o3d.io.read_triangle_mesh(path)
            if not mesh.is_empty() and len(mesh.triangles) > 0:
                if not mesh.has_vertex_normals():
                    mesh.compute_vertex_normals()
                cloud = o3d.geometry.PointCloud()
                cloud.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
                if mesh.has_vertex_colors():
                    cloud.colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors))
                return cloud, mesh
        except Exception:
            pass

    if ext in {".ply", ".pcd"}:
        try:
            cloud = load_point_cloud(path)
            if not cloud.is_empty():
                return cloud, None
        except Exception:
            pass

    try:
        cloud, _ = load_mesh_for_display(path, max_points=sample_points)
        if not cloud.is_empty():
            return cloud, None
    except Exception:
        pass

    # FreeSurfer surfaces sometimes arrive without .surf extension (e.g., "scalp").
    try:
        verts_m, faces = mne.read_surface(path, return_dict=False, read_metadata=False)
        verts_mm = _surface_vertices_to_mm(verts_m)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts_mm)
        mesh.triangles = o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32))
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(verts_mm)
        return cloud, mesh
    except Exception:
        pass

    cloud, mesh = load_mesh(path, n_points=sample_points, fast=True)
    return cloud, mesh


def _looks_like_inside_helmet_path(path: str) -> bool:
    name = Path(path).name.lower()
    return any(token in name for token in ("helmet", "inside", "_h.", "_h_", "-h.", "-h_"))


def _looks_like_outside_head_path(path: str) -> bool:
    name = Path(path).name.lower()
    return any(token in name for token in ("outside", "head", "scalp", "_r.", "_r_", "-r.", "-r_"))


class _CloudLoadWorker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object, object, object, object, object, object)
    failed = pyqtSignal(str)

    def __init__(self, inside_path: str, outside_path: str, mri_path: str) -> None:
        super().__init__()
        self.inside_path = inside_path
        self.outside_path = outside_path
        self.mri_path = mri_path

    def run(self) -> None:
        try:
            self.progress.emit("Loading inside cloud...")
            inside_cloud, inside_mesh = _load_any_geometry(self.inside_path)
            if inside_cloud.is_empty():
                raise ValueError("Inside cloud is empty")

            self.progress.emit("Loading outside cloud...")
            outside_cloud, outside_mesh = _load_any_geometry(self.outside_path)
            if outside_cloud.is_empty():
                raise ValueError("Outside cloud is empty")

            self.progress.emit("Loading MRI cloud...")
            mri_cloud, mri_mesh = _load_any_geometry(self.mri_path)
            if mri_cloud.is_empty():
                raise ValueError("MRI cloud is empty")

            self.finished.emit(
                inside_cloud, outside_cloud, mri_cloud, inside_mesh, outside_mesh, mri_mesh
            )
        except Exception as exc:
            self.failed.emit(str(exc))


class _SensorPreviewWorker(QObject):
    finished = pyqtSignal(object, object, float, float, object, object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        mri_registered_cloud: o3d.geometry.PointCloud,
        meg_path: str,
        X21: np.ndarray,
    ) -> None:
        super().__init__()
        self.mri_registered_cloud = mri_registered_cloud
        self.meg_path = meg_path
        self.X21 = X21

    def run(self) -> None:
        try:
            from scipy.spatial import cKDTree

            raw = mne.io.read_raw_fif(self.meg_path, verbose="error", preload=False)
            contact_m, detector_m = extract_meg_sensor_contact_and_detector_points(raw)
            if contact_m.size == 0 or detector_m.size == 0:
                raise ValueError("No MEG sensor points found in FIF file.")

            contact_mm = contact_m * 1000.0
            detector_mm = detector_m * 1000.0

            contact_cloud = o3d.geometry.PointCloud()
            contact_cloud.points = o3d.utility.Vector3dVector(contact_mm)
            contact_cloud.transform(self.X21)

            detector_cloud = o3d.geometry.PointCloud()
            detector_cloud.points = o3d.utility.Vector3dVector(detector_mm)
            detector_cloud.transform(self.X21)

            mri_points = np.asarray(self.mri_registered_cloud.points)
            tree = cKDTree(mri_points)
            contact_distances, _ = tree.query(np.asarray(contact_cloud.points), k=1, workers=-1)
            detector_distances, _ = tree.query(np.asarray(detector_cloud.points), k=1, workers=-1)
            contact_median = float(np.median(contact_distances))
            detector_median = float(np.median(detector_distances))

            self.finished.emit(
                contact_cloud,
                detector_cloud,
                contact_median,
                detector_median,
                contact_distances,
                detector_distances,
            )
        except Exception as exc:
            self.failed.emit(str(exc))


class _AutoHelmetFiducialsWorker(QObject):
    progress = pyqtSignal(str)
    candidates_ready = pyqtSignal(object, object)
    finished = pyqtSignal(object, object, object)
    failed = pyqtSignal(str)

    def __init__(self, inside_cloud: o3d.geometry.PointCloud) -> None:
        super().__init__()
        self.inside_cloud = inside_cloud

    @staticmethod
    def _downsample_colored_cloud(
        cloud: o3d.geometry.PointCloud, max_points: int = 200000
    ) -> o3d.geometry.PointCloud:
        points = np.asarray(cloud.points)
        if len(points) <= max_points:
            return cloud

        rng = np.random.default_rng(42)
        idx = rng.choice(len(points), size=max_points, replace=False)
        sampled = o3d.geometry.PointCloud()
        sampled.points = o3d.utility.Vector3dVector(points[idx])
        if cloud.has_colors():
            sampled.colors = o3d.utility.Vector3dVector(np.asarray(cloud.colors)[idx])
        if cloud.has_normals():
            sampled.normals = o3d.utility.Vector3dVector(np.asarray(cloud.normals)[idx])
        return sampled

    def run(self) -> None:
        try:
            cloud = copy.deepcopy(self.inside_cloud)
            if not cloud.has_colors():
                raise ValueError("Inside cloud has no color data; automatic fiducial detection needs red/green helmet markers.")

            original_points = len(np.asarray(cloud.points))
            cloud = self._downsample_colored_cloud(cloud)
            sampled_points = len(np.asarray(cloud.points))
            if sampled_points != original_points:
                self.progress.emit(
                    f"Auto fiducials: downsampled colored inside cloud {original_points:,} -> {sampled_points:,} points"
                )

            self.progress.emit("Auto fiducials: detecting red/green helmet markers...")
            diagnostics = detect_landmark_candidates(cloud)
            self.candidates_ready.emit(
                np.asarray(diagnostics["candidate_centers"]), diagnostics
            )
            landmarks = diagnostics["selected_centers"]
            if landmarks is None:
                raise ValueError(diagnostics["message"])

            X1, detected_landmarks, errors = head_to_helmet(cloud, landmarks=landmarks)
            if X1 is None or detected_landmarks is None:
                raise ValueError("Automatic helmet fiducial registration failed.")

            self.finished.emit(
                np.asarray(detected_landmarks),
                np.asarray(errors, dtype=float),
                diagnostics,
            )
        except Exception as exc:
            self.failed.emit(str(exc))


class _TransformComputeWorker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object, object, object, object, object, object, object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        inside_cloud: o3d.geometry.PointCloud,
        outside_cloud: o3d.geometry.PointCloud,
        mri_cloud: o3d.geometry.PointCloud,
        mri_mesh: Optional[o3d.geometry.TriangleMesh],
        picks: dict,
        fast_mode: bool = True,
        stabilize_icp: bool = False,
        legacy_mode: bool = False,
    ) -> None:
        super().__init__()
        self.inside_cloud = inside_cloud
        self.outside_cloud = outside_cloud
        self.mri_cloud = mri_cloud
        self.mri_mesh = mri_mesh
        self.picks = picks
        self.fast_mode = fast_mode
        self.stabilize_icp = stabilize_icp
        self.legacy_mode = legacy_mode

    @staticmethod
    def _compute_p2p(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
        src = o3d.geometry.PointCloud()
        src.points = o3d.utility.Vector3dVector(np.asarray(source_points))
        dst = o3d.geometry.PointCloud()
        dst.points = o3d.utility.Vector3dVector(np.asarray(target_points))
        n = min(len(source_points), len(target_points))
        corr = np.c_[np.arange(n), np.arange(n)].astype(int)
        p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint(False)
        return p2p.compute_transformation(src, dst, o3d.utility.Vector2iVector(corr))

    @staticmethod
    def _crop_cloud_around_points(
        cloud: o3d.geometry.PointCloud,
        anchor_points: np.ndarray,
        radius_mm: float,
    ) -> o3d.geometry.PointCloud:
        points = np.asarray(cloud.points)
        if len(points) == 0:
            return cloud

        anchor_points = np.asarray(anchor_points, dtype=float)
        if len(anchor_points) == 0:
            return cloud

        center = anchor_points.mean(axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        indices = np.where(distances <= radius_mm)[0]
        if len(indices) < 10:
            return cloud
        return cloud.select_by_index(indices)

    @staticmethod
    def _largest_dbscan_cluster(
        cloud: o3d.geometry.PointCloud, eps_mm: float = 5.0, min_points: int = 10
    ) -> o3d.geometry.PointCloud:
        points = np.asarray(cloud.points)
        if len(points) < min_points:
            return cloud

        labels = np.asarray(cloud.cluster_dbscan(eps=eps_mm, min_points=min_points))
        valid = labels[labels >= 0]
        if len(valid) == 0:
            return cloud

        unique, counts = np.unique(valid, return_counts=True)
        largest_label = unique[np.argmax(counts)]
        keep_idx = np.where(labels == largest_label)[0]
        if len(keep_idx) < min_points:
            return cloud
        return cloud.select_by_index(keep_idx)

    @staticmethod
    def _downsample_cloud(
        cloud: o3d.geometry.PointCloud, max_points: int
    ) -> o3d.geometry.PointCloud:
        points = np.asarray(cloud.points)
        n = len(points)
        if n <= max_points:
            return cloud
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=max_points, replace=False)
        return cloud.select_by_index(idx.tolist())

    @staticmethod
    def _refine_icp(
        source_cloud: o3d.geometry.PointCloud,
        target_cloud: o3d.geometry.PointCloud,
        init: np.ndarray,
        threshold_mm: float = 2.0,
        max_iteration: int = 8000,
    ) -> tuple[np.ndarray, float, float]:
        result = o3d.pipelines.registration.registration_icp(
            source_cloud,
            target_cloud,
            threshold_mm,
            init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration),
        )
        return result.transformation, float(result.fitness), float(result.inlier_rmse)

    @staticmethod
    def _distance_stats(
        source_cloud: o3d.geometry.PointCloud,
        target_cloud: o3d.geometry.PointCloud,
        transform: np.ndarray,
    ) -> dict[str, float]:
        transformed = copy.deepcopy(source_cloud)
        transformed.transform(transform)
        distances = np.asarray(transformed.compute_point_cloud_distance(target_cloud))
        if len(distances) == 0:
            return {"mean": float("nan"), "max": float("nan")}
        return {
            "mean": float(np.mean(distances)),
            "max": float(np.max(distances)),
        }

    @staticmethod
    def _distance_rmse(
        source_cloud: o3d.geometry.PointCloud,
        target_cloud: o3d.geometry.PointCloud,
        transform: np.ndarray,
    ) -> float:
        transformed = copy.deepcopy(source_cloud)
        transformed.transform(transform)
        distances = np.asarray(transformed.compute_point_cloud_distance(target_cloud))
        if len(distances) == 0:
            return float("nan")
        return float(np.sqrt(np.mean(np.square(distances))))

    def _stabilize_icp_transform(
        self,
        source_cloud: o3d.geometry.PointCloud,
        target_cloud: o3d.geometry.PointCloud,
        transform: np.ndarray,
        *,
        threshold_mm: float,
        max_iteration: int,
        max_rounds: int,
        rmse_tol: float,
        label: str,
    ) -> tuple[np.ndarray, float, float]:
        current = transform.copy()
        prev_rmse = self._distance_rmse(source_cloud, target_cloud, current)
        fit, rmse = float("nan"), prev_rmse
        for round_idx in range(1, max_rounds + 1):
            current, fit, rmse = self._refine_icp(
                source_cloud,
                target_cloud,
                current,
                threshold_mm=threshold_mm,
                max_iteration=max_iteration,
            )
            current = project_to_rigid_transform(current)
            delta = (
                abs(prev_rmse - rmse) if (prev_rmse == prev_rmse and rmse == rmse) else float("inf")
            )
            self.progress.emit(
                f"{label} stabilize round {round_idx}: rmse={rmse:.4f} mm (Δ={delta:.4f})"
            )
            if delta < rmse_tol:
                break
            prev_rmse = rmse
        return current, fit, rmse

    def run(self) -> None:
        try:
            self.progress.emit("Computing X1 (device -> inside) from helmet fiducials...")
            X1, _detected_landmarks, helmet_errors = head_to_helmet(
                self.inside_cloud,
                landmarks=self.picks["inside_fiducials"],
            )
            if X1 is None:
                raise ValueError("Could not compute X1 from helmet fiducials.")
            if helmet_errors is not None and len(helmet_errors) > 0:
                self.progress.emit(
                    "X1 landmark fit: "
                    f"mean={float(np.mean(helmet_errors)):.3f} mm, "
                    f"max={float(np.max(helmet_errors)):.3f} mm"
                )

            self.progress.emit("Computing outside->standard transform from anatomy points...")
            Xstd, outside_standard_cloud = head_to_standard(
                self.outside_cloud, self.picks["outside_anat"]
            )

            if self.legacy_mode:
                self.progress.emit(
                    "Legacy mode: using auto/global registration for X2 and anatomy-anchored MRI registration for X3."
                )

                legacy_inside_cloud = self._downsample_cloud(self.inside_cloud, max_points=100000)
                legacy_outside_cloud = self._downsample_cloud(
                    outside_standard_cloud, max_points=100000
                )
                legacy_mri_cloud = self._downsample_cloud(self.mri_cloud, max_points=100000)
                self.progress.emit(
                    "Legacy mode cloud sizes: "
                    f"inside={len(np.asarray(legacy_inside_cloud.points)):,}, "
                    f"outside={len(np.asarray(legacy_outside_cloud.points)):,}, "
                    f"mri={len(np.asarray(legacy_mri_cloud.points)):,}"
                )

                self.progress.emit("Computing X2 (inside -> head/standard) with global registration + ICP...")
                x2_before_stats = {"mean": float("nan"), "max": float("nan")}
                X2, fit2 = head_to_head(
                    legacy_outside_cloud,
                    legacy_inside_cloud,
                    progress_callback=lambda msg, _pct: self.progress.emit(f"X2: {msg}"),
                )
                X2 = project_to_rigid_transform(X2)
                if self.stabilize_icp:
                    X2, fit2, x2_rmse = self._stabilize_icp_transform(
                        legacy_inside_cloud,
                        legacy_outside_cloud,
                        X2,
                        threshold_mm=2.0,
                        max_iteration=2000,
                        max_rounds=4,
                        rmse_tol=0.01,
                        label="X2",
                    )
                else:
                    x2_rmse = self._distance_rmse(legacy_inside_cloud, legacy_outside_cloud, X2)
                x2_after_stats = self._distance_stats(legacy_inside_cloud, legacy_outside_cloud, X2)
                self.progress.emit(f"ICP X2 refine: fitness={fit2:.4f}, rmse={x2_rmse:.4f} mm")

                X21 = project_to_rigid_transform(X2 @ X1)
                self.progress.emit("Computed X21 = X2 @ X1")

                self.progress.emit("Computing X3 (MRI -> head/standard) from matching anatomy points + ICP...")
                outside_anat_std = TriplePanelRegistrationWindow._transform_points(
                    self.picks["outside_anat"], Xstd
                )
                x3_init = self._compute_p2p(self.picks["mri_facial"], outside_anat_std)
                x3_init = project_to_rigid_transform(x3_init)
                x3_before_stats = self._distance_stats(legacy_mri_cloud, legacy_outside_cloud, x3_init)
                X3, fit3, x3_rmse = self._refine_icp(
                    legacy_mri_cloud,
                    legacy_outside_cloud,
                    x3_init,
                    threshold_mm=2.5,
                    max_iteration=2000,
                )
                X3 = project_to_rigid_transform(X3)
                mri_registered_cloud = copy.deepcopy(self.mri_cloud)
                mri_registered_cloud.transform(X3)
                if self.stabilize_icp:
                    X3, fit3, x3_rmse = self._stabilize_icp_transform(
                        legacy_mri_cloud,
                        legacy_outside_cloud,
                        X3,
                        threshold_mm=2.0,
                        max_iteration=2000,
                        max_rounds=4,
                        rmse_tol=0.01,
                        label="X3",
                    )
                    mri_registered_cloud = copy.deepcopy(self.mri_cloud)
                    mri_registered_cloud.transform(X3)
                x3_after_stats = self._distance_stats(legacy_mri_cloud, legacy_outside_cloud, X3)
                self.progress.emit(f"ICP X3 refine: fitness={fit3:.4f}, rmse={x3_rmse:.4f} mm")

                refinement_stats = {
                    "x2": {
                        "source_points": int(len(np.asarray(legacy_inside_cloud.points))),
                        "target_points": int(len(np.asarray(legacy_outside_cloud.points))),
                        "fitness": fit2,
                        "rmse": x2_rmse,
                        "before": x2_before_stats,
                        "after": x2_after_stats,
                    },
                    "x3": {
                        "source_points": int(len(np.asarray(legacy_mri_cloud.points))),
                        "target_points": int(len(np.asarray(legacy_outside_cloud.points))),
                        "fitness": fit3,
                        "rmse": x3_rmse,
                        "before": x3_before_stats,
                        "after": x3_after_stats,
                    },
                }
                self.finished.emit(
                    X1, X2, X3, X21, outside_standard_cloud, mri_registered_cloud, refinement_stats
                )
                return

            outside_facial_std = TriplePanelRegistrationWindow._transform_points(
                self.picks["outside_facial"], Xstd
            )

            self.progress.emit("Computing X2 (inside -> head/standard)...")
            x2_init = self._compute_p2p(self.picks["inside_facial"], outside_facial_std)
            x2_init = project_to_rigid_transform(x2_init)
            if self.fast_mode:
                self.progress.emit("FAST mode: cropped/downsampled ICP for X2")
                inside_crop = self._crop_cloud_around_points(
                    self.inside_cloud, self.picks["inside_facial"], radius_mm=60.0
                )
                inside_crop = self._largest_dbscan_cluster(inside_crop, eps_mm=5.0, min_points=10)
                outside_crop = self._crop_cloud_around_points(
                    outside_standard_cloud, outside_facial_std, radius_mm=60.0
                )
                inside_ref = self._downsample_cloud(inside_crop, max_points=20000)
                outside_ref = self._downsample_cloud(outside_crop, max_points=20000)
                self.progress.emit(
                    "X2 fast sizes: "
                    f"inside={len(np.asarray(inside_ref.points)):,}, "
                    f"outside={len(np.asarray(outside_ref.points)):,}"
                )
                x2_before_stats = self._distance_stats(inside_ref, outside_ref, x2_init)
                X2, fit2, x2_rmse = self._refine_icp(
                    inside_ref, outside_ref, x2_init, threshold_mm=2.5, max_iteration=1400
                )
                X2 = project_to_rigid_transform(X2)
                if self.stabilize_icp:
                    X2, fit2, x2_rmse = self._stabilize_icp_transform(
                        inside_ref,
                        outside_ref,
                        X2,
                        threshold_mm=2.5,
                        max_iteration=1200,
                        max_rounds=3,
                        rmse_tol=0.01,
                        label="X2",
                    )
                x2_after_stats = self._distance_stats(inside_ref, outside_ref, X2)
            else:
                self.progress.emit("FULL mode: global registration + full-cloud ICP for X2")
                x2_before_stats = self._distance_stats(
                    self.inside_cloud, outside_standard_cloud, x2_init
                )
                X2, fit2 = head_to_head(outside_standard_cloud, self.inside_cloud)
                X2 = project_to_rigid_transform(X2)
                if self.stabilize_icp:
                    X2, fit2, x2_rmse = self._stabilize_icp_transform(
                        self.inside_cloud,
                        outside_standard_cloud,
                        X2,
                        threshold_mm=2.0,
                        max_iteration=2000,
                        max_rounds=4,
                        rmse_tol=0.01,
                        label="X2",
                    )
                x2_after_stats = self._distance_stats(self.inside_cloud, outside_standard_cloud, X2)
                if not self.stabilize_icp:
                    x2_rmse = self._distance_rmse(self.inside_cloud, outside_standard_cloud, X2)
            self.progress.emit(f"ICP X2 refine: fitness={fit2:.4f}, rmse={x2_rmse:.4f} mm")

            X21 = project_to_rigid_transform(X2 @ X1)
            self.progress.emit("Computed X21 = X2 @ X1")

            self.progress.emit("Computing X3 (MRI -> head/standard)...")
            x3_init = self._compute_p2p(self.picks["mri_facial"], outside_facial_std)
            x3_init = project_to_rigid_transform(x3_init)
            if self.fast_mode:
                self.progress.emit("FAST mode: cropped/downsampled ICP for X3")
                mri_crop = self._crop_cloud_around_points(
                    self.mri_cloud, self.picks["mri_facial"], radius_mm=60.0
                )
                outside_crop_for_mri = self._crop_cloud_around_points(
                    outside_standard_cloud, outside_facial_std, radius_mm=60.0
                )
                mri_ref = self._downsample_cloud(mri_crop, max_points=20000)
                outside_ref_for_mri = self._downsample_cloud(outside_crop_for_mri, max_points=20000)
                self.progress.emit(
                    "X3 fast sizes: "
                    f"mri={len(np.asarray(mri_ref.points)):,}, "
                    f"outside={len(np.asarray(outside_ref_for_mri.points)):,}"
                )
                x3_before_stats = self._distance_stats(mri_ref, outside_ref_for_mri, x3_init)
                X3, fit3, x3_rmse = self._refine_icp(
                    mri_ref,
                    outside_ref_for_mri,
                    x3_init,
                    threshold_mm=2.5,
                    max_iteration=1400,
                )
                X3 = project_to_rigid_transform(X3)
                if self.stabilize_icp:
                    X3, fit3, x3_rmse = self._stabilize_icp_transform(
                        mri_ref,
                        outside_ref_for_mri,
                        X3,
                        threshold_mm=2.5,
                        max_iteration=1200,
                        max_rounds=3,
                        rmse_tol=0.01,
                        label="X3",
                    )
                x3_after_stats = self._distance_stats(mri_ref, outside_ref_for_mri, X3)
                mri_registered_cloud = copy.deepcopy(self.mri_cloud)
                mri_registered_cloud.transform(X3)
            else:
                self.progress.emit("FULL mode: global registration + full-cloud ICP for X3")
                x3_before_stats = self._distance_stats(
                    self.mri_cloud, outside_standard_cloud, x3_init
                )
                mri_registered_cloud, X3, fit3 = head_to_mri(outside_standard_cloud, self.mri_cloud)
                X3 = project_to_rigid_transform(X3)
                if self.stabilize_icp:
                    X3, fit3, x3_rmse = self._stabilize_icp_transform(
                        self.mri_cloud,
                        outside_standard_cloud,
                        X3,
                        threshold_mm=2.0,
                        max_iteration=2000,
                        max_rounds=4,
                        rmse_tol=0.01,
                        label="X3",
                    )
                    mri_registered_cloud = copy.deepcopy(self.mri_cloud)
                    mri_registered_cloud.transform(X3)
                x3_after_stats = self._distance_stats(self.mri_cloud, outside_standard_cloud, X3)
                if not self.stabilize_icp:
                    x3_rmse = self._distance_rmse(self.mri_cloud, outside_standard_cloud, X3)
            self.progress.emit(f"ICP X3 refine: fitness={fit3:.4f}, rmse={x3_rmse:.4f} mm")
            refinement_stats = {
                "x2": {
                    "source_points": int(
                        len(
                            np.asarray(
                                (inside_ref.points if self.fast_mode else self.inside_cloud.points)
                            )
                        )
                    ),
                    "target_points": int(
                        len(
                            np.asarray(
                                (
                                    outside_ref.points
                                    if self.fast_mode
                                    else outside_standard_cloud.points
                                )
                            )
                        )
                    ),
                    "fitness": fit2,
                    "rmse": x2_rmse,
                    "before": x2_before_stats,
                    "after": x2_after_stats,
                },
                "x3": {
                    "source_points": int(
                        len(
                            np.asarray(
                                (mri_ref.points if self.fast_mode else self.mri_cloud.points)
                            )
                        )
                    ),
                    "target_points": int(
                        len(
                            np.asarray(
                                (
                                    outside_ref_for_mri.points
                                    if self.fast_mode
                                    else outside_standard_cloud.points
                                )
                            )
                        )
                    ),
                    "fitness": fit3,
                    "rmse": x3_rmse,
                    "before": x3_before_stats,
                    "after": x3_after_stats,
                },
            }
            self.finished.emit(
                X1, X2, X3, X21, outside_standard_cloud, mri_registered_cloud, refinement_stats
            )
        except Exception as exc:
            self.failed.emit(str(exc))


class TriplePanelRegistrationWindow(QMainWindow):
    """Main window for three-panel manual registration workflow."""

    _APP_DARK_STYLESHEET = """
    QMainWindow, QWidget {
        background-color: #12161d;
        color: #d9dee8;
    }
    QGroupBox {
        border: 1px solid #2c3542;
        border-radius: 8px;
        margin-top: 10px;
        padding-top: 10px;
        background-color: #171d26;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 6px 0 6px;
        color: #a6d8ff;
        font-weight: 600;
    }
    QLineEdit, QTextEdit {
        background-color: #0e1319;
        border: 1px solid #2d3744;
        border-radius: 6px;
        padding: 4px 6px;
        color: #e7edf7;
        selection-background-color: #2e6da4;
    }
    QPushButton {
        background-color: #243142;
        border: 1px solid #3a4a5f;
        border-radius: 6px;
        padding: 6px 10px;
        color: #eef4ff;
    }
    QPushButton:hover {
        background-color: #2e3f55;
    }
    QPushButton:pressed {
        background-color: #1f2a38;
    }
    QPushButton:disabled {
        background-color: #1a2029;
        color: #7e8998;
        border-color: #2a323d;
    }
    QLabel {
        color: #d9dee8;
    }
    QSplitter::handle {
        background-color: #2b3441;
    }
    """
    _PICK_STEP_DEFAULT_STYLE = (
        "QPushButton { background-color: #2a3341; border: 1px solid #415269; color: #e6edf7; } "
        "QPushButton:disabled { background-color: #1a2029; border: 1px solid #2a323d; color: #7e8998; }"
    )
    _PICK_STEP_DONE_STYLE = (
        "QPushButton { background-color: #2e6b45; border: 1px solid #4f9a6b; color: #f3fff6; } "
        "QPushButton:disabled { background-color: #1f2a24; border: 1px solid #304036; color: #7f9586; }"
    )

    def __init__(self, viewer_backend: str = "pyvista") -> None:
        super().__init__()
        self.setWindowTitle("YORC Tri-Panel Registration (MVP)")
        self.resize(1800, 1000)
        self.viewer_backend = viewer_backend

        self.inside_cloud: Optional[o3d.geometry.PointCloud] = None
        self.outside_cloud: Optional[o3d.geometry.PointCloud] = None
        self.mri_cloud: Optional[o3d.geometry.PointCloud] = None
        self.inside_mesh: Optional[o3d.geometry.TriangleMesh] = None
        self.outside_mesh: Optional[o3d.geometry.TriangleMesh] = None
        self.mri_mesh: Optional[o3d.geometry.TriangleMesh] = None

        self.outside_standard_cloud: Optional[o3d.geometry.PointCloud] = None
        self.mri_registered_cloud: Optional[o3d.geometry.PointCloud] = None
        self.mri_registered_mesh: Optional[o3d.geometry.TriangleMesh] = None
        self.sensor_cloud: Optional[o3d.geometry.PointCloud] = None
        self.sensor_contact_cloud: Optional[o3d.geometry.PointCloud] = None
        self.sensor_detector_cloud: Optional[o3d.geometry.PointCloud] = None

        self.X1: Optional[np.ndarray] = None
        self.X2: Optional[np.ndarray] = None
        self.X3: Optional[np.ndarray] = None
        self.X21: Optional[np.ndarray] = None

        self.picks = {
            "inside_fiducials": None,
            "inside_facial": None,
            "outside_anat": None,
            "outside_facial": None,
            "mri_facial": None,
        }

        self._is_loading = False
        self._is_computing = False
        self._is_previewing = False
        self._is_auto_detecting = False

        self._build_ui()
        self._on_legacy_mode_toggled(True)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        QTimer.singleShot(0, self._ensure_viewers_ready)

    def _ensure_viewers_ready(self) -> None:
        self._log("Initializing 3D renderers...")
        inside_ok = self.inside_view.ensure_plotter_initialized()
        outside_ok = self.outside_view.ensure_plotter_initialized()
        mri_ok = self.mri_view.ensure_plotter_initialized()
        self._log(f"Renderers ready: inside={inside_ok}, outside={outside_ok}, mri={mri_ok}")

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)

        root_layout.addWidget(self._build_files_group())
        root_layout.addWidget(self._build_modes_group())
        root_layout.addWidget(self._build_actions_group())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        viewer_cls = Viewer3D if self.viewer_backend == "pyvista" else Viewer3DNative

        self.inside_view = viewer_cls()
        self.inside_view.setObjectName("inside_view")
        self.outside_view = viewer_cls()
        self.outside_view.setObjectName("outside_view")
        self.mri_view = viewer_cls()
        self.mri_view.setObjectName("mri_view")
        self._bind_panel_reset_buttons()

        splitter.addWidget(self._wrap_panel("Inside LIDAR (helmet)", self.inside_view))
        splitter.addWidget(self._wrap_panel("Outside LIDAR (head)", self.outside_view))
        splitter.addWidget(self._wrap_panel("MRI scalp", self.mri_view))
        splitter.setSizes([600, 600, 600])

        root_layout.addWidget(splitter, 1)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)

        bottom_row = QWidget()
        bottom_layout = QHBoxLayout(bottom_row)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.addWidget(self.log_text, 1)
        bottom_layout.addWidget(self._build_refinement_stats_group(), 1)
        root_layout.addWidget(bottom_row)

        self.setCentralWidget(root)
        self.setStyleSheet(self._APP_DARK_STYLESHEET)
        self._log(f"Viewer backend: {self.viewer_backend}")

    def _bind_panel_reset_buttons(self) -> None:
        bindings = [
            (self.inside_view, "inside"),
            (self.outside_view, "outside"),
            (self.mri_view, "mri"),
        ]
        for view, key in bindings:
            btn = getattr(view, "reset_btn", None)
            if btn is None:
                continue
            try:
                btn.clicked.disconnect()
            except Exception:
                pass
            btn.clicked.connect(lambda _=False, panel=key: self._reset_panel_view(panel))

    def _reset_panel_view(self, panel: str) -> None:
        if panel == "inside":
            view = self.inside_view
            cloud = self.inside_cloud
            mesh = self.inside_mesh
            name = "inside_mesh"
            color = "lightgreen"
        elif panel == "outside":
            view = self.outside_view
            cloud = self.outside_cloud
            mesh = self.outside_mesh
            name = "outside_mesh"
            color = "lightblue"
        elif panel == "mri":
            view = self.mri_view
            cloud = self.mri_cloud
            mesh = self.mri_mesh
            name = "mri_scalp"
            color = "gold"
        else:
            return

        view.clear_all()
        if mesh is not None and not mesh.is_empty():
            view.add_mesh(
                mesh,
                name=name,
                color=None if panel in {"inside", "outside"} else color,
                opacity=1.0,
            )
        elif cloud is not None:
            view.add_point_cloud(cloud, name=name, color=color, point_size=3, auto_render=False)
        view.reset_camera()
        self._log(f"Reset {panel} panel to original loaded geometry.")

    def _build_refinement_stats_group(self) -> QGroupBox:
        group = QGroupBox("Refinement Stats")
        layout = QHBoxLayout(group)

        self.x2_stats_label = QLabel("-")
        self.x2_stats_label.setTextFormat(Qt.TextFormat.PlainText)
        self.x2_stats_label.setWordWrap(True)
        layout.addWidget(self.x2_stats_label, 1)

        self.x3_stats_label = QLabel("-")
        self.x3_stats_label.setTextFormat(Qt.TextFormat.PlainText)
        self.x3_stats_label.setWordWrap(True)
        layout.addWidget(self.x3_stats_label, 1)

        return group

    @staticmethod
    def _fmt_stat(value: float) -> str:
        if value != value:  # NaN guard
            return "n/a"
        return f"{value:.3f}"

    def _reset_refinement_stats(self) -> None:
        self.x2_stats_label.setText("-")
        self.x3_stats_label.setText("-")

    @staticmethod
    def _compact_stats_text(title: str, data: dict) -> str:
        before = data.get("before", {})
        after = data.get("after", {})
        return (
            f"{title}\n"
            f"pts src/tgt: {data.get('source_points', 0):,}/{data.get('target_points', 0):,}\n"
            f"mean mm: {TriplePanelRegistrationWindow._fmt_stat(before.get('mean', float('nan')))}"
            f" -> {TriplePanelRegistrationWindow._fmt_stat(after.get('mean', float('nan')))}\n"
            f"max mm: {TriplePanelRegistrationWindow._fmt_stat(before.get('max', float('nan')))}"
            f" -> {TriplePanelRegistrationWindow._fmt_stat(after.get('max', float('nan')))}\n"
            f"fit/rmse: {TriplePanelRegistrationWindow._fmt_stat(data.get('fitness', float('nan')))}"
            f" / {TriplePanelRegistrationWindow._fmt_stat(data.get('rmse', float('nan')))}"
        )

    def _update_refinement_stats(self, stats: dict) -> None:
        x2 = stats.get("x2", {})
        x3 = stats.get("x3", {})
        self.x2_stats_label.setText(self._compact_stats_text("X2 Inside→Outside", x2))
        self.x3_stats_label.setText(self._compact_stats_text("X3 MRI→Outside", x3))

    def _build_files_group(self) -> QGroupBox:
        group = QGroupBox("Input files")
        layout = QGridLayout(group)

        self.inside_edit, inside_btn = self._path_row("Inside LIDAR (.ply/.stl/.obj)")
        self.outside_edit, outside_btn = self._path_row("Outside LIDAR (.ply/.stl/.obj)")
        self.mri_edit, mri_btn = self._path_row("MRI scalp (.fif/.ply/.stl/.obj)")
        self.meg_edit, meg_btn = self._path_row("MEG file (.fif)")
        self.t1_edit, t1_btn = self._path_row("Subject T1 (.nii/.nii.gz)")
        self.talairach_edit, talairach_btn = self._path_row("FreeSurfer talairach.xfm")

        inside_btn.clicked.connect(
            lambda: self._browse(self.inside_edit, "Mesh/Cloud", "*.ply *.stl *.obj *.pcd")
        )
        outside_btn.clicked.connect(
            lambda: self._browse(self.outside_edit, "Mesh/Cloud", "*.ply *.stl *.obj *.pcd")
        )
        mri_btn.clicked.connect(
            lambda: self._browse(self.mri_edit, "MRI surface", "*.fif *.ply *.stl *.obj")
        )
        meg_btn.clicked.connect(lambda: self._browse(self.meg_edit, "MEG file", "*.fif"))
        t1_btn.clicked.connect(
            lambda: self._browse(self.t1_edit, "Subject T1", "*.nii *.nii.gz")
        )
        talairach_btn.clicked.connect(
            lambda: self._browse(self.talairach_edit, "Talairach transform", "*.xfm")
        )

        layout.addWidget(QLabel("Inside:"), 0, 0)
        layout.addWidget(self.inside_edit, 0, 1)
        layout.addWidget(inside_btn, 0, 2)

        layout.addWidget(QLabel("Outside:"), 1, 0)
        layout.addWidget(self.outside_edit, 1, 1)
        layout.addWidget(outside_btn, 1, 2)

        layout.addWidget(QLabel("MRI:"), 2, 0)
        layout.addWidget(self.mri_edit, 2, 1)
        layout.addWidget(mri_btn, 2, 2)

        layout.addWidget(QLabel("MEG:"), 3, 0)
        layout.addWidget(self.meg_edit, 3, 1)
        layout.addWidget(meg_btn, 3, 2)

        layout.addWidget(QLabel("BIDS T1:"), 4, 0)
        layout.addWidget(self.t1_edit, 4, 1)
        layout.addWidget(t1_btn, 4, 2)

        layout.addWidget(QLabel("Tal XFM:"), 5, 0)
        layout.addWidget(self.talairach_edit, 5, 1)
        layout.addWidget(talairach_btn, 5, 2)

        return group

    def _build_actions_group(self) -> QGroupBox:
        group = QGroupBox("Workflow")
        layout = QHBoxLayout(group)

        self.load_btn = QPushButton("1) Load Data")
        self.load_btn.clicked.connect(self.load_data)
        layout.addWidget(self.load_btn)

        self.pick_inside_btn = QPushButton("2) Pick Inside Fiducials (7)")
        self.pick_inside_btn.clicked.connect(self.pick_inside_fiducials)
        layout.addWidget(self.pick_inside_btn)

        self.auto_inside_btn = QPushButton("Auto Helmet Fids")
        self.auto_inside_btn.clicked.connect(self.auto_detect_inside_fiducials)
        layout.addWidget(self.auto_inside_btn)

        self.pick_outside_anat_btn = QPushButton("3) Pick Outside Anatomy (3)")
        self.pick_outside_anat_btn.clicked.connect(self.pick_outside_anatomy)
        layout.addWidget(self.pick_outside_anat_btn)

        self.pick_inside_face_btn = QPushButton("4) Pick Inside Face (3)")
        self.pick_inside_face_btn.clicked.connect(self.pick_inside_facial)
        layout.addWidget(self.pick_inside_face_btn)

        self.pick_outside_face_btn = QPushButton("5) Pick Outside Face (3)")
        self.pick_outside_face_btn.clicked.connect(self.pick_outside_facial)
        layout.addWidget(self.pick_outside_face_btn)

        self.pick_mri_face_btn = QPushButton("6) Pick MRI Face (3)")
        self.pick_mri_face_btn.clicked.connect(self.pick_mri_facial)
        layout.addWidget(self.pick_mri_face_btn)

        self.compute_btn = QPushButton("7) Compute Transforms")
        self.compute_btn.clicked.connect(self.compute_transforms)
        layout.addWidget(self.compute_btn)

        self.preview_btn = QPushButton("8) Preview Sensors")
        self.preview_btn.clicked.connect(self.preview_sensors)
        layout.addWidget(self.preview_btn)

        self.sensor_display_mode = QComboBox()
        self.sensor_display_mode.addItems(
            [
                "Display contact points (pad at scalp)",
                "Display detector centers (+6 mm)",
            ]
        )
        self.sensor_display_mode.setCurrentIndex(0)
        self.sensor_display_mode.currentIndexChanged.connect(self._on_sensor_display_mode_changed)
        layout.addWidget(self.sensor_display_mode)

        self.apply_btn = QPushButton("9) Apply to FIF")
        self.apply_btn.clicked.connect(self.apply_to_fif)
        layout.addWidget(self.apply_btn)

        self.export_bids_btn = QPushButton("Export BIDS Fids")
        self.export_bids_btn.clicked.connect(self.export_bids_metadata)
        layout.addWidget(self.export_bids_btn)

        self.save_picks_btn = QPushButton("Save Picks")
        self.save_picks_btn.clicked.connect(self.save_picks)
        layout.addWidget(self.save_picks_btn)

        self.load_picks_btn = QPushButton("Load Picks")
        self.load_picks_btn.clicked.connect(self.load_picks)
        layout.addWidget(self.load_picks_btn)

        self.show_x2_btn = QPushButton("Show X2 Overlay")
        self.show_x2_btn.clicked.connect(self.show_x2_overlay)
        layout.addWidget(self.show_x2_btn)

        self.show_x3_btn = QPushButton("Show X3 Overlay")
        self.show_x3_btn.clicked.connect(self.show_x3_overlay)
        layout.addWidget(self.show_x3_btn)

        self.restore_views_btn = QPushButton("Restore Views")
        self.restore_views_btn.clicked.connect(self.restore_views)
        layout.addWidget(self.restore_views_btn)

        self._pick_step_buttons = {
            "inside_fiducials": self.pick_inside_btn,
            "outside_anat": self.pick_outside_anat_btn,
            "inside_facial": self.pick_inside_face_btn,
            "outside_facial": self.pick_outside_face_btn,
            "mri_facial": self.pick_mri_face_btn,
        }
        self._refresh_pick_step_styles()

        return group

    def _build_modes_group(self) -> QGroupBox:
        group = QGroupBox("Modes")
        layout = QHBoxLayout(group)

        self.fast_mode_check = QCheckBox("Fast Mode")
        self.fast_mode_check.setChecked(True)
        self.fast_mode_check.setToolTip("Use cropped/downsampled ICP for faster transforms.")
        layout.addWidget(self.fast_mode_check)

        self.stabilize_icp_check = QCheckBox("Stabilize ICP")
        self.stabilize_icp_check.setChecked(False)
        self.stabilize_icp_check.setToolTip(
            "Run extra ICP rounds until RMSE change is small (slower, may improve fit)."
        )
        layout.addWidget(self.stabilize_icp_check)

        self.legacy_mode_check = QCheckBox("Legacy Mode")
        self.legacy_mode_check.setChecked(True)
        self.legacy_mode_check.setToolTip(
            "Use inside helmet fiducials plus matching outside and MRI anatomy picks (RPA, LPA, Nasion) for a more robust legacy-style workflow."
        )
        self.legacy_mode_check.toggled.connect(self._on_legacy_mode_toggled)
        layout.addWidget(self.legacy_mode_check)

        layout.addStretch(1)
        return group

    def _on_legacy_mode_toggled(self, enabled: bool) -> None:
        inside_face_button = getattr(self, "pick_inside_face_btn", None)
        outside_face_button = getattr(self, "pick_outside_face_btn", None)
        mri_face_button = getattr(self, "pick_mri_face_btn", None)

        if inside_face_button is not None:
            inside_face_button.setEnabled(not enabled)
        if outside_face_button is not None:
            outside_face_button.setEnabled(not enabled)
        if mri_face_button is not None:
            mri_face_button.setEnabled(True)
            mri_face_button.setText("6) Pick MRI Anatomy (3)" if enabled else "6) Pick MRI Face (3)")

        if enabled:
            self._log(
                "Legacy mode enabled: pick outside and MRI points as the same anatomy landmarks: RPA, LPA, Nasion."
            )
        else:
            self._log(
                "Legacy mode disabled: tri-panel facial picks are required for X2/X3 initialization."
            )

    def _path_row(self, placeholder: str) -> tuple[QLineEdit, QPushButton]:
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        btn = QPushButton("Browse")
        return edit, btn

    def _wrap_panel(self, title: str, widget: QWidget) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(QLabel(f"<b>{title}</b>"))
        layout.addWidget(widget)
        return container

    def _browse(self, target: QLineEdit, title: str, filt: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, title, os.path.expanduser("~"), filt)
        if path:
            target.setText(path)

    def _log(self, message: str) -> None:
        self.log_text.append(message)
        self.statusBar().showMessage(message)
        _tripanel_debug_log(message)

    def _dump_viewer_states(self, label: str) -> None:
        _tripanel_debug_log(f"viewer-state-dump: {label}")
        self.inside_view.debug_state(label)
        self.outside_view.debug_state(label)
        self.mri_view.debug_state(label)

    def _set_loading_state(self, loading: bool) -> None:
        self._is_loading = loading
        self.load_btn.setEnabled(not loading)
        if loading:
            self.setCursor(Qt.CursorShape.WaitCursor)
            self.statusBar().showMessage("Loading data...")
        else:
            self.unsetCursor()

    def _set_preview_state(self, previewing: bool) -> None:
        self._is_previewing = previewing
        self.preview_btn.setEnabled(not previewing)
        self.sensor_display_mode.setEnabled(not previewing)
        if previewing:
            self.statusBar().showMessage("Previewing sensors...")

    def _set_auto_detect_state(self, auto_detecting: bool) -> None:
        self._is_auto_detecting = auto_detecting
        self.auto_inside_btn.setEnabled(not auto_detecting)
        if auto_detecting:
            self.setCursor(Qt.CursorShape.WaitCursor)
            self.statusBar().showMessage("Automatically detecting helmet fiducials...")
        else:
            self.unsetCursor()

    def _set_compute_state(self, computing: bool) -> None:
        self._is_computing = computing
        self.compute_btn.setEnabled(not computing)
        self.fast_mode_check.setEnabled(not computing)
        self.stabilize_icp_check.setEnabled(not computing)
        if computing:
            self.statusBar().showMessage("Computing transforms...")
        else:
            self.statusBar().clearMessage()

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Error", message)
        self._log(f"❌ {message}")

    @staticmethod
    def _load_any_cloud(path: str, sample_points: int = 50000) -> o3d.geometry.PointCloud:
        return _load_any_cloud(path, sample_points=sample_points)

    @staticmethod
    def _points_to_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(np.asarray(points))
        return cloud

    @staticmethod
    def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        points = np.asarray(points)
        pts_h = np.c_[points, np.ones(len(points))]
        out = (transform @ pts_h.T).T
        return out[:, :3]

    @staticmethod
    def _extract_sensor_points_device_mm(meg_path: str) -> np.ndarray:
        raw = mne.io.read_raw_fif(meg_path, verbose="error", preload=False)
        points_m = extract_meg_sensor_detector_points(raw)
        if points_m.size == 0:
            raise ValueError("No MEG sensor points found in FIF file.")
        return points_m * 1000.0

    @staticmethod
    def _rigid_metrics(transform: np.ndarray) -> tuple[float, float]:
        R = np.asarray(transform, dtype=float)[:3, :3]
        det = float(np.linalg.det(R))
        ortho_err = float(np.linalg.norm(R.T @ R - np.eye(3), ord="fro"))
        return det, ortho_err

    def _apply_inside_sensor_penalty_mainthread(self) -> None:
        if (
            self.mri_registered_mesh is None
            or self.mri_registered_mesh.is_empty()
            or self.X3 is None
            or self.X21 is None
        ):
            self._log("Sensor-inside penalty refine: skipped (missing transformed scalp mesh)")
            return

        meg_path = self.meg_edit.text().strip()
        if not meg_path:
            self._log("Sensor-inside penalty refine: skipped (no MEG file selected)")
            return

        try:
            sensor_device_mm = self._extract_sensor_points_device_mm(meg_path)
        except Exception as exc:
            self._log(f"Sensor-inside penalty refine: skipped (MEG read failed: {exc})")
            return

        def _scene_from_mesh(mesh: o3d.geometry.TriangleMesh):
            scene = o3d.t.geometry.RaycastingScene()
            _ = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
            return scene

        def _inside_mask(scene, points_mm: np.ndarray) -> np.ndarray:
            occ = scene.compute_occupancy(
                o3d.core.Tensor(points_mm.astype(np.float32), dtype=o3d.core.Dtype.Float32)
            ).numpy()
            return occ > 0.5

        sensors_head = self._transform_points(sensor_device_mm, self.X21)
        mesh = self.mri_registered_mesh
        scene = _scene_from_mesh(mesh)
        before_mask = _inside_mask(scene, sensors_head)
        before_frac = float(np.mean(before_mask)) if len(before_mask) > 0 else 0.0

        total_shift = np.zeros(3, dtype=float)
        for _ in range(6):
            inside_mask = _inside_mask(scene, sensors_head)
            if not np.any(inside_mask):
                break
            inside_points = sensors_head[inside_mask].astype(np.float32)
            closest = scene.compute_closest_points(
                o3d.core.Tensor(inside_points, dtype=o3d.core.Dtype.Float32)
            )["points"].numpy()
            delta = closest - inside_points
            mean_shift = delta.mean(axis=0)
            if np.linalg.norm(mean_shift) < 1e-3:
                break

            # Rigid-only penalty step: translation only (no scale/shear/extra rotation).
            # Use opposite direction so inside sensors move toward/outside the scalp surface.
            step = -0.8 * mean_shift
            T = np.eye(4)
            T[:3, 3] = step
            self.X3 = project_to_rigid_transform(T @ self.X3)
            mesh.transform(T)
            self.mri_registered_cloud.transform(T)
            total_shift += step
            scene = _scene_from_mesh(mesh)

        after_mask = _inside_mask(scene, sensors_head)
        after_frac = float(np.mean(after_mask)) if len(after_mask) > 0 else 0.0
        self._log(
            "Sensor-inside penalty refine (main-thread rigid translation): "
            f"inside fraction {before_frac:.3f} -> {after_frac:.3f}, "
            f"total shift={np.linalg.norm(total_shift):.3f} mm"
        )

    def load_data(self) -> None:
        if self._is_loading:
            self._log("Load already in progress...")
            return

        _tripanel_debug_log(f"debug log path: {DEBUG_LOG_PATH}")
        inside_path = self.inside_edit.text().strip()
        outside_path = self.outside_edit.text().strip()
        mri_path = self.mri_edit.text().strip()

        if not inside_path or not outside_path or not mri_path:
            self._show_error("Please select inside, outside, and MRI files.")
            return

        if _looks_like_outside_head_path(inside_path) and _looks_like_inside_helmet_path(
            outside_path
        ):
            self._log(
                "Detected swapped LIDAR inputs from filenames; swapping inside/outside assignments."
            )
            inside_path, outside_path = outside_path, inside_path
            self.inside_edit.setText(inside_path)
            self.outside_edit.setText(outside_path)

        _tripanel_debug_log(
            f"input paths inside={inside_path} outside={outside_path} mri={mri_path}"
        )
        self._reset_refinement_stats()
        self._set_loading_state(True)

        self._load_thread = QThread(self)
        self._load_worker = _CloudLoadWorker(inside_path, outside_path, mri_path)
        self._load_worker.moveToThread(self._load_thread)

        self._load_thread.started.connect(self._load_worker.run)
        self._load_worker.progress.connect(self._log)
        self._load_worker.finished.connect(self._on_clouds_loaded)
        self._load_worker.failed.connect(self._on_clouds_failed)
        self._load_worker.finished.connect(self._load_thread.quit)
        self._load_worker.failed.connect(self._load_thread.quit)
        self._load_thread.finished.connect(self._load_worker.deleteLater)
        self._load_thread.finished.connect(self._load_thread.deleteLater)

        self._load_thread.start()

    def _on_clouds_loaded(
        self,
        inside_cloud: o3d.geometry.PointCloud,
        outside_cloud: o3d.geometry.PointCloud,
        mri_cloud: o3d.geometry.PointCloud,
        inside_mesh: Optional[o3d.geometry.TriangleMesh],
        outside_mesh: Optional[o3d.geometry.TriangleMesh],
        mri_mesh: Optional[o3d.geometry.TriangleMesh],
    ) -> None:
        try:
            self.inside_cloud = inside_cloud
            self.outside_cloud = outside_cloud
            self.mri_cloud = mri_cloud
            self.inside_mesh = inside_mesh
            self.outside_mesh = outside_mesh
            self.mri_mesh = mri_mesh

            inside_n = len(np.asarray(self.inside_cloud.points))
            outside_n = len(np.asarray(self.outside_cloud.points))
            mri_n = len(np.asarray(self.mri_cloud.points))
            self._log(f"Inside points: {inside_n:,}")
            self._log(f"Outside points: {outside_n:,}")
            self._log(f"MRI points: {mri_n:,}")
            if self.inside_mesh is not None and not self.inside_mesh.is_empty():
                self._log(f"Inside mesh triangles: {len(np.asarray(self.inside_mesh.triangles)):,}")
                self.inside_view.add_mesh(
                    self.inside_mesh, name="inside_mesh", color=None, opacity=1.0
                )
            else:
                self.inside_view.add_point_cloud(
                    self.inside_cloud,
                    name="inside_mesh",
                    color="lightgreen",
                    point_size=3,
                    auto_render=False,
                )

            if self.outside_mesh is not None and not self.outside_mesh.is_empty():
                self._log(
                    f"Outside mesh triangles: {len(np.asarray(self.outside_mesh.triangles)):,}"
                )
                self.outside_view.add_mesh(
                    self.outside_mesh, name="outside_mesh", color=None, opacity=1.0
                )
            else:
                self.outside_view.add_point_cloud(
                    self.outside_cloud,
                    name="outside_mesh",
                    color="lightblue",
                    point_size=3,
                    auto_render=False,
                )

            if self.mri_mesh is not None and not self.mri_mesh.is_empty():
                self._log(f"MRI mesh triangles: {len(np.asarray(self.mri_mesh.triangles)):,}")
                self.mri_view.add_mesh(self.mri_mesh, name="mri_scalp", color="gold", opacity=1.0)
            else:
                self.mri_view.add_point_cloud(
                    self.mri_cloud,
                    name="mri_scalp",
                    color="gold",
                    point_size=3,
                    auto_render=False,
                )

            self.inside_view.add_debug_anchor(
                np.asarray(self.inside_cloud.points).mean(axis=0),
                name="inside_anchor",
                color="tomato",
            )
            self.outside_view.add_debug_anchor(
                np.asarray(self.outside_cloud.points).mean(axis=0),
                name="outside_anchor",
                color="orange",
            )
            self.mri_view.add_debug_anchor(
                np.asarray(self.mri_cloud.points).mean(axis=0),
                name="mri_anchor",
                color="magenta",
            )

            self._log("Running final camera reset/render on all panels...")
            self.inside_view.reset_camera()
            self.outside_view.reset_camera()
            self.mri_view.reset_camera()
            self._dump_viewer_states("post-sync-reset")
            QTimer.singleShot(700, lambda: self._dump_viewer_states("t+700ms"))
            QTimer.singleShot(1500, lambda: self._dump_viewer_states("t+1500ms"))

            self._log("✅ Data loaded")
            _tripanel_debug_log("Data load complete")
            self.statusBar().showMessage("Load complete: inside/outside/MRI rendered.", 5000)
        except Exception as exc:
            _tripanel_debug_log(f"rendering exception after load: {exc}")
            self._show_error(f"Failed to render loaded data: {exc}")
        finally:
            self._set_loading_state(False)

    def _on_clouds_failed(self, error_message: str) -> None:
        _tripanel_debug_log(f"load_data exception: {error_message}")
        self._set_loading_state(False)
        self._show_error(f"Failed to load data: {error_message}")

    def _store_picks(self, key: str, points: list[np.ndarray], color: str, view: QWidget) -> None:
        arr = np.asarray(points)
        self.picks[key] = arr
        self._clear_pick_markers_for_key(key, view)
        view.add_sphere_markers(arr, radius=2.5, color=color, name=f"{key}_markers")
        self._set_pick_step_done(key, done=len(arr) > 0)
        self._log(f"✅ Stored {len(arr)} points for {key}")

    def _set_pick_step_done(self, key: str, done: bool) -> None:
        btn = getattr(self, "_pick_step_buttons", {}).get(key)
        if btn is None:
            return
        btn.setStyleSheet(self._PICK_STEP_DONE_STYLE if done else self._PICK_STEP_DEFAULT_STYLE)

    def _refresh_pick_step_styles(self) -> None:
        for key in getattr(self, "_pick_step_buttons", {}):
            points = self.picks.get(key)
            done = points is not None and len(points) > 0
            self._set_pick_step_done(key, done=done)

    def _on_picks_cleared(self, key: str, view: QWidget) -> None:
        self.picks[key] = None
        self._clear_pick_markers_for_key(key, view)
        self._set_pick_step_done(key, done=False)
        self._log(f"Cleared points for {key}")

    @staticmethod
    def _pick_meta():
        return {
            "inside_fiducials": ("red", "inside"),
            "inside_facial": ("purple", "inside"),
            "outside_anat": ("orange", "outside"),
            "outside_facial": ("magenta", "outside"),
            "mri_facial": ("cyan", "mri"),
        }

    def _get_view_by_name(self, name: str):
        return {
            "inside": self.inside_view,
            "outside": self.outside_view,
            "mri": self.mri_view,
        }[name]

    def _clear_pick_markers_for_key(self, key: str, view: QWidget) -> None:
        geometries = getattr(view, "geometries", {})
        if not isinstance(geometries, dict):
            return
        prefix = f"{key}_markers_"
        for geom_name in [name for name in geometries if name.startswith(prefix)]:
            view.remove_geometry(geom_name)

    def _clear_auto_candidate_markers(self) -> None:
        geometries = getattr(self.inside_view, "geometries", {})
        if not isinstance(geometries, dict):
            return
        prefix = "auto_inside_candidates_"
        for geom_name in [name for name in geometries if name.startswith(prefix)]:
            self.inside_view.remove_geometry(geom_name)

    def _show_auto_candidate_markers(self, points: np.ndarray) -> None:
        self._clear_auto_candidate_markers()
        if points is None or len(points) == 0:
            return
        self.inside_view.add_sphere_markers(
            np.asarray(points), radius=2.0, color="yellow", name="auto_inside_candidates"
        )

    @staticmethod
    def _format_auto_detection_summary(diagnostics: dict) -> str:
        cluster_counts = diagnostics.get("candidate_counts")
        if isinstance(cluster_counts, np.ndarray):
            cluster_counts = cluster_counts.tolist()
        counts_text = ", ".join(str(int(value)) for value in (cluster_counts or []))
        if not counts_text:
            counts_text = "none"

        summary = (
            "Auto fiducials: "
            f"overlap={int(diagnostics.get('n_unique_overlap_points', 0))} points, "
            f"clusters={int(diagnostics.get('n_clusters', 0))}, "
            f"sizes=[{counts_text}]"
        )
        if diagnostics.get("used_largest_subset"):
            selected_counts = diagnostics.get("selected_counts")
            if isinstance(selected_counts, np.ndarray):
                selected_counts = selected_counts.tolist()
            kept_text = ", ".join(str(int(value)) for value in (selected_counts or []))
            summary += f"; keeping largest clusters [{kept_text}]"
        return summary

    def save_picks(self) -> None:
        serializable = {
            key: value.tolist() if value is not None else None for key, value in self.picks.items()
        }
        payload = {
            "inside_path": self.inside_edit.text().strip(),
            "outside_path": self.outside_edit.text().strip(),
            "mri_path": self.mri_edit.text().strip(),
            "picks": serializable,
        }

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Picks",
            os.path.expanduser("~/yorc_picks.json"),
            "JSON Files (*.json)",
        )
        if not path:
            return

        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            self._log(f"✅ Saved picks to {path}")
        except Exception as exc:
            self._show_error(f"Failed to save picks: {exc}")

    def load_picks(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Picks",
            os.path.expanduser("~"),
            "JSON Files (*.json)",
        )
        if not path:
            return

        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)

            loaded = payload.get("picks", {})
            applied = []
            for key, (color, view_name) in self._pick_meta().items():
                points = loaded.get(key)
                self.picks[key] = None if points is None else np.asarray(points, dtype=float)
                if self.picks[key] is not None and len(self.picks[key]) > 0:
                    view = self._get_view_by_name(view_name)
                    self._clear_pick_markers_for_key(key, view)
                    view.add_sphere_markers(
                        self.picks[key], radius=2.5, color=color, name=f"{key}_markers"
                    )
                    applied.append(f"{key}={len(self.picks[key])}")

            self._refresh_pick_step_styles()
            self._log(f"✅ Loaded picks from {path} ({', '.join(applied) if applied else 'none'})")
        except Exception as exc:
            self._show_error(f"Failed to load picks: {exc}")

    def pick_inside_fiducials(self) -> None:
        if self.inside_cloud is None:
            self._show_error("Load data first.")
            return
        self._log("Pick 7 helmet fiducials in the inside panel.")
        self.inside_view.enable_picking(
            mode="inside_fiducials",
            num_points=7,
            callback=lambda pts: self._store_picks(
                "inside_fiducials", pts, "red", self.inside_view
            ),
            clear_callback=lambda: self._on_picks_cleared("inside_fiducials", self.inside_view),
        )

    def auto_detect_inside_fiducials(self) -> None:
        if self.inside_cloud is None:
            self._show_error("Load data first.")
            return
        if self._is_auto_detecting:
            self._log("Automatic fiducial detection already in progress...")
            return

        self._set_auto_detect_state(True)
        self._log("Starting automatic helmet fiducial detection...")

        self._auto_inside_thread = QThread(self)
        self._auto_inside_worker = _AutoHelmetFiducialsWorker(copy.deepcopy(self.inside_cloud))
        self._auto_inside_worker.moveToThread(self._auto_inside_thread)

        self._auto_inside_thread.started.connect(self._auto_inside_worker.run)
        self._auto_inside_worker.progress.connect(self._log)
        self._auto_inside_worker.candidates_ready.connect(self._on_auto_inside_candidates_ready)
        self._auto_inside_worker.finished.connect(self._on_auto_inside_fiducials_finished)
        self._auto_inside_worker.failed.connect(self._on_auto_inside_fiducials_failed)
        self._auto_inside_worker.finished.connect(self._auto_inside_thread.quit)
        self._auto_inside_worker.failed.connect(self._auto_inside_thread.quit)
        self._auto_inside_thread.finished.connect(self._auto_inside_worker.deleteLater)
        self._auto_inside_thread.finished.connect(self._auto_inside_thread.deleteLater)

        self._auto_inside_thread.start()

    def _on_auto_inside_candidates_ready(self, candidate_centers: np.ndarray, diagnostics: dict) -> None:
        self._show_auto_candidate_markers(np.asarray(candidate_centers))
        self._log(self._format_auto_detection_summary(diagnostics))

    def _on_auto_inside_fiducials_finished(
        self, landmarks: np.ndarray, errors: np.ndarray, diagnostics: dict
    ) -> None:
        try:
            self._store_picks("inside_fiducials", list(np.asarray(landmarks)), "red", self.inside_view)
            self._log(
                "✅ Auto-detected helmet fiducials: "
                f"n={len(landmarks)}, mean error={float(np.mean(errors)):.3f} mm, "
                f"max error={float(np.max(errors)):.3f} mm"
            )
            if diagnostics.get("used_largest_subset"):
                self._log(diagnostics["message"])
        finally:
            self._set_auto_detect_state(False)

    def _on_auto_inside_fiducials_failed(self, message: str) -> None:
        self._set_auto_detect_state(False)
        self._show_error(f"Automatic fiducial detection failed: {message}")

    def pick_outside_anatomy(self) -> None:
        if self.outside_cloud is None:
            self._show_error("Load data first.")
            return
        self._log("Pick 3 anatomy points in outside panel: RPA, LPA, Nasion.")
        self.outside_view.enable_picking(
            mode="outside_anat",
            num_points=3,
            callback=lambda pts: self._store_picks(
                "outside_anat", pts, "orange", self.outside_view
            ),
            clear_callback=lambda: self._on_picks_cleared("outside_anat", self.outside_view),
        )

    def pick_outside_facial(self) -> None:
        if self.outside_cloud is None:
            self._show_error("Load data first.")
            return
        self._log("Pick 3 facial points in outside panel: Right eye, Left eye, Nose.")
        self.outside_view.enable_picking(
            mode="outside_facial",
            num_points=3,
            callback=lambda pts: self._store_picks(
                "outside_facial", pts, "magenta", self.outside_view
            ),
            clear_callback=lambda: self._on_picks_cleared("outside_facial", self.outside_view),
        )

    def pick_inside_facial(self) -> None:
        if self.inside_cloud is None:
            self._show_error("Load data first.")
            return
        self._log("Pick 3 facial points in inside panel: Right eye, Left eye, Nose.")
        self.inside_view.enable_picking(
            mode="inside_facial",
            num_points=3,
            callback=lambda pts: self._store_picks(
                "inside_facial", pts, "purple", self.inside_view
            ),
            clear_callback=lambda: self._on_picks_cleared("inside_facial", self.inside_view),
        )

    def pick_mri_facial(self) -> None:
        if self.mri_cloud is None:
            self._show_error("Load data first.")
            return
        if self.legacy_mode_check.isChecked():
            self._log("Pick corresponding 3 anatomy points in MRI panel: RPA, LPA, Nasion.")
        else:
            self._log("Pick corresponding 3 facial points in MRI panel: Right eye, Left eye, Nose.")
        self.mri_view.enable_picking(
            mode="mri_facial",
            num_points=3,
            callback=lambda pts: self._store_picks("mri_facial", pts, "cyan", self.mri_view),
            clear_callback=lambda: self._on_picks_cleared("mri_facial", self.mri_view),
        )

    def _compute_p2p(self, source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
        src = self._points_to_cloud(source_points)
        dst = self._points_to_cloud(target_points)

        n = min(len(source_points), len(target_points))
        corr = np.c_[np.arange(n), np.arange(n)].astype(int)
        p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        return p2p.compute_transformation(src, dst, o3d.utility.Vector2iVector(corr))

    def _refine_icp(
        self,
        source_cloud: o3d.geometry.PointCloud,
        target_cloud: o3d.geometry.PointCloud,
        init: np.ndarray,
        threshold_mm: float = 2.0,
    ) -> np.ndarray:
        result = o3d.pipelines.registration.registration_icp(
            source_cloud,
            target_cloud,
            threshold_mm,
            init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=8000),
        )
        self._log(f"ICP refine: fitness={result.fitness:.4f}, rmse={result.inlier_rmse:.4f} mm")
        return result.transformation

    def compute_transforms(self) -> None:
        if self._is_computing:
            self._log("Transform computation already in progress...")
            return

        legacy_mode = bool(self.legacy_mode_check.isChecked())
        required = ["inside_fiducials", "outside_anat"]
        if legacy_mode:
            required.append("mri_facial")
        else:
            required.extend(["inside_facial", "outside_facial", "mri_facial"])
        missing = [k for k in required if self.picks.get(k) is None]
        if missing:
            self._show_error(f"Missing picks: {', '.join(missing)}")
            return

        if self.inside_cloud is None or self.outside_cloud is None or self.mri_cloud is None:
            self._show_error("Load all datasets first.")
            return

        self._reset_refinement_stats()
        self._set_compute_state(True)
        fast_mode = bool(self.fast_mode_check.isChecked())
        stabilize_icp = bool(self.stabilize_icp_check.isChecked())
        self._log(
            "Starting transform computation "
            f"({'FAST' if fast_mode else 'FULL'} mode, "
            f"{'stabilized' if stabilize_icp else 'single-pass'}, "
            f"{'legacy' if legacy_mode else 'tri-panel'})..."
        )

        worker_picks = {k: np.asarray(v).copy() for k, v in self.picks.items() if v is not None}

        self._compute_thread = QThread(self)
        self._compute_worker = _TransformComputeWorker(
            copy.deepcopy(self.inside_cloud),
            copy.deepcopy(self.outside_cloud),
            copy.deepcopy(self.mri_cloud),
            copy.deepcopy(self.mri_mesh) if self.mri_mesh is not None else None,
            worker_picks,
            fast_mode=fast_mode,
            stabilize_icp=stabilize_icp,
            legacy_mode=legacy_mode,
        )
        self._compute_worker.moveToThread(self._compute_thread)

        self._compute_thread.started.connect(self._compute_worker.run)
        self._compute_worker.progress.connect(self._log)
        self._compute_worker.finished.connect(self._on_compute_finished)
        self._compute_worker.failed.connect(self._on_compute_failed)
        self._compute_worker.finished.connect(self._compute_thread.quit)
        self._compute_worker.failed.connect(self._compute_thread.quit)
        self._compute_thread.finished.connect(self._compute_worker.deleteLater)
        self._compute_thread.finished.connect(self._compute_thread.deleteLater)

        self._compute_thread.start()

    def _on_compute_finished(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        X3: np.ndarray,
        X21: np.ndarray,
        outside_standard_cloud: o3d.geometry.PointCloud,
        mri_registered_cloud: o3d.geometry.PointCloud,
        refinement_stats: dict,
    ) -> None:
        try:
            self.X1 = X1
            self.X2 = X2
            self.X3 = X3
            self.X21 = X21
            # Hard guard: all transforms must remain rigid (rotation + translation only).
            self.X1 = project_to_rigid_transform(self.X1)
            self.X2 = project_to_rigid_transform(self.X2)
            self.X3 = project_to_rigid_transform(self.X3)
            self.X21 = project_to_rigid_transform(self.X21)
            self.outside_standard_cloud = outside_standard_cloud
            self.mri_registered_cloud = mri_registered_cloud
            self.mri_registered_mesh = None
            if self.mri_mesh is not None and not self.mri_mesh.is_empty():
                self.mri_registered_mesh = copy.deepcopy(self.mri_mesh)
                self.mri_registered_mesh.transform(self.X3)

            # Clear old MRI panel content before showing registered result.
            self.mri_view.clear_all()
            if self.mri_registered_mesh is not None and not self.mri_registered_mesh.is_empty():
                self.mri_view.add_mesh(
                    self.mri_registered_mesh,
                    name="mri_registered",
                    color="lightyellow",
                    opacity=0.35,
                    auto_render=False,
                )
            else:
                self.mri_view.add_point_cloud(
                    self.mri_registered_cloud,
                    name="mri_registered",
                    color="lightyellow",
                    point_size=2,
                    auto_render=False,
                )
            if hasattr(self.mri_view, "render"):
                self.mri_view.render()
            self._update_refinement_stats(refinement_stats)
            for name, T in (("X1", self.X1), ("X2", self.X2), ("X3", self.X3), ("X21", self.X21)):
                det, ortho_err = self._rigid_metrics(T)
                self._log(f"{name} rigid check: det(R)={det:.6f}, ortho_err={ortho_err:.2e}")
            self._log("✅ Transforms computed")
        except Exception as exc:
            self._show_error(f"Failed to render computed transforms: {exc}")
        finally:
            self._set_compute_state(False)

    def _on_compute_failed(self, message: str) -> None:
        self._set_compute_state(False)
        self._show_error(f"Failed to compute transforms: {message}")

    def show_x2_overlay(self) -> None:
        if self.X2 is None or self.inside_cloud is None or self.outside_standard_cloud is None:
            self._show_error("Compute transforms first.")
            return

        inside_x2 = copy.deepcopy(self.inside_cloud)
        inside_x2.transform(self.X2)

        self.outside_view.clear_all()
        self.outside_view.add_point_cloud(
            self.outside_standard_cloud,
            name="x2_target",
            color="lightblue",
            point_size=2,
            auto_render=False,
        )
        self.outside_view.add_point_cloud(
            inside_x2,
            name="x2_source",
            color="red",
            point_size=2,
            auto_render=False,
        )
        self.outside_view.reset_camera()
        self._log("Displayed X2 overlay (inside transformed vs outside-standard).")

    def show_x3_overlay(self) -> None:
        if self.X3 is None or self.mri_cloud is None or self.outside_standard_cloud is None:
            self._show_error("Compute transforms first.")
            return

        self.mri_view.clear_all()
        self.mri_view.add_point_cloud(
            self.outside_standard_cloud,
            name="x3_target",
            color="lightblue",
            point_size=2,
            auto_render=False,
        )
        if self.mri_registered_mesh is not None and not self.mri_registered_mesh.is_empty():
            self.mri_view.add_mesh(
                self.mri_registered_mesh,
                name="x3_source_mesh",
                color="lightyellow",
                opacity=0.35,
                auto_render=False,
            )
        else:
            mri_x3 = copy.deepcopy(self.mri_cloud)
            mri_x3.transform(self.X3)
            self.mri_view.add_point_cloud(
                mri_x3,
                name="x3_source",
                color="gold",
                point_size=2,
                auto_render=False,
            )
        self.mri_view.reset_camera()
        self._log("Displayed X3 overlay (MRI transformed vs outside-standard).")

    def restore_views(self) -> None:
        if self.inside_cloud is None or self.outside_cloud is None or self.mri_cloud is None:
            self._show_error("Load data first.")
            return

        self.inside_view.clear_all()
        if self.inside_mesh is not None and not self.inside_mesh.is_empty():
            self.inside_view.add_mesh(self.inside_mesh, name="inside_mesh", color=None, opacity=1.0)
        else:
            self.inside_view.add_point_cloud(
                self.inside_cloud,
                name="inside_mesh",
                color="lightgreen",
                point_size=3,
                auto_render=False,
            )

        self.outside_view.clear_all()
        if self.outside_mesh is not None and not self.outside_mesh.is_empty():
            self.outside_view.add_mesh(
                self.outside_mesh, name="outside_mesh", color=None, opacity=1.0
            )
        else:
            self.outside_view.add_point_cloud(
                self.outside_cloud,
                name="outside_mesh",
                color="lightblue",
                point_size=3,
                auto_render=False,
            )

        self.mri_view.clear_all()
        if self.mri_registered_mesh is not None and not self.mri_registered_mesh.is_empty():
            self.mri_view.add_mesh(
                self.mri_registered_mesh,
                name="mri_registered",
                color="lightyellow",
                opacity=0.35,
                auto_render=False,
            )
        elif self.mri_mesh is not None and not self.mri_mesh.is_empty():
            self.mri_view.add_mesh(self.mri_mesh, name="mri_scalp", color="gold", opacity=1.0)
        else:
            self.mri_view.add_point_cloud(
                self.mri_cloud,
                name="mri_scalp",
                color="gold",
                point_size=3,
                auto_render=False,
            )

        if self.sensor_contact_cloud is not None and self.sensor_detector_cloud is not None:
            display_cloud = self._active_sensor_display_cloud()
            inside_cloud, outside_cloud, n_inside, n_outside = self._split_sensor_cloud_by_scalp(
                display_cloud=display_cloud,
                detector_cloud=self.sensor_detector_cloud,
            )
            if inside_cloud is not None and n_inside > 0:
                self.mri_view.add_point_cloud(
                    inside_cloud,
                    name="sensors_inside",
                    color="red",
                    point_size=8,
                    auto_render=False,
                )
            if outside_cloud is not None and n_outside > 0:
                self.mri_view.add_point_cloud(
                    outside_cloud,
                    name="sensors_outside",
                    color="blue",
                    point_size=8,
                    auto_render=False,
                )

        self.inside_view.reset_camera()
        self.outside_view.reset_camera()
        self.mri_view.reset_camera()
        self._log("Restored base panel views.")

    def preview_sensors(self) -> None:
        if self._is_previewing:
            self._log("Sensor preview already running...")
            return

        meg_path = self.meg_edit.text().strip()
        if not meg_path:
            self._show_error("Select a MEG .fif file.")
            return
        if self.X21 is None or self.mri_registered_cloud is None:
            self._show_error("Compute transforms first.")
            return

        self._set_preview_state(True)
        self._log("Computing sensor preview...")

        self._preview_thread = QThread(self)
        self._preview_worker = _SensorPreviewWorker(
            copy.deepcopy(self.mri_registered_cloud), meg_path, self.X21.copy()
        )
        self._preview_worker.moveToThread(self._preview_thread)

        self._preview_thread.started.connect(self._preview_worker.run)
        self._preview_worker.finished.connect(self._on_preview_finished)
        self._preview_worker.failed.connect(self._on_preview_failed)
        self._preview_worker.finished.connect(self._preview_thread.quit)
        self._preview_worker.failed.connect(self._preview_thread.quit)
        self._preview_thread.finished.connect(self._preview_worker.deleteLater)
        self._preview_thread.finished.connect(self._preview_thread.deleteLater)

        self._preview_thread.start()

    def _on_preview_finished(
        self,
        sensor_contact_cloud: o3d.geometry.PointCloud,
        sensor_detector_cloud: o3d.geometry.PointCloud,
        contact_median_dist: float,
        detector_median_dist: float,
        _contact_distances,
        _detector_distances,
    ) -> None:
        try:
            self.sensor_contact_cloud = sensor_contact_cloud
            self.sensor_detector_cloud = sensor_detector_cloud
            self.sensor_cloud = self._active_sensor_display_cloud()
            # Rebuild MRI panel to avoid stacking old/unregistered geometry.
            self.mri_view.clear_all()
            if self.mri_registered_mesh is not None and not self.mri_registered_mesh.is_empty():
                self.mri_view.add_mesh(
                    self.mri_registered_mesh,
                    name="mri_registered",
                    color="lightyellow",
                    opacity=0.35,
                    auto_render=False,
                )
            elif self.mri_registered_cloud is not None:
                self.mri_view.add_point_cloud(
                    self.mri_registered_cloud,
                    name="mri_registered",
                    color="lightyellow",
                    point_size=2,
                    auto_render=False,
                )

            inside_cloud, outside_cloud, n_inside, n_outside = self._split_sensor_cloud_by_scalp(
                display_cloud=self.sensor_cloud,
                detector_cloud=self.sensor_detector_cloud,
            )
            if inside_cloud is not None and n_inside > 0:
                self.mri_view.add_point_cloud(
                    inside_cloud,
                    name="sensors_inside",
                    color="red",
                    point_size=8,
                    auto_render=False,
                )
            if outside_cloud is not None and n_outside > 0:
                self.mri_view.add_point_cloud(
                    outside_cloud,
                    name="sensors_outside",
                    color="blue",
                    point_size=8,
                    auto_render=False,
                )
            if inside_cloud is None and outside_cloud is None:
                # Fallback if mesh-based split is unavailable.
                self.mri_view.add_point_cloud(
                    self.sensor_cloud, name="sensors", color="blue", point_size=8, auto_render=False
                )
                self._log("Sensor split unavailable; rendered all sensors in blue.")

            # Render without resetting camera to avoid expensive scene-wide camera fit.
            if hasattr(self.mri_view, "render"):
                self.mri_view.render()
            self._log(
                "✅ Sensor preview ready. "
                f"Median contact->scalp: {contact_median_dist:.3f} mm, "
                f"detector->scalp: {detector_median_dist:.3f} mm"
            )
            if inside_cloud is not None and outside_cloud is not None:
                self._log(
                    f"Sensor location split: inside={n_inside} (red), outside={n_outside} (blue)"
                )
        except Exception as exc:
            self._show_error(f"Failed to render sensor preview: {exc}")
        finally:
            self._set_preview_state(False)

    def _on_preview_failed(self, message: str) -> None:
        self._set_preview_state(False)
        self._show_error(f"Failed to preview sensors: {message}")

    def _on_sensor_display_mode_changed(self, _index: int) -> None:
        if self.sensor_contact_cloud is None or self.sensor_detector_cloud is None:
            return
        self.sensor_cloud = self._active_sensor_display_cloud()
        self.mri_view.clear_all()
        if self.mri_registered_mesh is not None and not self.mri_registered_mesh.is_empty():
            self.mri_view.add_mesh(
                self.mri_registered_mesh,
                name="mri_registered",
                color="lightyellow",
                opacity=0.35,
                auto_render=False,
            )
        elif self.mri_registered_cloud is not None:
            self.mri_view.add_point_cloud(
                self.mri_registered_cloud,
                name="mri_registered",
                color="lightyellow",
                point_size=2,
                auto_render=False,
            )
        inside_cloud, outside_cloud, n_inside, n_outside = self._split_sensor_cloud_by_scalp(
            display_cloud=self.sensor_cloud,
            detector_cloud=self.sensor_detector_cloud,
        )
        if inside_cloud is not None and n_inside > 0:
            self.mri_view.add_point_cloud(
                inside_cloud,
                name="sensors_inside",
                color="red",
                point_size=8,
                auto_render=False,
            )
        if outside_cloud is not None and n_outside > 0:
            self.mri_view.add_point_cloud(
                outside_cloud,
                name="sensors_outside",
                color="blue",
                point_size=8,
                auto_render=False,
            )
        if hasattr(self.mri_view, "render"):
            self.mri_view.render()
        self._log(
            "Sensor display mode changed to "
            f"{'contact' if self.sensor_display_mode.currentIndex() == 0 else 'detector'} "
            "(inside/outside colors use detector positions)."
        )

    def _active_sensor_display_cloud(self) -> Optional[o3d.geometry.PointCloud]:
        if self.sensor_contact_cloud is None or self.sensor_detector_cloud is None:
            return None
        return (
            self.sensor_contact_cloud
            if self.sensor_display_mode.currentIndex() == 0
            else self.sensor_detector_cloud
        )

    def _split_sensor_cloud_by_scalp(
        self,
        display_cloud: Optional[o3d.geometry.PointCloud] = None,
        detector_cloud: Optional[o3d.geometry.PointCloud] = None,
    ) -> tuple[
        Optional[o3d.geometry.PointCloud],
        Optional[o3d.geometry.PointCloud],
        int,
        int,
    ]:
        if (
            self.sensor_contact_cloud is None
            or self.sensor_detector_cloud is None
            or self.mri_registered_mesh is None
            or self.mri_registered_mesh.is_empty()
        ):
            return None, None, 0, 0

        if display_cloud is None:
            display_cloud = self._active_sensor_display_cloud()
        if detector_cloud is None:
            detector_cloud = self.sensor_detector_cloud
        if display_cloud is None or detector_cloud is None:
            return None, None, 0, 0

        display_points = np.asarray(display_cloud.points)
        detector_points = np.asarray(detector_cloud.points)
        if len(display_points) == 0 or len(detector_points) == 0:
            return None, None, 0, 0
        if len(display_points) != len(detector_points):
            self._log(
                "Sensor split warning: display/detector point count mismatch; "
                "falling back to detector points for display."
            )
            display_points = detector_points

        try:
            mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(self.mri_registered_mesh)
            scene = o3d.t.geometry.RaycastingScene()
            _ = scene.add_triangles(mesh_t)
            signed = scene.compute_signed_distance(
                o3d.core.Tensor(detector_points, dtype=o3d.core.Dtype.Float32)
            ).numpy()
            inside_mask = signed < 0.0
        except Exception as exc:
            self._log(f"Sensor split warning: could not classify inside/outside ({exc})")
            return None, None, 0, 0

        inside_points = display_points[inside_mask]
        outside_points = display_points[~inside_mask]

        inside_cloud = o3d.geometry.PointCloud()
        outside_cloud = o3d.geometry.PointCloud()
        if len(inside_points) > 0:
            inside_cloud.points = o3d.utility.Vector3dVector(inside_points)
        if len(outside_points) > 0:
            outside_cloud.points = o3d.utility.Vector3dVector(outside_points)

        return inside_cloud, outside_cloud, len(inside_points), len(outside_points)

    def apply_to_fif(self) -> None:
        meg_path = self.meg_edit.text().strip()
        if not meg_path:
            self._show_error("Select a MEG .fif file.")
            return
        if self.X21 is None or self.X3 is None:
            self._show_error("Compute transforms first.")
            return

        try:
            out_path = write_output(meg_path, self.X21, self.X3)
            self._log(f"✅ Wrote transforms: {out_path}")
            QMessageBox.information(self, "Done", f"Transforms written to:\n{out_path}")
        except Exception as exc:
            self._show_error(f"Failed to apply transforms: {exc}")

    def export_bids_metadata(self) -> None:
        meg_path = self.meg_edit.text().strip()
        t1_path = self.t1_edit.text().strip()
        talairach_path = self.talairach_edit.text().strip()

        if not meg_path:
            self._show_error("Select a MEG .fif file.")
            return
        if not t1_path or not talairach_path:
            self._show_error("Select both BIDS T1 and Talairach transform paths.")
            return
        if self.X21 is None or self.X3 is None:
            self._show_error("Compute transforms first.")
            return

        try:
            meg_out, json_out = export_bids_fiducials(
                meg_data_path=meg_path,
                talairach_xfm_path=talairach_path,
                t1_path=t1_path,
                dev_head_transform=self.X21,
                mri_to_head_transform=self.X3,
            )
            self._log(f"✅ Wrote BIDS-compatible MEG fiducials: {meg_out}")
            self._log(f"✅ Wrote BIDS MRI landmarks JSON: {json_out}")
            QMessageBox.information(
                self,
                "BIDS Fiducials Exported",
                f"Updated MEG file:\n{meg_out}\n\nUpdated MRI JSON:\n{json_out}",
            )
        except Exception as exc:
            self._show_error(f"Failed to export BIDS fiducials: {exc}")
