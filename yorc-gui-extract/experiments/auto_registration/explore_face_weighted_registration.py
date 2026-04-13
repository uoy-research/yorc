#!/usr/bin/env python3

"""Explore outside-LIDAR to MRI registration with anterior or face emphasis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from yorc.core.io_utils import load_freesurfer_bem, load_mesh, load_point_cloud
from yorc.core.registration import execute_global_registration, preprocess_point_cloud, project_to_rigid_transform


def _load_any_surface(path: Path, n_points: int) -> o3d.geometry.PointCloud:
    suffix = path.suffix.lower()
    if suffix == ".fif":
        point_cloud, _mesh = load_freesurfer_bem(str(path))
        return point_cloud

    if suffix in {".ply", ".pcd"}:
        try:
            return load_point_cloud(str(path))
        except ValueError:
            point_cloud, _mesh = load_mesh(str(path), n_points=n_points, fast=True)
            return point_cloud

    if suffix in {".stl", ".obj"}:
        point_cloud, _mesh = load_mesh(str(path), n_points=n_points, fast=True)
        return point_cloud

    raise ValueError(f"Unsupported input format: {path}")


def _downsample_to_limit(point_cloud: o3d.geometry.PointCloud, max_points: int) -> o3d.geometry.PointCloud:
    points = np.asarray(point_cloud.points)
    if len(points) <= max_points:
        return point_cloud

    rng = np.random.default_rng(42)
    indices = np.sort(rng.choice(len(points), size=max_points, replace=False))

    downsampled = o3d.geometry.PointCloud()
    downsampled.points = o3d.utility.Vector3dVector(points[indices])
    if point_cloud.has_colors():
        downsampled.colors = o3d.utility.Vector3dVector(np.asarray(point_cloud.colors)[indices])
    if point_cloud.has_normals():
        downsampled.normals = o3d.utility.Vector3dVector(np.asarray(point_cloud.normals)[indices])
    return downsampled


def _make_rigid_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def _principal_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    cov = centered.T @ centered / max(len(points), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    basis = eigenvectors[:, order]
    if np.linalg.det(basis) < 0:
        basis[:, -1] *= -1.0
    return basis, centroid


def _principal_axis_initializations(source_cloud: o3d.geometry.PointCloud, target_cloud: o3d.geometry.PointCloud) -> list[tuple[str, np.ndarray]]:
    source_basis, source_centroid = _principal_axes(np.asarray(source_cloud.points))
    target_basis, target_centroid = _principal_axes(np.asarray(target_cloud.points))

    sign_options = [
        ("pca_identity", np.diag([1.0, 1.0, 1.0])),
        ("pca_flip_xy", np.diag([-1.0, -1.0, 1.0])),
        ("pca_flip_xz", np.diag([-1.0, 1.0, -1.0])),
        ("pca_flip_yz", np.diag([1.0, -1.0, -1.0])),
    ]

    initializations = []
    for label, sign_matrix in sign_options:
        rotation = target_basis @ sign_matrix @ source_basis.T
        if np.linalg.det(rotation) < 0:
            continue
        translation = target_centroid - rotation @ source_centroid
        initializations.append((label, project_to_rigid_transform(_make_rigid_transform(rotation, translation))))
    return initializations


def _canonical_mri_to_head_initializations(source_cloud: o3d.geometry.PointCloud, target_cloud: o3d.geometry.PointCloud) -> list[tuple[str, np.ndarray]]:
    source_points = np.asarray(source_cloud.points)
    target_points = np.asarray(target_cloud.points)
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)

    ras_to_head = np.array(
        [
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    variants = [
        ("ras_to_head", ras_to_head),
        ("ras_to_head_rot_x_180", np.diag([1.0, -1.0, -1.0]) @ ras_to_head),
        ("ras_to_head_rot_y_180", np.diag([-1.0, 1.0, -1.0]) @ ras_to_head),
        ("ras_to_head_rot_z_180", np.diag([-1.0, -1.0, 1.0]) @ ras_to_head),
    ]

    initializations = []
    for label, rotation in variants:
        if np.linalg.det(rotation) < 0:
            continue
        translation = target_centroid - rotation @ source_centroid
        initializations.append((label, project_to_rigid_transform(_make_rigid_transform(rotation, translation))))
    return initializations


def _deduplicate_initializations(initializations: list[tuple[str, np.ndarray]]) -> list[tuple[str, np.ndarray]]:
    deduped = []
    seen = set()
    for label, transform in initializations:
        key = tuple(np.round(transform, decimals=6).ravel())
        if key in seen:
            continue
        seen.add(key)
        deduped.append((label, transform))
    return deduped


def _quantile_mask(points: np.ndarray, axis_index: int, keep_fraction: float) -> np.ndarray:
    threshold = np.quantile(points[:, axis_index], 1.0 - keep_fraction)
    return points[:, axis_index] >= threshold


def _subset_cloud(point_cloud: o3d.geometry.PointCloud, mask: np.ndarray) -> o3d.geometry.PointCloud:
    indices = np.flatnonzero(mask).tolist()
    return point_cloud.select_by_index(indices)


def _duplicate_subset(point_cloud: o3d.geometry.PointCloud, subset_mask: np.ndarray, weight: int) -> o3d.geometry.PointCloud:
    points = np.asarray(point_cloud.points)
    subset_points = points[subset_mask]
    if len(subset_points) == 0 or weight <= 1:
        return point_cloud

    stacked_points = [points, *([subset_points] * (weight - 1))]
    weighted = o3d.geometry.PointCloud()
    weighted.points = o3d.utility.Vector3dVector(np.vstack(stacked_points))
    return weighted


def _rmse_to_target(source_cloud: o3d.geometry.PointCloud, target_cloud: o3d.geometry.PointCloud, transform: np.ndarray) -> float:
    moved = o3d.geometry.PointCloud(source_cloud)
    moved.transform(transform)
    distances = np.asarray(moved.compute_point_cloud_distance(target_cloud), dtype=float)
    if len(distances) == 0:
        return float("inf")
    return float(np.sqrt(np.mean(np.square(distances))))


def _fitness_and_rmse(
    source_cloud: o3d.geometry.PointCloud,
    target_cloud: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
    threshold: float,
    max_iteration: int,
) -> tuple[np.ndarray, float, float]:
    result = o3d.pipelines.registration.registration_icp(
        source_cloud,
        target_cloud,
        threshold,
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration),
    )
    rigid = project_to_rigid_transform(result.transformation)
    return rigid, float(result.fitness), float(result.inlier_rmse)


def _global_initialization(source_cloud: o3d.geometry.PointCloud, target_cloud: o3d.geometry.PointCloud, voxel_size: float) -> np.ndarray:
    source_down, source_fpfh = preprocess_point_cloud(source_cloud, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target_cloud, voxel_size)
    result = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    return project_to_rigid_transform(result.transformation)


def evaluate_methods(
    outside_cloud: o3d.geometry.PointCloud,
    mri_cloud: o3d.geometry.PointCloud,
    voxel_size: float,
    threshold: float,
    max_iteration: int,
    axis_index: int,
    face_fraction: float,
    face_weight: int,
) -> dict:
    base_init = _global_initialization(mri_cloud, outside_cloud, voxel_size)
    init_candidates = _deduplicate_initializations(
        [("global", base_init)]
        + _principal_axis_initializations(mri_cloud, outside_cloud)
        + _canonical_mri_to_head_initializations(mri_cloud, outside_cloud)
    )

    outside_points = np.asarray(outside_cloud.points)
    mri_points = np.asarray(mri_cloud.points)
    outside_face_mask = _quantile_mask(outside_points, axis_index, face_fraction)
    mri_face_mask = _quantile_mask(mri_points, axis_index, face_fraction)

    outside_face = _subset_cloud(outside_cloud, outside_face_mask)
    mri_face = _subset_cloud(mri_cloud, mri_face_mask)
    outside_weighted = _duplicate_subset(outside_cloud, outside_face_mask, face_weight)
    mri_weighted = _duplicate_subset(mri_cloud, mri_face_mask, face_weight)

    methods = {
        "full": (mri_cloud, outside_cloud),
        "anterior_crop": (mri_face, outside_face),
        "face_weighted": (mri_weighted, outside_weighted),
    }

    report = {
        "parameters": {
            "voxel_size": voxel_size,
            "threshold": threshold,
            "max_iteration": max_iteration,
            "axis_index": axis_index,
            "face_fraction": face_fraction,
            "face_weight": face_weight,
        },
        "candidates": [label for label, _transform in init_candidates],
        "methods": {},
    }

    for method_name, (source_variant, target_variant) in methods.items():
        best = None
        best_score = None
        for init_label, init_transform in init_candidates:
            rigid, fitness, icp_rmse = _fitness_and_rmse(
                source_variant,
                target_variant,
                init_transform,
                threshold,
                max_iteration,
            )
            full_rmse = _rmse_to_target(mri_cloud, outside_cloud, rigid)
            face_rmse = _rmse_to_target(mri_face, outside_face, rigid)
            score = (face_rmse, full_rmse, -fitness, icp_rmse)
            candidate = {
                "init": init_label,
                "transform": rigid.tolist(),
                "fitness": fitness,
                "icp_rmse": icp_rmse,
                "full_rmse": full_rmse,
                "face_rmse": face_rmse,
            }
            if best is None or score < best_score:
                best = candidate
                best_score = score
        report["methods"][method_name] = best

    report["selected_method"] = min(
        report["methods"],
        key=lambda key: (
            report["methods"][key]["face_rmse"],
            report["methods"][key]["full_rmse"],
            -report["methods"][key]["fitness"],
        ),
    )
    return report


def _save_transformed_clouds(
    output_dir: Path,
    report: dict,
    mri_cloud: o3d.geometry.PointCloud,
    outside_cloud: o3d.geometry.PointCloud,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_dir / "outside_cloud.ply"), outside_cloud)
    for method_name, method_report in report["methods"].items():
        transformed = o3d.geometry.PointCloud(mri_cloud)
        transformed.transform(np.asarray(method_report["transform"], dtype=float))
        o3d.io.write_point_cloud(str(output_dir / f"mri_{method_name}.ply"), transformed)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2))


def _visualize_selected(report: dict, mri_cloud: o3d.geometry.PointCloud, outside_cloud: o3d.geometry.PointCloud) -> None:
    method = report["selected_method"]
    transform = np.asarray(report["methods"][method]["transform"], dtype=float)
    transformed = o3d.geometry.PointCloud(mri_cloud)
    transformed.transform(transform)
    transformed.paint_uniform_color([1.0, 0.85, 0.1])
    target = o3d.geometry.PointCloud(outside_cloud)
    target.paint_uniform_color([0.6, 0.7, 1.0])
    o3d.visualization.draw_geometries([transformed, target], width=1200, height=900)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outside", required=True, help="Outside LIDAR surface (.ply/.pcd/.stl/.obj)")
    parser.add_argument("--mri", required=True, help="MRI scalp surface (.stl/.obj/.ply/.fif)")
    parser.add_argument("--input-points", type=int, default=100000, help="Points sampled from mesh inputs")
    parser.add_argument("--max-points", type=int, default=100000, help="Maximum points used per cloud")
    parser.add_argument("--voxel-size", type=float, default=2.0, help="Voxel size for global registration")
    parser.add_argument("--threshold", type=float, default=2.5, help="ICP inlier threshold in mm")
    parser.add_argument("--max-iteration", type=int, default=3000, help="Maximum ICP iterations")
    parser.add_argument(
        "--crop-axis",
        choices=["x", "y", "z"],
        default="x",
        help="Axis treated as anterior for face emphasis (default: x)",
    )
    parser.add_argument(
        "--face-fraction",
        type=float,
        default=0.25,
        help="Fraction of the most anterior points treated as face region",
    )
    parser.add_argument(
        "--face-weight",
        type=int,
        default=4,
        help="Approximate face weight by duplicating anterior points this many times",
    )
    parser.add_argument("--write-clouds", help="Optional output directory for transformed clouds and JSON report")
    parser.add_argument("--visualize", action="store_true", help="Visualize the selected registration result")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    axis_index = {"x": 0, "y": 1, "z": 2}[args.crop_axis]

    outside_cloud = _load_any_surface(Path(args.outside), args.input_points)
    mri_cloud = _load_any_surface(Path(args.mri), args.input_points)
    outside_cloud = _downsample_to_limit(outside_cloud, args.max_points)
    mri_cloud = _downsample_to_limit(mri_cloud, args.max_points)

    report = evaluate_methods(
        outside_cloud=outside_cloud,
        mri_cloud=mri_cloud,
        voxel_size=args.voxel_size,
        threshold=args.threshold,
        max_iteration=args.max_iteration,
        axis_index=axis_index,
        face_fraction=args.face_fraction,
        face_weight=args.face_weight,
    )

    print(json.dumps(report, indent=2))

    if args.write_clouds:
        _save_transformed_clouds(Path(args.write_clouds), report, mri_cloud, outside_cloud)

    if args.visualize:
        _visualize_selected(report, mri_cloud, outside_cloud)


if __name__ == "__main__":
    main()