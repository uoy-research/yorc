#!/usr/bin/env python3

"""Explore outside-LIDAR to MRI registration with anterior or face emphasis."""

from __future__ import annotations

import argparse
import json
import math
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


def _compose_canonical_initial_transform(
    rotation: np.ndarray,
    source_centroid: np.ndarray,
    target_centroid: np.ndarray,
    compose_order: str,
) -> np.ndarray:
    if compose_order == "centroid_lock":
        translation = target_centroid - rotation @ source_centroid
        return project_to_rigid_transform(_make_rigid_transform(rotation, translation))
    if compose_order == "translate_then_rotate":
        centroid_translation = target_centroid - source_centroid
        translation_first = _make_rigid_transform(np.eye(3, dtype=float), centroid_translation)
        rotation_second = _make_rigid_transform(rotation, np.zeros(3, dtype=float))
        return project_to_rigid_transform(rotation_second @ translation_first)
    raise ValueError(f"Unsupported canonical init compose order: {compose_order}")


def _quarter_turn_rotation(axis: str, quarter_turns: int) -> np.ndarray:
    turns = quarter_turns % 4
    identity = np.eye(3, dtype=float)
    if axis == "x":
        variants = [
            identity,
            np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=float),
            np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=float),
            np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=float),
        ]
        return variants[turns]
    if axis == "y":
        variants = [
            identity,
            np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=float),
            np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]], dtype=float),
            np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
        ]
        return variants[turns]
    if axis == "z":
        variants = [
            identity,
            np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float),
            np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float),
            np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float),
        ]
        return variants[turns]
    raise ValueError(f"Unsupported rotation axis: {axis}")


def _right_angle_orientation_variants() -> list[tuple[str, np.ndarray]]:
    operations = [
        ("rot_x_90", _quarter_turn_rotation("x", 1)),
        ("rot_x_-90", _quarter_turn_rotation("x", -1)),
        ("rot_y_90", _quarter_turn_rotation("y", 1)),
        ("rot_y_-90", _quarter_turn_rotation("y", -1)),
        ("rot_z_90", _quarter_turn_rotation("z", 1)),
        ("rot_z_-90", _quarter_turn_rotation("z", -1)),
    ]
    identity = np.eye(3, dtype=float)
    preferred_pairs = [
        (
            "rot_y_90_then_rot_z_-90",
            _quarter_turn_rotation("z", -1) @ _quarter_turn_rotation("y", 1),
        ),
    ]

    variants = [("identity", identity), *operations, *preferred_pairs]
    for first_label, first_rotation in operations:
        for second_label, second_rotation in operations:
            variants.append((f"{first_label}_then_{second_label}", second_rotation @ first_rotation))
    return variants


def _deduplicate_initializations(initializations: list[tuple[str, np.ndarray]]) -> list[dict]:
    deduped: list[dict] = []
    index_by_key: dict[tuple[float, ...], int] = {}
    for label, transform in initializations:
        key = tuple(np.round(transform, decimals=6).ravel())
        existing_index = index_by_key.get(key)
        if existing_index is None:
            index_by_key[key] = len(deduped)
            deduped.append({"label": label, "transform": transform, "aliases": [label]})
            continue
        aliases = deduped[existing_index]["aliases"]
        if label not in aliases:
            aliases.append(label)
    return deduped


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


def _canonical_mri_to_head_initializations(
    source_cloud: o3d.geometry.PointCloud,
    target_cloud: o3d.geometry.PointCloud,
    compose_order: str = "centroid_lock",
) -> list[tuple[str, np.ndarray]]:
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
    variants = []
    for label, post_rotation in _right_angle_orientation_variants():
        variant_label = "ras_to_head" if label == "identity" else f"ras_to_head_{label}"
        variants.append((variant_label, post_rotation @ ras_to_head))

    initializations = []
    for label, rotation in variants:
        if np.linalg.det(rotation) < 0:
            continue
        initializations.append(
            (
                label,
                _compose_canonical_initial_transform(
                    rotation,
                    source_centroid,
                    target_centroid,
                    compose_order,
                ),
            )
        )
    return initializations


def _quantile_mask(points: np.ndarray, axis_index: int, keep_fraction: float) -> np.ndarray:
    threshold = np.quantile(points[:, axis_index], 1.0 - keep_fraction)
    return points[:, axis_index] >= threshold


def _subset_cloud(point_cloud: o3d.geometry.PointCloud, mask: np.ndarray) -> o3d.geometry.PointCloud:
    indices = np.flatnonzero(mask).tolist()
    return point_cloud.select_by_index(indices)


def _sphere_filter(point_cloud: o3d.geometry.PointCloud, radius_mm: float) -> o3d.geometry.PointCloud:
    """Keep only points within *radius_mm* of the head-centre estimate.

    The centroid is computed iteratively so that neck/shoulder points cannot
    bias it downward:
      Pass 1 – seed centre from the top-60 % of points along the tallest axis
               (avoids neck pulling the mean down).
      Pass 2 – recompute centre from the points kept in pass 1, then apply the
               final crop at the requested radius.
    """
    points = np.asarray(point_cloud.points)
    if len(points) == 0 or radius_mm <= 0:
        return point_cloud

    # Which raw axis has the most spread? Use it as the "vertical" axis.
    ranges = points.max(axis=0) - points.min(axis=0)
    tall_axis = int(np.argmax(ranges))

    # Pass 1: seed from upper 60 % along the tallest axis.
    upper_thresh = np.quantile(points[:, tall_axis], 0.40)
    upper_mask = points[:, tall_axis] >= upper_thresh
    seed_centroid = points[upper_mask].mean(axis=0)
    pass1_dists = np.linalg.norm(points - seed_centroid, axis=1)
    pass1_mask = pass1_dists <= radius_mm * 1.4

    # Pass 2: recompute centre from pass-1 survivors, apply final radius.
    if pass1_mask.sum() == 0:
        pass1_mask = np.ones(len(points), dtype=bool)  # fallback
    refined_centroid = points[pass1_mask].mean(axis=0)
    final_dists = np.linalg.norm(points - refined_centroid, axis=1)
    mask = final_dists <= radius_mm
    kept = _subset_cloud(point_cloud, mask)
    print(
        f"  [neck_filter] radius={radius_mm:.0f} mm  "
        f"tall_axis={tall_axis}  "
        f"kept {mask.sum()}/{len(points)} points"
    )
    return kept


def _duplicate_subset(point_cloud: o3d.geometry.PointCloud, subset_mask: np.ndarray, weight: int) -> o3d.geometry.PointCloud:
    points = np.asarray(point_cloud.points)
    subset_points = points[subset_mask]
    if len(subset_points) == 0 or weight <= 1:
        return point_cloud

    stacked_points = [points, *([subset_points] * (weight - 1))]
    weighted = o3d.geometry.PointCloud()
    weighted.points = o3d.utility.Vector3dVector(np.vstack(stacked_points))
    return weighted


def _colorize_cloud_by_mask(
    point_cloud: o3d.geometry.PointCloud,
    mask: np.ndarray,
    base_color: tuple[float, float, float],
    highlight_color: tuple[float, float, float],
) -> o3d.geometry.PointCloud:
    colored = o3d.geometry.PointCloud(point_cloud)
    points = np.asarray(colored.points)
    colors = np.tile(np.asarray(base_color, dtype=float), (len(points), 1))
    colors[np.asarray(mask, dtype=bool)] = np.asarray(highlight_color, dtype=float)
    colored.colors = o3d.utility.Vector3dVector(colors)
    return colored


def _face_mask_from_points(points: np.ndarray, axis_index: int, face_fraction: float) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    return _quantile_mask(points, axis_index, face_fraction)


def _face_mask(point_cloud: o3d.geometry.PointCloud, axis_index: int, face_fraction: float) -> np.ndarray:
    return _face_mask_from_points(np.asarray(point_cloud.points), axis_index, face_fraction)


def _transformed_face_mask(
    point_cloud: o3d.geometry.PointCloud,
    transform: np.ndarray,
    axis_index: int,
    face_fraction: float,
) -> np.ndarray:
    points = np.asarray(point_cloud.points)
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    return _face_mask_from_points(_transform_points(points, transform), axis_index, face_fraction)


def _colorize_face_region(
    point_cloud: o3d.geometry.PointCloud,
    axis_index: int,
    face_fraction: float,
    base_color: tuple[float, float, float],
    face_color: tuple[float, float, float],
    mask: np.ndarray | None = None,
) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    if mask is None:
        mask = _face_mask(point_cloud, axis_index, face_fraction)
    return _colorize_cloud_by_mask(point_cloud, mask, base_color, face_color), np.asarray(mask, dtype=bool)


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points_h = np.column_stack([points, np.ones(len(points), dtype=float)])
    return (np.asarray(transform, dtype=float) @ points_h.T).T[:, :3]


def _sample_points(points: np.ndarray, max_points: int, seed: int = 42) -> np.ndarray:
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(points), size=max_points, replace=False))
    return points[indices]


def _rmse_to_target(source_cloud: o3d.geometry.PointCloud, target_cloud: o3d.geometry.PointCloud, transform: np.ndarray) -> float:
    moved = o3d.geometry.PointCloud(source_cloud)
    moved.transform(transform)
    distances = np.asarray(moved.compute_point_cloud_distance(target_cloud), dtype=float)
    if len(distances) == 0:
        return float("inf")
    return float(np.sqrt(np.mean(np.square(distances))))


def _translation_only_icp(
    source_cloud: o3d.geometry.PointCloud,
    target_cloud: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
    threshold: float,
    max_iteration: int,
    tolerance: float = 1e-3,
) -> tuple[np.ndarray, float, float]:
    source_points = np.asarray(source_cloud.points, dtype=float)
    target_points = np.asarray(target_cloud.points, dtype=float)
    if len(source_points) == 0 or len(target_points) == 0:
        return np.asarray(init_transform, dtype=float).copy(), 0.0, float("inf")

    current = np.asarray(init_transform, dtype=float).copy()
    kdtree = o3d.geometry.KDTreeFlann(target_cloud)

    for _ in range(max_iteration):
        moved_points = _transform_points(source_points, current)
        matched_source = []
        matched_target = []
        squared_distances = []

        for point in moved_points:
            neighbors, indices, dist2 = kdtree.search_knn_vector_3d(point, 1)
            if neighbors == 0:
                continue
            distance = math.sqrt(dist2[0])
            if distance > threshold:
                continue
            matched_source.append(point)
            matched_target.append(target_points[indices[0]])
            squared_distances.append(dist2[0])

        if not matched_source:
            break

        matched_source = np.asarray(matched_source, dtype=float)
        matched_target = np.asarray(matched_target, dtype=float)
        delta = np.mean(matched_target - matched_source, axis=0)
        current[:3, 3] += delta
        if np.linalg.norm(delta) <= tolerance:
            break

    moved_points = _transform_points(source_points, current)
    inlier_squared_distances = []
    for point in moved_points:
        neighbors, _indices, dist2 = kdtree.search_knn_vector_3d(point, 1)
        if neighbors == 0:
            continue
        if dist2[0] <= threshold * threshold:
            inlier_squared_distances.append(dist2[0])

    if not inlier_squared_distances:
        return current, 0.0, float("inf")
    inlier_squared_distances = np.asarray(inlier_squared_distances, dtype=float)
    fitness = float(len(inlier_squared_distances) / max(len(source_points), 1))
    rmse = float(np.sqrt(np.mean(inlier_squared_distances)))
    return current, fitness, rmse


def _face_only_rigid_polish(
    source_cloud: o3d.geometry.PointCloud,
    target_cloud: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
    axis_index: int,
    face_fraction: float,
    threshold: float,
    max_iteration: int,
) -> tuple[np.ndarray, float, float, dict]:
    source_face_mask = _transformed_face_mask(source_cloud, init_transform, axis_index, face_fraction)
    target_face_mask = _face_mask(target_cloud, axis_index, face_fraction)
    source_face = _subset_cloud(source_cloud, source_face_mask)
    target_face = _subset_cloud(target_cloud, target_face_mask)

    if len(source_face.points) == 0 or len(target_face.points) == 0:
        return np.asarray(init_transform, dtype=float).copy(), 0.0, float("inf"), {
            "threshold": float(threshold),
            "max_iteration": int(max_iteration),
            "source_points": int(len(source_face.points)),
            "target_points": int(len(target_face.points)),
        }

    result = o3d.pipelines.registration.registration_icp(
        source_face,
        target_face,
        threshold,
        np.asarray(init_transform, dtype=float),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration),
    )
    rigid = project_to_rigid_transform(result.transformation)
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source_face,
        target_face,
        threshold,
        rigid,
    )
    return rigid, float(evaluation.fitness), float(evaluation.inlier_rmse), {
        "threshold": float(threshold),
        "max_iteration": int(max_iteration),
        "source_points": int(len(source_face.points)),
        "target_points": int(len(target_face.points)),
    }


def _fitness_and_rmse(
    source_cloud: o3d.geometry.PointCloud,
    target_cloud: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
    threshold: float,
    max_iteration: int,
    translation_first: bool = False,
    translation_first_threshold: float | None = None,
    translation_first_max_iteration: int = 40,
    translation_last: bool = False,
    translation_last_threshold: float | None = None,
    translation_last_max_iteration: int = 20,
) -> tuple[np.ndarray, float, float, dict | None]:
    translation_stages: dict[str, dict] = {}
    initial_transform = np.asarray(init_transform, dtype=float)
    if translation_first:
        translation_threshold = (
            float(translation_first_threshold)
            if translation_first_threshold is not None
            else float(max(threshold * 10.0, 30.0))
        )
        initial_transform, translation_fitness, translation_rmse = _translation_only_icp(
            source_cloud,
            target_cloud,
            initial_transform,
            translation_threshold,
            translation_first_max_iteration,
        )
        translation_stages["pre_rigid"] = {
            "threshold": translation_threshold,
            "max_iteration": translation_first_max_iteration,
            "fitness": translation_fitness,
            "rmse": translation_rmse,
            "transform": initial_transform.tolist(),
        }

    result = o3d.pipelines.registration.registration_icp(
        source_cloud,
        target_cloud,
        threshold,
        initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration),
    )
    rigid = project_to_rigid_transform(result.transformation)
    fitness = float(result.fitness)
    icp_rmse = float(result.inlier_rmse)

    if translation_last:
        polish_threshold = (
            float(translation_last_threshold)
            if translation_last_threshold is not None
            else float(max(threshold * 4.0, 10.0))
        )
        rigid, polish_fitness, polish_rmse = _translation_only_icp(
            source_cloud,
            target_cloud,
            rigid,
            polish_threshold,
            translation_last_max_iteration,
        )
        translation_stages["post_rigid"] = {
            "threshold": polish_threshold,
            "max_iteration": translation_last_max_iteration,
            "fitness": polish_fitness,
            "rmse": polish_rmse,
            "transform": rigid.tolist(),
        }
        evaluation = o3d.pipelines.registration.evaluate_registration(
            source_cloud,
            target_cloud,
            threshold,
            rigid,
        )
        fitness = float(evaluation.fitness)
        icp_rmse = float(evaluation.inlier_rmse)

    return rigid, fitness, icp_rmse, (translation_stages or None)


def _source_variant_for_method(
    mri_cloud: o3d.geometry.PointCloud,
    method_name: str,
    transform: np.ndarray,
    axis_index: int,
    face_fraction: float,
    face_weight: int,
) -> o3d.geometry.PointCloud:
    if method_name == "full":
        return mri_cloud

    mri_face_mask = _transformed_face_mask(
        mri_cloud,
        transform,
        axis_index,
        face_fraction,
    )
    if method_name == "anterior_crop":
        return _subset_cloud(mri_cloud, mri_face_mask)
    if method_name == "face_weighted":
        return _duplicate_subset(mri_cloud, mri_face_mask, face_weight)
    raise ValueError(f"Unsupported method: {method_name}")


def _global_initialization(source_cloud: o3d.geometry.PointCloud, target_cloud: o3d.geometry.PointCloud, voxel_size: float) -> np.ndarray:
    source_down, source_fpfh = preprocess_point_cloud(source_cloud, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target_cloud, voxel_size)
    result = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    return project_to_rigid_transform(result.transformation)


def _write_starting_pose_contact_sheets(
    output_dir: Path,
    outside_cloud: o3d.geometry.PointCloud,
    mri_cloud: o3d.geometry.PointCloud,
    canonical_init_compose_order: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    canonical_candidates = _deduplicate_initializations(
        _canonical_mri_to_head_initializations(
            mri_cloud,
            outside_cloud,
            compose_order=canonical_init_compose_order,
        )
    )
    outside_points = _sample_points(np.asarray(outside_cloud.points), 4000, seed=7)
    mri_points = _sample_points(np.asarray(mri_cloud.points), 4000, seed=11)

    transformed_candidates = []
    for candidate in canonical_candidates:
        transformed_candidates.append(
            {
                "label": candidate["label"],
                "aliases": candidate["aliases"],
                "points": _transform_points(mri_points, candidate["transform"]),
            }
        )

    combined_points = np.vstack([outside_points] + [candidate["points"] for candidate in transformed_candidates])
    mins = combined_points.min(axis=0)
    maxs = combined_points.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    half_extent = 0.55 * span if span > 0 else 1.0

    projections = [
        ("xy", (0, 1), ("x", "y")),
        ("xz", (0, 2), ("x", "z")),
        ("yz", (1, 2), ("y", "z")),
    ]
    n_candidates = len(transformed_candidates)
    cols = 4
    rows = math.ceil(n_candidates / cols)

    for suffix, axes, labels in projections:
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 4.8, rows * 3.6), squeeze=False)
        for ax in axs.ravel():
            ax.axis("off")

        axis_limits = [
            (center[axis] - half_extent, center[axis] + half_extent)
            for axis in axes
        ]
        outside_proj = outside_points[:, axes]
        for ax, candidate in zip(axs.ravel(), transformed_candidates):
            moved_proj = candidate["points"][:, axes]
            ax.scatter(outside_proj[:, 0], outside_proj[:, 1], s=0.5, c="#8ea9ff", alpha=0.30, linewidths=0)
            ax.scatter(moved_proj[:, 0], moved_proj[:, 1], s=0.5, c="#f2c230", alpha=0.30, linewidths=0)
            alias_note = "" if len(candidate["aliases"]) == 1 else f" (+{len(candidate['aliases']) - 1} aliases)"
            ax.set_title(f"{candidate['label']}{alias_note}", fontsize=8)
            ax.set_xlim(*axis_limits[0])
            ax.set_ylim(*axis_limits[1])
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(labels[0], fontsize=7)
            ax.set_ylabel(labels[1], fontsize=7)
            ax.tick_params(labelsize=6)
            ax.axis("on")

        fig.suptitle(
            f"Canonical starting poses projected onto {labels[0]}{labels[1]} plane",
            fontsize=14,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(output_dir / f"starting_pose_variants_{suffix}.png", dpi=200)
        plt.close(fig)


def _write_final_registration_png(
    output_dir: Path,
    report: dict,
    mri_cloud: o3d.geometry.PointCloud,
    outside_cloud: o3d.geometry.PointCloud,
    axis_index: int,
    face_fraction: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    method_name = report["selected_method"]
    method_report = report["methods"][method_name]
    transform = np.asarray(method_report["transform"], dtype=float)

    outside_points_full = np.asarray(outside_cloud.points)
    mri_points_full = np.asarray(mri_cloud.points)
    outside_face_mask_full = _face_mask(outside_cloud, axis_index, face_fraction)
    transformed_points_full = _transform_points(mri_points_full, transform)
    transformed_face_mask_full = _face_mask_from_points(transformed_points_full, axis_index, face_fraction)

    outside_points = _sample_points(outside_points_full, 8000, seed=17)
    transformed_points = _sample_points(transformed_points_full, 8000, seed=19)
    outside_face_points = _sample_points(outside_points_full[outside_face_mask_full], 3000, seed=23)
    transformed_face_points = _sample_points(transformed_points_full[transformed_face_mask_full], 3000, seed=29)

    combined_points = np.vstack([outside_points, transformed_points])
    mins = combined_points.min(axis=0)
    maxs = combined_points.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    half_extent = 0.55 * span if span > 0 else 1.0

    projections = [
        ("XY", (0, 1), ("x", "y")),
        ("XZ", (0, 2), ("x", "z")),
        ("YZ", (1, 2), ("y", "z")),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.4), squeeze=False)

    for ax, (title, axes, labels) in zip(axs.ravel(), projections):
        axis_limits = [
            (center[axis] - half_extent, center[axis] + half_extent)
            for axis in axes
        ]
        ax.scatter(
            outside_points[:, axes[0]],
            outside_points[:, axes[1]],
            s=0.6,
            c="#9eafc7",
            alpha=0.22,
            linewidths=0,
        )
        if len(outside_face_points) > 0:
            ax.scatter(
                outside_face_points[:, axes[0]],
                outside_face_points[:, axes[1]],
                s=0.8,
                c="#1a7aff",
                alpha=0.65,
                linewidths=0,
            )
        ax.scatter(
            transformed_points[:, axes[0]],
            transformed_points[:, axes[1]],
            s=0.6,
            c="#e8c13a",
            alpha=0.22,
            linewidths=0,
        )
        if len(transformed_face_points) > 0:
            ax.scatter(
                transformed_face_points[:, axes[0]],
                transformed_face_points[:, axes[1]],
                s=0.8,
                c="#ff3737",
                alpha=0.65,
                linewidths=0,
            )
        ax.set_title(title, fontsize=12)
        ax.set_xlim(*axis_limits[0])
        ax.set_ylim(*axis_limits[1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(labels[0], fontsize=10)
        ax.set_ylabel(labels[1], fontsize=10)
        ax.tick_params(labelsize=8)

    init_label = method_report["init"]
    fig.suptitle(
        (
            f"Final registration: {method_name} | {init_label} | "
            f"face_rmse={method_report['face_rmse']:.2f} | "
            f"full_rmse={method_report['full_rmse']:.2f}"
        ),
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_dir / "final_registration_selected.png", dpi=220)
    plt.close(fig)


def evaluate_methods(
    outside_cloud: o3d.geometry.PointCloud,
    mri_cloud: o3d.geometry.PointCloud,
    voxel_size: float,
    threshold: float,
    max_iteration: int,
    axis_index: int,
    face_fraction: float,
    face_weight: int,
    init_label: str | None = None,
    canonical_init_compose_order: str = "centroid_lock",
    translation_first: bool = False,
    translation_first_threshold: float | None = None,
    translation_first_max_iteration: int = 40,
    translation_last: bool = False,
    translation_last_threshold: float | None = None,
    translation_last_max_iteration: int = 20,
    face_polish: bool = False,
    face_polish_threshold: float | None = None,
    face_polish_max_iteration: int = 200,
    neck_filter_radius: float | None = None,
) -> dict:
    if neck_filter_radius is not None and neck_filter_radius > 0:
        print(f"Applying neck filter (radius {neck_filter_radius:.0f} mm) to outside cloud...")
        outside_cloud = _sphere_filter(outside_cloud, neck_filter_radius)
        print(f"Applying neck filter (radius {neck_filter_radius:.0f} mm) to MRI cloud...")
        mri_cloud = _sphere_filter(mri_cloud, neck_filter_radius)

    base_init = _global_initialization(mri_cloud, outside_cloud, voxel_size)
    init_candidates = _deduplicate_initializations(
        [("global", base_init)]
        + _principal_axis_initializations(mri_cloud, outside_cloud)
        + _canonical_mri_to_head_initializations(
            mri_cloud,
            outside_cloud,
            compose_order=canonical_init_compose_order,
        )
    )
    if init_label is not None:
        init_candidates = [
            init_candidate
            for init_candidate in init_candidates
            if init_label == init_candidate["label"] or init_label in init_candidate["aliases"]
        ]
        if not init_candidates:
            raise ValueError(f"Unknown init label: {init_label}")

    outside_points = np.asarray(outside_cloud.points)
    outside_face_mask = _quantile_mask(outside_points, axis_index, face_fraction)

    outside_face = _subset_cloud(outside_cloud, outside_face_mask)
    outside_weighted = _duplicate_subset(outside_cloud, outside_face_mask, face_weight)
    methods = {
        "full": outside_cloud,
        "anterior_crop": outside_face,
        "face_weighted": outside_weighted,
    }

    report = {
        "parameters": {
            "voxel_size": voxel_size,
            "threshold": threshold,
            "max_iteration": max_iteration,
            "axis_index": axis_index,
            "face_fraction": face_fraction,
            "face_weight": face_weight,
            "init_label": init_label,
            "canonical_init_compose_order": canonical_init_compose_order,
            "translation_first": translation_first,
            "translation_first_threshold": translation_first_threshold,
            "translation_first_max_iteration": translation_first_max_iteration,
            "translation_last": translation_last,
            "translation_last_threshold": translation_last_threshold,
            "translation_last_max_iteration": translation_last_max_iteration,
            "face_polish": face_polish,
            "face_polish_threshold": face_polish_threshold,
            "face_polish_max_iteration": face_polish_max_iteration,
            "neck_filter_radius": neck_filter_radius,
        },
        "ranking_score_order": ["face_rmse", "full_rmse", "-fitness", "icp_rmse"],
        "candidates": [
            {"label": init_candidate["label"], "aliases": init_candidate["aliases"]}
            for init_candidate in init_candidates
        ],
        "methods": {},
        "candidate_rankings": {},
    }

    for method_name, target_variant in methods.items():
        ranked_candidates = []
        for init_candidate in init_candidates:
            init_label = init_candidate["label"]
            init_transform = init_candidate["transform"]
            source_variant = _source_variant_for_method(
                mri_cloud,
                method_name,
                init_transform,
                axis_index,
                face_fraction,
                face_weight,
            )
            rigid, fitness, icp_rmse, refinement_stages = _fitness_and_rmse(
                source_variant,
                target_variant,
                init_transform,
                threshold,
                max_iteration,
                translation_first=translation_first,
                translation_first_threshold=translation_first_threshold,
                translation_first_max_iteration=translation_first_max_iteration,
                translation_last=translation_last,
                translation_last_threshold=translation_last_threshold,
                translation_last_max_iteration=translation_last_max_iteration,
            )
            if face_polish:
                polish_threshold = (
                    float(face_polish_threshold)
                    if face_polish_threshold is not None
                    else float(max(threshold * 2.0, 6.0))
                )
                rigid, polish_fitness, polish_icp_rmse, polish_info = _face_only_rigid_polish(
                    mri_cloud,
                    outside_cloud,
                    rigid,
                    axis_index,
                    face_fraction,
                    polish_threshold,
                    face_polish_max_iteration,
                )
                source_variant = _source_variant_for_method(
                    mri_cloud,
                    method_name,
                    rigid,
                    axis_index,
                    face_fraction,
                    face_weight,
                )
                evaluation = o3d.pipelines.registration.evaluate_registration(
                    source_variant,
                    target_variant,
                    threshold,
                    rigid,
                )
                fitness = float(evaluation.fitness)
                icp_rmse = float(evaluation.inlier_rmse)
                if refinement_stages is None:
                    refinement_stages = {}
                refinement_stages["face_polish"] = {
                    **polish_info,
                    "fitness": polish_fitness,
                    "icp_rmse": polish_icp_rmse,
                    "transform": rigid.tolist(),
                }
            full_rmse = _rmse_to_target(mri_cloud, outside_cloud, rigid)
            final_mri_face_mask = _transformed_face_mask(
                mri_cloud,
                rigid,
                axis_index,
                face_fraction,
            )
            final_mri_face = _subset_cloud(mri_cloud, final_mri_face_mask)
            face_rmse = _rmse_to_target(final_mri_face, outside_face, rigid)
            score = (face_rmse, full_rmse, -fitness, icp_rmse)
            candidate = {
                "init": init_label,
                "init_aliases": init_candidate["aliases"],
                "transform": rigid.tolist(),
                "fitness": fitness,
                "icp_rmse": icp_rmse,
                "full_rmse": full_rmse,
                "face_rmse": face_rmse,
            }
            if refinement_stages is not None:
                candidate["refinement_stages"] = refinement_stages
                translation_stages = {
                    stage_name: stage_info
                    for stage_name, stage_info in refinement_stages.items()
                    if stage_name in {"pre_rigid", "post_rigid"}
                }
                if translation_stages:
                    candidate["translation_stages"] = translation_stages
            ranked_candidates.append((score, candidate))
        ranked_candidates.sort(key=lambda item: item[0])
        ordered_candidates = []
        for rank, (_score, candidate) in enumerate(ranked_candidates, start=1):
            candidate["rank"] = rank
            ordered_candidates.append(candidate)
        report["candidate_rankings"][method_name] = ordered_candidates
        report["methods"][method_name] = ordered_candidates[0]

    report["selected_method"] = min(
        report["methods"],
        key=lambda key: (
            report["methods"][key]["face_rmse"],
            report["methods"][key]["full_rmse"],
            -report["methods"][key]["fitness"],
        ),
    )
    report["method_ranking"] = []
    for rank, method_name in enumerate(
        sorted(
            report["methods"],
            key=lambda key: (
                report["methods"][key]["face_rmse"],
                report["methods"][key]["full_rmse"],
                -report["methods"][key]["fitness"],
                report["methods"][key]["icp_rmse"],
            ),
        ),
        start=1,
    ):
        best = report["methods"][method_name]
        report["method_ranking"].append(
            {
                "rank": rank,
                "method": method_name,
                "init": best["init"],
                "init_aliases": best["init_aliases"],
                "fitness": best["fitness"],
                "icp_rmse": best["icp_rmse"],
                "full_rmse": best["full_rmse"],
                "face_rmse": best["face_rmse"],
            }
        )
    return report


def _write_ranking_summary(output_dir: Path, report: dict) -> None:
    lines = [
        "# Ranking Summary",
        "",
        f"Score order: {', '.join(report['ranking_score_order'])}",
        "",
        "## Method Ranking",
        "",
        "| Rank | Method | Init | Face RMSE | Full RMSE | Fitness | ICP RMSE |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for entry in report["method_ranking"]:
        lines.append(
            f"| {entry['rank']} | {entry['method']} | {entry['init']} | "
            f"{entry['face_rmse']:.3f} | {entry['full_rmse']:.3f} | "
            f"{entry['fitness']:.4f} | {entry['icp_rmse']:.4f} |"
        )

    for method_name, candidates in report["candidate_rankings"].items():
        lines.extend(
            [
                "",
                f"## {method_name} Candidate Ranking",
                "",
                "| Rank | Init | Face RMSE | Full RMSE | Fitness | ICP RMSE |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for candidate in candidates:
            lines.append(
                f"| {candidate['rank']} | {candidate['init']} | "
                f"{candidate['face_rmse']:.3f} | {candidate['full_rmse']:.3f} | "
                f"{candidate['fitness']:.4f} | {candidate['icp_rmse']:.4f} |"
            )

    (output_dir / "ranking_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_transformed_clouds(
    output_dir: Path,
    report: dict,
    mri_cloud: o3d.geometry.PointCloud,
    outside_cloud: o3d.geometry.PointCloud,
    axis_index: int,
    face_fraction: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_dir / "outside_cloud.ply"), outside_cloud)
    outside_colored, outside_face_mask = _colorize_face_region(
        outside_cloud,
        axis_index,
        face_fraction,
        base_color=(0.62, 0.68, 0.78),
        face_color=(0.10, 0.48, 1.00),
    )
    o3d.io.write_point_cloud(str(output_dir / "outside_face_colored.ply"), outside_colored)
    for method_name, method_report in report["methods"].items():
        transformed = o3d.geometry.PointCloud(mri_cloud)
        transformed.transform(np.asarray(method_report["transform"], dtype=float))
        o3d.io.write_point_cloud(str(output_dir / f"mri_{method_name}.ply"), transformed)
        transformed_colored, _ = _colorize_face_region(
            transformed,
            axis_index,
            face_fraction,
            base_color=(0.95, 0.78, 0.15),
            face_color=(1.00, 0.20, 0.20),
        )
        o3d.io.write_point_cloud(str(output_dir / f"mri_{method_name}_face_colored.ply"), transformed_colored)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2))
    _write_ranking_summary(output_dir, report)
    _write_starting_pose_contact_sheets(
        output_dir,
        outside_cloud,
        mri_cloud,
        canonical_init_compose_order=report["parameters"]["canonical_init_compose_order"],
    )
    _write_final_registration_png(
        output_dir,
        report,
        mri_cloud,
        outside_cloud,
        axis_index=axis_index,
        face_fraction=face_fraction,
    )


def _visualize_selected(
    report: dict,
    mri_cloud: o3d.geometry.PointCloud,
    outside_cloud: o3d.geometry.PointCloud,
    axis_index: int,
    face_fraction: float,
) -> None:
    method = report["selected_method"]
    transform = np.asarray(report["methods"][method]["transform"], dtype=float)
    target, _outside_face_mask = _colorize_face_region(
        outside_cloud,
        axis_index,
        face_fraction,
        base_color=(0.62, 0.68, 0.78),
        face_color=(0.10, 0.48, 1.00),
    )
    transformed = o3d.geometry.PointCloud(mri_cloud)
    transformed.transform(transform)
    transformed, _mri_face_mask = _colorize_face_region(
        transformed,
        axis_index,
        face_fraction,
        base_color=(0.95, 0.78, 0.15),
        face_color=(1.00, 0.20, 0.20),
    )
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
    parser.add_argument(
        "--init-label",
        help="Optional exact init label or alias to evaluate instead of the full candidate set",
    )
    parser.add_argument(
        "--canonical-init-compose",
        choices=["centroid_lock", "translate_then_rotate"],
        default="centroid_lock",
        help="How canonical ras_to_head starts are composed: keep centroids aligned after rotation, or translate centroids first then rotate",
    )
    parser.add_argument(
        "--translation-first",
        action="store_true",
        help="Run a translation-only ICP stage before the standard rigid ICP stage",
    )
    parser.add_argument(
        "--translation-first-threshold",
        type=float,
        help="Optional inlier threshold in mm for the translation-only ICP stage",
    )
    parser.add_argument(
        "--translation-first-max-iteration",
        type=int,
        default=40,
        help="Maximum iterations for the translation-only ICP stage",
    )
    parser.add_argument(
        "--translation-last",
        action="store_true",
        help="Run a translation-only polish stage after the rigid ICP stage",
    )
    parser.add_argument(
        "--translation-last-threshold",
        type=float,
        help="Optional inlier threshold in mm for the post-rigid translation-only polish stage",
    )
    parser.add_argument(
        "--translation-last-max-iteration",
        type=int,
        default=20,
        help="Maximum iterations for the post-rigid translation-only polish stage",
    )
    parser.add_argument(
        "--face-polish",
        action="store_true",
        help="Run a final low-threshold rigid ICP pass on the aligned face region",
    )
    parser.add_argument(
        "--face-polish-threshold",
        type=float,
        help="Optional inlier threshold in mm for the final face-only rigid polish stage",
    )
    parser.add_argument(
        "--face-polish-max-iteration",
        type=int,
        default=200,
        help="Maximum iterations for the final face-only rigid polish stage",
    )
    parser.add_argument(
        "--neck-filter-radius",
        type=float,
        default=None,
        metavar="MM",
        help=(
            "If set, crop both clouds to a sphere of this radius (mm) centred at each "
            "cloud's centroid before registration.  Removes neck, shoulders, and hair. "
            "A value of 100-130 mm typically works well for head-sized scans."
        ),
    )
    parser.add_argument(
        "--write-clouds",
        default=None,
        help=(
            "Output directory for transformed clouds, JSON report, and PNGs. "
            "Defaults to a timestamped subdirectory of the current working directory "
            "(runs/<YYYYMMDD_HHMMSS>/)."
        ),
    )
    parser.add_argument("--no-write", action="store_true", help="Suppress automatic output-directory creation")
    parser.add_argument("--visualize", action="store_true", help="Visualize the selected registration result")
    return parser


def main() -> None:
    import datetime

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
        init_label=args.init_label,
        canonical_init_compose_order=args.canonical_init_compose,
        translation_first=args.translation_first,
        translation_first_threshold=args.translation_first_threshold,
        translation_first_max_iteration=args.translation_first_max_iteration,
        translation_last=args.translation_last,
        translation_last_threshold=args.translation_last_threshold,
        translation_last_max_iteration=args.translation_last_max_iteration,
        face_polish=args.face_polish,
        face_polish_threshold=args.face_polish_threshold,
        face_polish_max_iteration=args.face_polish_max_iteration,
        neck_filter_radius=args.neck_filter_radius,
    )

    print(json.dumps(report, indent=2))

    write_dir: Path | None = None
    if not args.no_write:
        if args.write_clouds:
            write_dir = Path(args.write_clouds)
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            write_dir = Path.cwd() / "runs" / timestamp
            print(f"[output] Writing to {write_dir}")

    if write_dir is not None:
        _save_transformed_clouds(
            write_dir,
            report,
            mri_cloud,
            outside_cloud,
            axis_index=axis_index,
            face_fraction=args.face_fraction,
        )

    if args.visualize:
        _visualize_selected(
            report,
            mri_cloud,
            outside_cloud,
            axis_index=axis_index,
            face_fraction=args.face_fraction,
        )


if __name__ == "__main__":
    main()
