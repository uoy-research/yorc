"""
Automatic landmark detection for YORC.

Detects red-green colored roundel markers on the MEG helmet using
color filtering and DBSCAN clustering.
"""

import numpy as np
import open3d as o3d


def _nearest_point_dist(a, B):
    """Calculate minimum distance from point a to any point in array B."""
    dists = np.linalg.norm(a - B, axis=1)
    return dists.min()


def find_landmarks(
    source_cloud,
    red_color=(0.8, 0.1, 0.3),
    green_color=(0.3, 0.8, 0.3),
    filter_percentile=0.020,
    proximity_threshold=5.0,
    cluster_eps=5.0,
    cluster_min_points=10,
):
    """
    Automatically find landmarks when they're red-green roundels on the helmet.

    Args:
        source_cloud: Open3D PointCloud with color information
        red_color: Target RGB values for red detection (0-1 scale)
        green_color: Target RGB values for green detection (0-1 scale)
        filter_percentile: Fraction of points to keep in color filter
        proximity_threshold: Max distance (mm) for red-green point pairing
        cluster_eps: DBSCAN epsilon (distance threshold for clustering)
        cluster_min_points: DBSCAN minimum points per cluster

    Returns:
        numpy array of landmark center positions (N x 3), or None if detection fails
    """
    diagnostics = detect_landmark_candidates(
        source_cloud,
        red_color=red_color,
        green_color=green_color,
        filter_percentile=filter_percentile,
        proximity_threshold=proximity_threshold,
        cluster_eps=cluster_eps,
        cluster_min_points=cluster_min_points,
    )
    return diagnostics["selected_centers"]


def detect_landmark_candidates(
    source_cloud,
    red_color=(0.8, 0.1, 0.3),
    green_color=(0.3, 0.8, 0.3),
    filter_percentile=0.020,
    proximity_threshold=5.0,
    cluster_eps=5.0,
    cluster_min_points=10,
    min_landmarks=5,
    max_landmarks=7,
):
    """Return candidate helmet fiducial clusters and the subset chosen for registration."""
    diagnostics = {
        "candidate_centers": np.empty((0, 3), dtype=float),
        "candidate_counts": np.empty((0,), dtype=int),
        "selected_centers": None,
        "selected_counts": np.empty((0,), dtype=int),
        "n_keep": 0,
        "n_overlap_points": 0,
        "n_unique_overlap_points": 0,
        "n_clusters": 0,
        "noise_points": 0,
        "used_largest_subset": False,
        "message": "Automatic fiducial detection did not run.",
    }

    if not source_cloud.has_colors():
        diagnostics["message"] = "Inside cloud has no color data."
        return diagnostics

    colors = np.asarray(source_cloud.colors)
    points = np.asarray(source_cloud.points)
    n_points = len(points)
    n_keep = max(1, int(n_points * filter_percentile))
    diagnostics["n_keep"] = n_keep

    red_scores = 1 - np.linalg.norm(colors - np.array(red_color), axis=1)
    red_point_index = np.argpartition(red_scores, -n_keep)[-n_keep:]

    green_scores = 1 - np.linalg.norm(colors - np.array(green_color), axis=1)
    green_point_index = np.argpartition(green_scores, -n_keep)[-n_keep:]

    good_points = []
    red_points_arr = points[red_point_index]
    green_points_arr = points[green_point_index]

    for pp in green_point_index:
        if _nearest_point_dist(points[pp], red_points_arr) < proximity_threshold:
            good_points.append(int(pp))

    for pp in red_point_index:
        if _nearest_point_dist(points[pp], green_points_arr) < proximity_threshold:
            good_points.append(int(pp))

    diagnostics["n_overlap_points"] = len(good_points)
    unique_good_points = np.array(sorted(set(good_points)), dtype=int)
    diagnostics["n_unique_overlap_points"] = len(unique_good_points)
    if len(unique_good_points) == 0:
        diagnostics["message"] = "No red/green overlap points were found."
        return diagnostics

    good_point_cloud = source_cloud.select_by_index(unique_good_points.tolist())
    cluster_labels = np.array(
        good_point_cloud.cluster_dbscan(
            eps=cluster_eps, min_points=cluster_min_points, print_progress=False
        )
    )
    if len(cluster_labels) == 0 or np.all(cluster_labels == -1):
        diagnostics["noise_points"] = int(len(cluster_labels))
        diagnostics["message"] = "No fiducial clusters were found after DBSCAN."
        return diagnostics

    diagnostics["noise_points"] = int((cluster_labels == -1).sum())
    unique_labels = sorted(label for label in set(cluster_labels.tolist()) if label >= 0)
    if not unique_labels:
        diagnostics["message"] = "No fiducial clusters remained after removing noise points."
        return diagnostics

    good_points_arr = np.asarray(good_point_cloud.points)
    cluster_centers = []
    cluster_counts = []
    for label in unique_labels:
        cluster_mask = cluster_labels == label
        cluster_points = good_points_arr[cluster_mask]
        if len(cluster_points) == 0:
            continue
        cluster_centers.append(cluster_points.mean(axis=0))
        cluster_counts.append(int(cluster_mask.sum()))

    if not cluster_centers:
        diagnostics["message"] = "No fiducial cluster centers could be computed."
        return diagnostics

    cluster_centers = np.asarray(cluster_centers, dtype=float)
    cluster_counts = np.asarray(cluster_counts, dtype=int)
    order = np.argsort(cluster_counts)[::-1]
    cluster_centers = cluster_centers[order]
    cluster_counts = cluster_counts[order]

    diagnostics["candidate_centers"] = cluster_centers
    diagnostics["candidate_counts"] = cluster_counts
    diagnostics["n_clusters"] = int(len(cluster_centers))

    if len(cluster_centers) < min_landmarks:
        diagnostics["message"] = (
            f"Found {len(cluster_centers)} candidate clusters; need at least {min_landmarks}."
        )
        return diagnostics

    selected_count = min(len(cluster_centers), max_landmarks)
    diagnostics["selected_centers"] = cluster_centers[:selected_count]
    diagnostics["selected_counts"] = cluster_counts[:selected_count]
    diagnostics["used_largest_subset"] = len(cluster_centers) > max_landmarks
    if diagnostics["used_largest_subset"]:
        diagnostics["message"] = (
            f"Found {len(cluster_centers)} candidate clusters; keeping the {selected_count} largest."
        )
    else:
        diagnostics["message"] = f"Found {selected_count} candidate clusters."

    return diagnostics


def visualize_landmark_detection(source_cloud, landmarks):
    """
    Visualize detected landmarks overlaid on the source cloud.

    Args:
        source_cloud: Original Open3D PointCloud
        landmarks: Array of landmark positions (N x 3)

    Returns:
        Open3D PointCloud with landmarks highlighted
    """
    # Create spheres at landmark positions
    landmark_meshes = []
    for _i, pos in enumerate(landmarks):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=3.0)
        sphere.translate(pos)
        sphere.paint_uniform_color([1, 0, 0])  # Red
        landmark_meshes.append(sphere)

    return landmark_meshes
