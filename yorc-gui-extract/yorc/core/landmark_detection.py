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
    if not source_cloud.has_colors():
        return None

    colors = np.asarray(source_cloud.colors)
    points = np.asarray(source_cloud.points)
    n_points = len(points)

    # Number of points to keep in each color filter
    n_keep = int(n_points * filter_percentile)

    # Filter for red points
    red_scores = 1 - np.linalg.norm(colors - np.array(red_color), axis=1)
    red_point_index = np.argpartition(red_scores, -n_keep)[-n_keep:]

    # Filter for green points
    green_scores = 1 - np.linalg.norm(colors - np.array(green_color), axis=1)
    green_point_index = np.argpartition(green_scores, -n_keep)[-n_keep:]

    # Find overlap: points where both red and green are nearby
    # (characteristic of red-green roundels)
    good_points = []

    red_points_arr = points[red_point_index]
    green_points_arr = points[green_point_index]

    # Keep green points that have a nearby red point
    for pp in green_point_index:
        if _nearest_point_dist(points[pp], red_points_arr) < proximity_threshold:
            good_points.append(pp)

    # Keep red points that have a nearby green point
    for pp in red_point_index:
        if _nearest_point_dist(points[pp], green_points_arr) < proximity_threshold:
            good_points.append(pp)

    if len(good_points) == 0:
        return None

    # Create cloud of points that we believe are part of the target roundels
    good_point_cloud = source_cloud.select_by_index(good_points)

    # Cluster the target points - should give one cluster per roundel
    cluster_labels = np.array(
        good_point_cloud.cluster_dbscan(
            eps=cluster_eps, min_points=cluster_min_points, print_progress=False
        )
    )

    if len(cluster_labels) == 0 or np.all(cluster_labels == -1):
        return None

    # Get unique cluster labels (excluding noise label -1)
    unique_labels = set(cluster_labels)
    unique_labels.discard(-1)

    if len(unique_labels) == 0:
        return None

    # Get mean position of each cluster (landmark center)
    good_points_arr = np.asarray(good_point_cloud.points)
    n_clusters = len(unique_labels)
    cluster_centers = np.zeros([n_clusters, 3])
    cluster_counts = np.zeros(n_clusters)

    for i, point in enumerate(good_points_arr):
        label = cluster_labels[i]
        if label >= 0:  # Skip noise points
            cluster_centers[label] += point
            cluster_counts[label] += 1

    # Avoid division by zero
    cluster_counts[cluster_counts == 0] = 1
    cluster_centers = cluster_centers / cluster_counts[:, np.newaxis]

    # Validate number of landmarks found
    n_landmarks = len(cluster_centers)
    if n_landmarks > 7 or n_landmarks < 5:
        # Return None to signal that automatic detection failed
        return None

    return cluster_centers


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
