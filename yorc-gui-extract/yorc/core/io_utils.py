"""
I/O utilities for YORC.

Functions for loading and saving meshes, point clouds, and MEG data.
"""

import os

import mne
import numpy as np
import open3d as o3d


def load_mesh(filepath, n_points=100000, fast=False):
    """
    Load a mesh file and convert to point cloud.

    Args:
        filepath: Path to mesh file (.ply, .stl, .obj)
        n_points: Number of points to sample from mesh
        fast: If True, use faster uniform sampling instead of Poisson disk

    Returns:
        tuple: (Open3D PointCloud, original mesh)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format not supported
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Mesh file not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()

    if ext in [".ply", ".stl", ".obj"]:
        mesh = o3d.io.read_triangle_mesh(filepath)
        if mesh.is_empty():
            raise ValueError(f"Failed to load mesh from {filepath}")

        # Sample points from mesh
        if fast:
            # Uniform sampling is much faster than Poisson disk
            point_cloud = mesh.sample_points_uniformly(n_points)
        else:
            point_cloud = mesh.sample_points_poisson_disk(n_points)
        return point_cloud, mesh
    else:
        raise ValueError(f"Unsupported mesh format: {ext}")


def load_mesh_for_display(filepath, max_points=50000):
    """
    Load a mesh file optimized for display with adaptive decimation.

    Args:
        filepath: Path to mesh file
        max_points: Maximum number of points for display (default 50k for safety)

    Returns:
        tuple: (Open3D PointCloud, None) - mesh is not loaded for speed
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Mesh file not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".ply":
        # First try PLY as point cloud (fast path for scanner exports)
        pcd = o3d.io.read_point_cloud(filepath)
        if not pcd.is_empty():
            n_original = len(pcd.points)
            print(f"[io_utils] Loaded {n_original:,} points from {os.path.basename(filepath)}")

            # Adaptive downsampling based on original size
            if n_original > max_points:
                # Use numpy random sampling for speed (much faster than Open3D)
                import random

                random.seed(42)  # Reproducible results
                target_n = max_points
                print(f"[io_utils] Decimating to {target_n:,} points for display")

                indices = random.sample(range(n_original), target_n)
                pcd_points = np.asarray(pcd.points)[indices]
                pcd_colors = None
                if pcd.has_colors():
                    pcd_colors = np.asarray(pcd.colors)[indices]

                pcd_new = o3d.geometry.PointCloud()
                pcd_new.points = o3d.utility.Vector3dVector(pcd_points)
                if pcd_colors is not None:
                    pcd_new.colors = o3d.utility.Vector3dVector(pcd_colors)
                pcd = pcd_new
                print(f"[io_utils] Decimation complete: {len(pcd.points):,} points")

            return pcd, None

        # If point-cloud read is empty, this is likely a mesh-style PLY.
        mesh = o3d.io.read_triangle_mesh(filepath)
        if mesh.is_empty():
            raise ValueError(f"Failed to load PLY as point cloud or mesh: {filepath}")

        print(f"[io_utils] Loaded PLY mesh, sampling {max_points:,} points for display")
        pcd = mesh.sample_points_uniformly(max_points)
        return pcd, None

    elif ext in [".stl", ".obj"]:
        # STL/OBJ must be read as mesh then sampled
        mesh = o3d.io.read_triangle_mesh(filepath)
        if mesh.is_empty():
            raise ValueError(f"Failed to load mesh from {filepath}")
        print(f"[io_utils] Sampling {max_points:,} points from mesh")
        pcd = mesh.sample_points_uniformly(max_points)
        return pcd, None

    else:
        raise ValueError(f"Unsupported format: {ext}")


def load_freesurfer_bem(filepath):
    """
    Load FreeSurfer BEM surface from .fif file.

    Args:
        filepath: Path to FreeSurfer BEM surface (.fif)

    Returns:
        tuple: (Open3D PointCloud, Open3D TriangleMesh)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If no BEM surfaces found or file is invalid
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"BEM file not found: {filepath}")

    try:
        bem_surfaces = mne.read_bem_surfaces(filepath, verbose="error")
        if not bem_surfaces:
            raise ValueError("No BEM surfaces found in .fif file")

        # Use the first surface
        surf = bem_surfaces[0]
        vertices = surf["rr"]  # Vertices in meters
        faces = surf["tris"]  # Triangle indices

        # Convert to millimeters (Open3D units)
        vertices_mm = vertices * 1000.0

        # Create Open3D mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices_mm)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()

        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices_mm)

        return pcd, mesh

    except Exception as e:
        raise ValueError(f"Failed to load BEM surface from {filepath}: {str(e)}") from e


def load_point_cloud(filepath):
    """
    Load a point cloud file directly.

    Args:
        filepath: Path to point cloud file (.ply, .pcd)

    Returns:
        Open3D PointCloud

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format not supported
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Point cloud file not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()

    if ext in [".ply", ".pcd"]:
        pcd = o3d.io.read_point_cloud(filepath)
        if pcd.is_empty():
            raise ValueError(f"Failed to load point cloud from {filepath}")
        return pcd
    else:
        raise ValueError(f"Unsupported point cloud format: {ext}")


def load_meg_data(filepath, preload=False):
    """
    Load MEG data from FIF file.

    Args:
        filepath: Path to .fif file
        preload: Whether to preload data into memory

    Returns:
        mne.io.Raw object

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"MEG data file not found: {filepath}")

    raw = mne.io.read_raw_fif(filepath, verbose="error", preload=preload)
    return raw


def save_meg_data(raw, filepath, overwrite=True):
    """
    Save MEG data to FIF file.

    Args:
        raw: mne.io.Raw object
        filepath: Output path
        overwrite: Whether to overwrite existing file
    """
    raw.save(filepath, overwrite=overwrite)


def load_transform(filepath):
    """
    Load MNE transform from file.

    Args:
        filepath: Path to transform file (_trans.fif)

    Returns:
        mne Transform object
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Transform file not found: {filepath}")

    return mne.read_trans(filepath)


def save_transform(transform, filepath, overwrite=True):
    """
    Save MNE transform to file.

    Args:
        transform: mne Transform object
        filepath: Output path
        overwrite: Whether to overwrite existing file
    """
    mne.write_trans(filepath, transform, overwrite=overwrite)


def validate_files(file_dict):
    """
    Validate that all required files exist.

    Args:
        file_dict: Dictionary mapping file description to path

    Returns:
        tuple: (all_valid, list of missing files)
    """
    missing = []
    for desc, path in file_dict.items():
        if path and not os.path.isfile(path):
            missing.append(f"{desc}: {path}")

    return len(missing) == 0, missing


def get_file_info(filepath):
    """
    Get basic information about a file.

    Args:
        filepath: Path to file

    Returns:
        dict with file information
    """
    if not os.path.isfile(filepath):
        return None

    stat = os.stat(filepath)
    return {
        "path": filepath,
        "filename": os.path.basename(filepath),
        "size_bytes": stat.st_size,
        "size_mb": stat.st_size / (1024 * 1024),
        "extension": os.path.splitext(filepath)[1].lower(),
    }
