"""
Geometry conversion utilities for YORC.

Functions for converting between Open3D and PyVista formats.
"""

import numpy as np


def o3d_to_pyvista(o3d_geometry):
    """
    Convert Open3D geometry to PyVista PolyData.

    Args:
        o3d_geometry: Open3D PointCloud or TriangleMesh

    Returns:
        PyVista PolyData object
    """
    import open3d as o3d
    import pyvista as pv

    if isinstance(o3d_geometry, o3d.geometry.PointCloud):
        points = np.asarray(o3d_geometry.points)
        cloud = pv.PolyData(points)

        if o3d_geometry.has_colors():
            colors = (np.asarray(o3d_geometry.colors) * 255).astype(np.uint8)
            cloud["RGB"] = colors

        if o3d_geometry.has_normals():
            normals = np.asarray(o3d_geometry.normals)
            cloud["Normals"] = normals

        return cloud

    elif isinstance(o3d_geometry, o3d.geometry.TriangleMesh):
        vertices = np.asarray(o3d_geometry.vertices)
        faces = np.asarray(o3d_geometry.triangles)

        # PyVista needs face count prepended to each face
        n_faces = len(faces)
        faces_pv = np.hstack([np.full((n_faces, 1), 3, dtype=np.int64), faces]).ravel()

        mesh = pv.PolyData(vertices, faces_pv)

        if o3d_geometry.has_vertex_colors():
            colors = (np.asarray(o3d_geometry.vertex_colors) * 255).astype(np.uint8)
            mesh["RGB"] = colors

        if o3d_geometry.has_vertex_normals():
            normals = np.asarray(o3d_geometry.vertex_normals)
            mesh["Normals"] = normals

        return mesh

    else:
        raise TypeError(f"Unsupported Open3D geometry type: {type(o3d_geometry)}")


def pyvista_to_o3d(pv_data):
    """
    Convert PyVista PolyData to Open3D geometry.

    Args:
        pv_data: PyVista PolyData object

    Returns:
        Open3D PointCloud or TriangleMesh
    """
    import open3d as o3d

    points = np.asarray(pv_data.points)

    # Check if this is a mesh (has faces) or just points
    if pv_data.n_cells > 0 and pv_data.faces is not None and len(pv_data.faces) > 0:
        # It's a mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(points)

        # Convert PyVista face format to Open3D
        # PyVista: [n_verts, v0, v1, v2, n_verts, v0, v1, v2, ...]
        faces = pv_data.faces.reshape(-1, 4)[:, 1:4]  # Skip the count column
        mesh.triangles = o3d.utility.Vector3iVector(faces)

        if "RGB" in pv_data.array_names:
            colors = pv_data["RGB"].astype(np.float64) / 255.0
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

        return mesh

    else:
        # It's a point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if "RGB" in pv_data.array_names:
            colors = pv_data["RGB"].astype(np.float64) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)

        if "Normals" in pv_data.array_names:
            normals = pv_data["Normals"]
            pcd.normals = o3d.utility.Vector3dVector(normals)

        return pcd


def create_sphere_marker(center, radius=3.0, color=(1, 0, 0)):
    """
    Create a sphere marker for visualization.

    Args:
        center: (x, y, z) position
        radius: Sphere radius
        color: RGB color tuple (0-1 scale)

    Returns:
        PyVista Sphere
    """
    import pyvista as pv

    sphere = pv.Sphere(radius=radius, center=center)
    return sphere


def create_point_markers(points, radius=3.0, colors=None):
    """
    Create sphere markers for multiple points.

    Args:
        points: Nx3 array of positions
        radius: Sphere radius
        colors: Optional list of RGB colors for each point

    Returns:
        PyVista MultiBlock containing all spheres
    """
    import pyvista as pv

    markers = pv.MultiBlock()
    for i, point in enumerate(points):
        sphere = pv.Sphere(radius=radius, center=point)
        if colors is not None and i < len(colors):
            sphere["color"] = np.array([colors[i]] * sphere.n_points)
        markers.append(sphere)

    return markers


def decimate_for_display(geometry, target_points=10000):
    """
    Decimate a point cloud or mesh for faster display.

    Args:
        geometry: Open3D PointCloud or TriangleMesh
        target_points: Target number of points

    Returns:
        Decimated geometry (same type as input)
    """
    import open3d as o3d

    if isinstance(geometry, o3d.geometry.PointCloud):
        n_points = len(geometry.points)
        if n_points <= target_points:
            return geometry

        # Use uniform downsampling
        ratio = int(np.ceil(n_points / target_points))
        return geometry.uniform_down_sample(ratio)

    elif isinstance(geometry, o3d.geometry.TriangleMesh):
        n_triangles = len(geometry.triangles)
        if n_triangles <= target_points:
            return geometry

        ratio = target_points / n_triangles
        return geometry.simplify_quadric_decimation(int(n_triangles * ratio))

    return geometry
