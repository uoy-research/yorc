"""
Core registration functions for YORC.

This module contains the main registration pipeline functions for aligning
LIDAR scans, head positions, and MRI data.
"""

import copy

import mne
import numpy as np
import open3d as o3d
from mne.io.constants import FIFF

# Approximate distance from Rubidium chamber to external tip of
# sensor housing that makes contact with head
SENSOR_LENGTH = 6e-3

# Known positions of stickers in helmet reference frame (mm)
HELMET_STICKER_POSITIONS = np.array(
    [
        [102.325, 0.221, 16.345],
        [92.079, 66.226, -27.207],
        [67.431, 113.778, -7.799],
        [-0.117, 138.956, -5.576],
        [-67.431, 113.778, -7.799],
        [-92.079, 66.226, -27.207],
        [-102.325, 0.221, 16.345],
    ]
)


def project_to_rigid_transform(transform):
    """Project a 4x4 transform onto rigid SE(3): rotation + translation only."""
    T = np.asarray(transform, dtype=float).copy()
    R = T[:3, :3]
    U, _, Vt = np.linalg.svd(R)
    R_rigid = U @ Vt
    if np.linalg.det(R_rigid) < 0:
        U[:, -1] *= -1
        R_rigid = U @ Vt
    T[:3, :3] = R_rigid
    T[3, :] = [0.0, 0.0, 0.0, 1.0]
    return T


def extract_meg_sensor_contact_and_detector_points(raw, sensor_length=SENSOR_LENGTH):
    """
    Extract MEG sensor contact points in device frame (meters).

    Uses channel location (loc[:3]) and orientation vector (loc[9:12]),
    shifted by SENSOR_LENGTH toward the scalp contact point.
    """
    contact_points = []
    detector_points = []
    for chan in raw.info["chs"]:
        if chan.get("kind") != FIFF.FIFFV_MEG_CH:
            continue

        loc = np.asarray(chan.get("loc", []), dtype=float)
        if loc.size < 12:
            continue

        pos = loc[:3]
        ori = loc[9:12]
        if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(ori))):
            continue

        ori_norm = np.linalg.norm(ori)
        if ori_norm < 1e-9:
            continue
        ori = ori / ori_norm

        contact = pos - ori * float(sensor_length)
        contact_points.append(contact)
        detector_points.append(pos.copy())

    return np.asarray(contact_points, dtype=float), np.asarray(detector_points, dtype=float)


def extract_meg_sensor_points(raw, sensor_length=SENSOR_LENGTH):
    """Backwards-compatible alias: returns contact points (meters)."""
    contact_points, _ = extract_meg_sensor_contact_and_detector_points(raw, sensor_length)
    return contact_points


def extract_meg_sensor_detector_points(raw):
    """Return MEG detector-center points (meters)."""
    _, detector_points = extract_meg_sensor_contact_and_detector_points(raw, SENSOR_LENGTH)
    return detector_points


def preprocess_point_cloud(pcd, voxel_size):
    """
    Preprocess point cloud for global registration.

    Performs voxel downsampling, normal estimation, and FPFH feature computation.

    Args:
        pcd: Open3D PointCloud object
        voxel_size: Voxel size for downsampling (mm)

    Returns:
        tuple: (downsampled_cloud, fpfh_features)
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """
    Execute RANSAC-based global registration using FPFH features.

    Args:
        source_down: Downsampled source point cloud
        target_down: Downsampled target point cloud
        source_fpfh: FPFH features for source
        target_fpfh: FPFH features for target
        voxel_size: Voxel size used for preprocessing

    Returns:
        Registration result with transformation matrix
    """
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return result


def _nearest_point_dist(a, B):
    """Calculate minimum distance from point a to any point in array B."""
    dists = np.linalg.norm(a - B, axis=1)
    return dists.min()


def head_to_helmet(source_cloud, landmarks=None, progress_callback=None):
    """
    Register LIDAR scan to helmet coordinates using detected landmarks.

    Args:
        source_cloud: Open3D PointCloud of the inside-helmet scan
        landmarks: Optional array of detected landmark positions. If None,
                   landmarks will be automatically detected.
        progress_callback: Optional callback(message, percent) for progress updates

    Returns:
        tuple: (X1 transformation matrix, detected landmarks, registration errors)
    """
    from sympy.utilities.iterables import multiset_permutations

    from .landmark_detection import find_landmarks

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    if progress_callback:
        progress_callback("Detecting helmet landmarks...", 10)

    # Detect landmarks if not provided
    if landmarks is None:
        landmarks = find_landmarks(source_cloud)
        if landmarks is None:
            return None, None, "Could not detect landmarks"

    red_points = landmarks

    # Create cloud of known helmet sticker positions
    rst_cloud = o3d.geometry.PointCloud()
    rst_cloud.points = o3d.utility.Vector3dVector(HELMET_STICKER_POSITIONS)
    rst_cloud.paint_uniform_color([0, 0, 1])

    # Create cloud of detected landmarks
    red_point_cloud = o3d.geometry.PointCloud()
    red_point_cloud.points = o3d.utility.Vector3dVector(red_points)

    if progress_callback:
        progress_callback("Finding optimal landmark correspondence...", 30)

    # Brute force permutations to find best correspondence
    landmark_indices = list(range(7))
    landmark_perms = np.asarray(list(multiset_permutations(landmark_indices)))
    landmark_perms = landmark_perms[:, 0 : len(red_points)]
    landmark_perms = np.unique(landmark_perms, axis=0)

    # Set up correspondence array
    corr = np.zeros([len(red_points), 2])
    corr[0 : len(red_points), 1] = np.arange(len(red_points))

    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    min_err = float("inf")
    X1 = None

    for i, perm in enumerate(landmark_perms):
        corr[:, 0] = perm
        trans = p2p.compute_transformation(
            rst_cloud, red_point_cloud, o3d.utility.Vector2iVector(corr)
        )
        test = copy.deepcopy(rst_cloud)
        test.transform(trans)
        err_trans = p2p.compute_rmse(test, red_point_cloud, o3d.utility.Vector2iVector(corr))
        if err_trans < min_err:
            min_err = err_trans
            X1 = trans

        if progress_callback and i % 100 == 0:
            progress = 30 + int(60 * i / len(landmark_perms))
            progress_callback(f"Testing permutation {i}/{len(landmark_perms)}...", progress)

    # Calculate per-landmark registration errors
    test = copy.deepcopy(rst_cloud)
    test.transform(X1)
    test_points = np.asarray(test.points)

    errors = []
    for lm in red_points:
        errors.append(_nearest_point_dist(lm, test_points))

    if progress_callback:
        progress_callback("Helmet registration complete", 100)

    return project_to_rigid_transform(X1), red_points, errors


def head_to_standard(target_cloud, anatomical_points):
    """
    Transform head scan to CTF-like anatomical coordinate system.

    Args:
        target_cloud: Open3D PointCloud of the outside-head scan
        anatomical_points: Array of 3 points: [R_preauricular, L_preauricular, nasion]
                          Can be indices into target_cloud or 3D coordinates

    Returns:
        tuple: (transformation matrix, transformed point cloud)
    """
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    # Handle both index-based and coordinate-based input
    if isinstance(anatomical_points[0], (int, np.integer)):
        # Points are indices
        R_aur = np.asarray(target_cloud.points)[anatomical_points[0]]
        L_aur = np.asarray(target_cloud.points)[anatomical_points[1]]
        nas = np.asarray(target_cloud.points)[anatomical_points[2]]
    else:
        # Points are coordinates - find nearest points in cloud
        R_aur = np.asarray(anatomical_points[0])
        L_aur = np.asarray(anatomical_points[1])
        nas = np.asarray(anatomical_points[2])
        # For coordinate-based, we need to create synthetic indices

    # Get position of CTF-style origin in original LIDAR data
    origin = R_aur + (R_aur - L_aur) / 2.0

    # Define anatomical points in 'standard' space
    standard = np.zeros([3, 3])
    # Right pre-auricular on -ve y-axis
    standard[0] = [0, -np.linalg.norm(R_aur - L_aur) / 2.0, 0]
    # Left pre-auricular on +ve y-axis
    standard[1] = [0, np.linalg.norm(R_aur - L_aur) / 2.0, 0]
    # Nasion on x-axis
    standard[2] = [np.linalg.norm(origin - nas), 0, 0]

    # Create cloud of anatomical points from input
    anat_cloud = o3d.geometry.PointCloud()
    anat_cloud.points = o3d.utility.Vector3dVector([R_aur, L_aur, nas])

    # Create cloud of standard positions
    standard_cloud = o3d.geometry.PointCloud()
    standard_cloud.points = o3d.utility.Vector3dVector(standard)

    # Define correspondence
    corr = np.array([[0, 0], [1, 1], [2, 2]])

    # Calculate transform
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    trans_init = p2p.compute_transformation(
        anat_cloud, standard_cloud, o3d.utility.Vector2iVector(corr)
    )

    trans_init = project_to_rigid_transform(trans_init)
    transformed_cloud = copy.deepcopy(target_cloud)
    transformed_cloud.transform(trans_init)

    return trans_init, transformed_cloud


def head_to_head(target_cloud, source_cloud, progress_callback=None):
    """
    Register helmet scan to anatomically-oriented head scan using ICP.

    Args:
        target_cloud: Open3D PointCloud of the head scan (already in standard coords)
        source_cloud: Open3D PointCloud of the helmet scan
        progress_callback: Optional callback(message, percent) for progress updates

    Returns:
        tuple: (X2 transformation matrix, ICP fitness score)
    """
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    if progress_callback:
        progress_callback("Computing global registration...", 10)

    # Global registration
    voxel_size = 2
    source_down, source_fpfh = preprocess_point_cloud(source_cloud, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target_cloud, voxel_size)
    result_global = execute_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size
    )
    trans_init = result_global.transformation

    if progress_callback:
        progress_callback("Refining with ICP...", 50)

    # Refine registration with ICP
    threshold = 2.00
    result_icp = o3d.pipelines.registration.registration_icp(
        source_cloud,
        target_cloud,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=8000),
    )

    if progress_callback:
        progress_callback("Head-to-head registration complete", 100)

    return project_to_rigid_transform(result_icp.transformation), result_icp.fitness


def head_to_mri(target_cloud, source_cloud, progress_callback=None):
    """
    Register MRI scalp surface to head scan using ICP.

    Args:
        target_cloud: Open3D PointCloud of the head scan (in standard coords)
        source_cloud: Open3D PointCloud of the MRI scalp surface
        progress_callback: Optional callback(message, percent) for progress updates

    Returns:
        tuple: (transformed MRI cloud, X3 transformation matrix, ICP fitness score)
    """
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    if progress_callback:
        progress_callback("Cropping MRI surface...", 5)

    # Crop MRI to avoid edge effects
    source_crop = copy.deepcopy(source_cloud)
    points = np.asarray(source_crop.points)
    y_threshold = 0.0
    source_crop = source_crop.select_by_index(np.where(points[:, 1] > y_threshold)[0])
    points = np.asarray(source_crop.points)
    z_threshold = -100.0
    source_crop = source_crop.select_by_index(np.where(points[:, 2] > z_threshold)[0])

    if progress_callback:
        progress_callback("Computing global registration...", 20)

    # Global registration
    voxel_size = 2
    source_down, source_fpfh = preprocess_point_cloud(source_crop, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target_cloud, voxel_size)
    result_global = execute_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size
    )
    trans_init = result_global.transformation

    if progress_callback:
        progress_callback("Refining with ICP...", 60)

    # Refine registration with ICP
    threshold = 2.00
    result_icp = o3d.pipelines.registration.registration_icp(
        source_crop,
        target_cloud,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=8000),
    )

    # Apply transform to full (uncropped) source cloud
    rigid = project_to_rigid_transform(result_icp.transformation)
    transformed_cloud = copy.deepcopy(source_cloud)
    transformed_cloud.transform(rigid)

    if progress_callback:
        progress_callback("MRI registration complete", 100)

    return transformed_cloud, rigid, result_icp.fitness


def check_headpoints(mri_cloud, meg_data_path, X21):
    """
    Validate registration by checking sensor-to-scalp distances.

    Args:
        mri_cloud: Open3D PointCloud of the MRI scalp
        meg_data_path: Path to MEG data file
        X21: Combined device-to-head transformation matrix

    Returns:
        tuple: (sensor_cloud, median_distance, distances)
    """
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    X21 = project_to_rigid_transform(X21)
    raw = mne.io.read_raw_fif(meg_data_path, verbose="error", preload=False)

    # Extract MEG sensor positions (contact points), convert m -> mm
    head_points = extract_meg_sensor_points(raw) * 1000.0
    if head_points.size == 0:
        raise ValueError("No MEG sensor points found in FIF file.")

    # Create point cloud
    head_point_cloud = o3d.geometry.PointCloud()
    head_point_cloud.points = o3d.utility.Vector3dVector(head_points)
    head_point_cloud.paint_uniform_color([0, 0, 1])
    head_point_cloud.transform(X21)

    # Calculate distances using a KD-tree (much faster than brute force)
    from scipy.spatial import cKDTree

    mri_points = np.asarray(mri_cloud.points)
    head_points = np.asarray(head_point_cloud.points)
    tree = cKDTree(mri_points)
    distances, _ = tree.query(head_points, k=1, workers=-1)
    median_dist = np.median(distances)

    return head_point_cloud, median_dist, distances


def write_output(meg_data_path, X21, X3):
    """
    Write registration results to MEG data file.

    Args:
        meg_data_path: Path to MEG data file
        X21: Combined device-to-head transformation matrix
        X3: MRI-to-head transformation matrix

    Returns:
        str: Path to the output trans file
    """
    raw = mne.io.read_raw_fif(meg_data_path, verbose="error", preload=True)
    X21 = project_to_rigid_transform(X21)
    X3 = project_to_rigid_transform(X3)

    # Create dev_head transform
    dev_head_t = mne.transforms.Transform("meg", "head", trans=None)
    dev_head_t["trans"] = X21.copy()
    # Convert mm to meters for MNE
    dev_head_t["trans"][0:3, 3] = np.divide(dev_head_t["trans"][0:3, 3], 1000)
    raw.info.update(dev_head_t=dev_head_t)

    # Generate head points from MEG sensor locations (contact points)
    head_points = extract_meg_sensor_points(raw)
    if head_points.size == 0:
        raise ValueError("No MEG sensor points found in FIF file.")

    # Transform to head reference frame
    head_point_cloud = o3d.geometry.PointCloud()
    head_point_cloud.points = o3d.utility.Vector3dVector(head_points)
    head_point_cloud.transform(dev_head_t["trans"])
    head_points = np.asarray(head_point_cloud.points)

    # Add digitization points
    montage = mne.channels.make_dig_montage(hsp=head_points, coord_frame="head")
    raw.set_montage(montage)
    raw.save(meg_data_path, overwrite=True)

    # Write separate trans file for MRI->head transform
    mri_head_t = mne.transforms.Transform("mri", "head", trans=None)
    mri_head_t["trans"] = X3.copy()
    mri_head_t["trans"][0:3, 3] = np.divide(mri_head_t["trans"][0:3, 3], 1000)

    # Generate output filename
    if ".fif" in meg_data_path:
        outfile = meg_data_path.rsplit(".fif", 1)[0] + "_trans.fif"
    else:
        outfile = meg_data_path + "_trans.fif"

    mne.write_trans(outfile, mri_head_t, overwrite=True)
    return outfile
