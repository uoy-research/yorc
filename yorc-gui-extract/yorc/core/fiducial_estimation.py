"""
Automatic fiducial estimation using MNE and FreeSurfer.

Provides automatic estimation of anatomical landmarks (LPA, nasion, RPA)
from FreeSurfer subject data.
"""

import mne
import numpy as np


def estimate_fiducials_from_mri(subjects_dir, subject):
    """
    Estimate anatomical fiducials from FreeSurfer subject data.

    Uses MNE's get_mni_fiducials() to estimate LPA, nasion, and RPA
    positions based on the subject's Talairach transform.

    Args:
        subjects_dir: Path to FreeSurfer subjects directory
        subject: Subject ID (name of subject folder)

    Returns:
        dict with keys 'lpa', 'nasion', 'rpa', each containing (x, y, z) in mm,
        or None if estimation fails
    """
    try:
        fids = mne.coreg.get_mni_fiducials(subject, subjects_dir=subjects_dir)

        # Extract fiducial positions
        # MNE returns in meters, we need millimeters
        fiducials = {}
        for fid in fids:
            ident = fid["ident"]
            pos = fid["r"] * 1000  # Convert to mm

            if ident == mne.io.constants.FIFF.FIFFV_POINT_LPA:
                fiducials["lpa"] = pos
            elif ident == mne.io.constants.FIFF.FIFFV_POINT_NASION:
                fiducials["nasion"] = pos
            elif ident == mne.io.constants.FIFF.FIFFV_POINT_RPA:
                fiducials["rpa"] = pos

        # Verify we got all three fiducials
        if len(fiducials) != 3:
            return None

        return fiducials

    except Exception as e:
        print(f"Fiducial estimation failed: {e}")
        return None


def transform_fiducials_to_lidar(fiducials_mri, mri_to_lidar_transform):
    """
    Transform MRI-space fiducials to LIDAR head scan space.

    Args:
        fiducials_mri: dict with 'lpa', 'nasion', 'rpa' keys in MRI space (mm)
        mri_to_lidar_transform: 4x4 transformation matrix from MRI to LIDAR space

    Returns:
        dict with transformed fiducial positions
    """
    import open3d as o3d

    # Create point cloud with fiducial positions
    fid_points = np.array(
        [
            fiducials_mri["rpa"],  # Order matches head_to_standard expectations
            fiducials_mri["lpa"],
            fiducials_mri["nasion"],
        ]
    )

    fid_cloud = o3d.geometry.PointCloud()
    fid_cloud.points = o3d.utility.Vector3dVector(fid_points)
    fid_cloud.transform(mri_to_lidar_transform)

    transformed_points = np.asarray(fid_cloud.points)

    return {
        "rpa": transformed_points[0],
        "lpa": transformed_points[1],
        "nasion": transformed_points[2],
    }


def fiducials_to_anatomical_array(fiducials):
    """
    Convert fiducials dict to array format expected by head_to_standard.

    Args:
        fiducials: dict with 'lpa', 'nasion', 'rpa' keys

    Returns:
        numpy array of shape (3, 3) with [R_preauricular, L_preauricular, nasion]
    """
    return np.array([fiducials["rpa"], fiducials["lpa"], fiducials["nasion"]])


def validate_fiducials(fiducials):
    """
    Perform basic validation of fiducial positions.

    Checks that:
    - LPA and RPA are roughly symmetric about the midline
    - Nasion is anterior to the pre-auricular points
    - Distances are anatomically plausible

    Args:
        fiducials: dict with 'lpa', 'nasion', 'rpa' keys in mm

    Returns:
        tuple: (is_valid, message)
    """
    lpa = np.array(fiducials["lpa"])
    rpa = np.array(fiducials["rpa"])
    nas = np.array(fiducials["nasion"])

    # Check inter-auricular distance (typically 120-180mm)
    ia_dist = np.linalg.norm(lpa - rpa)
    if ia_dist < 100 or ia_dist > 200:
        return False, f"Inter-auricular distance {ia_dist:.1f}mm outside typical range (100-200mm)"

    # Check nasion-to-midpoint distance (typically 80-120mm)
    midpoint = (lpa + rpa) / 2
    nas_dist = np.linalg.norm(nas - midpoint)
    if nas_dist < 60 or nas_dist > 150:
        return False, f"Nasion distance {nas_dist:.1f}mm outside typical range (60-150mm)"

    return True, "Fiducials appear valid"
