"""
Core registration and processing functions for YORC.
"""

from .bids_integration import export_bids_fiducials, make_bids_json_fids, write_mri_bids_fiducials
from .fiducial_estimation import estimate_fiducials_from_mri
from .io_utils import load_meg_data, load_mesh, load_mesh_for_display, save_meg_data
from .landmark_detection import find_landmarks
from .registration import (
    check_headpoints,
    execute_global_registration,
    head_to_head,
    head_to_helmet,
    head_to_mri,
    head_to_standard,
    preprocess_point_cloud,
    write_output,
)

__all__ = [
    "head_to_helmet",
    "head_to_standard",
    "head_to_head",
    "head_to_mri",
    "check_headpoints",
    "write_output",
    "preprocess_point_cloud",
    "execute_global_registration",
    "find_landmarks",
    "estimate_fiducials_from_mri",
    "make_bids_json_fids",
    "write_mri_bids_fiducials",
    "export_bids_fiducials",
    "load_mesh",
    "load_mesh_for_display",
    "load_meg_data",
    "save_meg_data",
]
