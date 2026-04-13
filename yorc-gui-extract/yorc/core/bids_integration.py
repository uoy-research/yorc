"""BIDS-facing fiducial export helpers.

These functions port the legacy YORC_BIDS workflow into the packaged core so
the GUI can export matching MRI/head fiducials without depending on the old
top-level script.
"""

from __future__ import annotations

import json
from pathlib import Path

import mne
import nibabel as nib
import numpy as np
from mne.transforms import apply_trans

from .registration import SENSOR_LENGTH, extract_meg_sensor_points, project_to_rigid_transform


FSAVERAGE_FIDUCIALS_MM = np.array(
    [
        [0.9, 80.0, -45.5],
        [78.0, -27.0, -61.5],
        [-78.0, -27.0, -61.5],
    ],
    dtype=float,
)


def _strip_nii_suffix(path: str) -> str:
    if path.endswith(".nii.gz"):
        return path[:-7]
    if path.endswith(".nii"):
        return path[:-4]
    return str(Path(path).with_suffix(""))


def _as_mm_transform(transform) -> np.ndarray:
    if isinstance(transform, dict) and "trans" in transform:
        matrix = np.asarray(transform["trans"], dtype=float).copy()
        matrix[0:3, 3] *= 1000.0
        return project_to_rigid_transform(matrix)

    matrix = np.asarray(transform, dtype=float).copy()
    if matrix.shape != (4, 4):
        raise ValueError("Transform must be a 4x4 matrix or an MNE transform with a 'trans' key.")
    return project_to_rigid_transform(matrix)


def _as_mne_transform_mm_to_m(transform, from_frame: str, to_frame: str) -> mne.transforms.Transform:
    matrix_mm = _as_mm_transform(transform)
    matrix_m = matrix_mm.copy()
    matrix_m[0:3, 3] /= 1000.0
    return mne.transforms.Transform(from_frame, to_frame, trans=matrix_m)


def make_bids_json_fids(talairach_xfm_path: str, mri_to_head_transform) -> tuple[np.ndarray, np.ndarray]:
    """Create corresponding head-space and MRI-space fiducials.

    Args:
        talairach_xfm_path: FreeSurfer talairach.xfm path.
        mri_to_head_transform: MRI-to-head transform as either a 4x4 mm matrix
            or an MNE transform.

    Returns:
        Tuple of ``(meg_fids_m, mri_fids_mm)``.
    """
    tal_xfm = mne.transforms._read_fs_xfm(talairach_xfm_path)[0]
    inv_tal_xfm = np.linalg.inv(tal_xfm)

    fsavg_mm = FSAVERAGE_FIDUCIALS_MM
    fsavg_h = np.c_[fsavg_mm, np.ones(len(fsavg_mm))]
    mri_fids_mm = (inv_tal_xfm @ fsavg_h.T).T[:, :3]

    mri_to_head_mm = _as_mm_transform(mri_to_head_transform)
    mri_h = np.c_[mri_fids_mm, np.ones(len(mri_fids_mm))]
    meg_fids_mm = (mri_to_head_mm @ mri_h.T).T[:, :3]
    meg_fids_m = meg_fids_mm / 1000.0

    return meg_fids_m, mri_fids_mm


def write_meg_bids_fiducials(
    meg_data_path: str,
    meg_fids_m: np.ndarray,
    dev_head_transform=None,
    sensor_length: float = SENSOR_LENGTH,
) -> str:
    """Write BIDS-compatible fiducials into an MEG FIF file.

    Args:
        meg_data_path: MEG FIF path to update in place.
        meg_fids_m: Fiducials in head coordinates, meters, ordered as
            ``[NAS, LPA, RPA]``.
        dev_head_transform: Optional device-to-head transform to apply before
            generating head-shape points. Can be a 4x4 mm matrix or MNE
            transform.
        sensor_length: Contact-point offset in meters.

    Returns:
        The MEG FIF path written.
    """
    raw = mne.io.read_raw_fif(meg_data_path, verbose="error", preload=True)

    if dev_head_transform is not None:
        raw.info.update(
            dev_head_t=_as_mne_transform_mm_to_m(dev_head_transform, "meg", "head")
        )

    dev_head_t = raw.info.get("dev_head_t")
    if dev_head_t is None:
        raise ValueError("MEG file has no dev_head_t; provide a device-to-head transform.")

    head_points = extract_meg_sensor_points(raw, sensor_length=sensor_length)
    if head_points.size == 0:
        raise ValueError("No MEG sensor points found in FIF file.")

    head_points_h = np.c_[head_points, np.ones(len(head_points))]
    head_points_head = (np.asarray(dev_head_t["trans"], dtype=float) @ head_points_h.T).T[:, :3]

    montage = mne.channels.make_dig_montage(
        hsp=head_points_head,
        nasion=np.asarray(meg_fids_m[0], dtype=float),
        lpa=np.asarray(meg_fids_m[1], dtype=float),
        rpa=np.asarray(meg_fids_m[2], dtype=float),
        coord_frame="head",
    )
    raw.set_montage(montage)
    raw.save(meg_data_path, overwrite=True)

    return meg_data_path


def write_mri_bids_fiducials(mri_fids_mm: np.ndarray, t1_path: str) -> str:
    """Write MRI fiducials into the BIDS T1 JSON sidecar.

    Args:
        mri_fids_mm: Fiducials in MRI coordinates, millimeters, ordered as
            ``[NAS, LPA, RPA]``.
        t1_path: Path to subject T1 NIfTI.

    Returns:
        The JSON sidecar path written.
    """
    img = nib.load(t1_path)
    affine = img.header.get_sform()
    inv_affine = np.linalg.inv(affine)
    mri_fids_vox = np.round(apply_trans(inv_affine, np.asarray(mri_fids_mm, dtype=float))).astype(int)

    json_path = _strip_nii_suffix(t1_path) + ".json"
    if Path(json_path).exists():
        with open(json_path, encoding="utf-8") as handle:
            t1_json_data = json.load(handle)
    else:
        t1_json_data = {}

    t1_json_data["AnatomicalLandmarkCoordinates"] = {
        "NAS": mri_fids_vox[0].tolist(),
        "LPA": mri_fids_vox[1].tolist(),
        "RPA": mri_fids_vox[2].tolist(),
    }

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(t1_json_data, handle, ensure_ascii=False, indent=4)

    return json_path


def export_bids_fiducials(
    meg_data_path: str,
    talairach_xfm_path: str,
    t1_path: str,
    dev_head_transform,
    mri_to_head_transform,
) -> tuple[str, str]:
    """Write matching BIDS fiducials to both MEG and MRI metadata."""
    meg_fids_m, mri_fids_mm = make_bids_json_fids(talairach_xfm_path, mri_to_head_transform)
    meg_out = write_meg_bids_fiducials(meg_data_path, meg_fids_m, dev_head_transform)
    json_out = write_mri_bids_fiducials(mri_fids_mm, t1_path)
    return meg_out, json_out