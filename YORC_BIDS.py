#!/usr/bin/env python3

'''
Author:
R Aveyard
2025-10-24

------------------------------------------------------------------------
Copyright (c) The University of York.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
3. Neither the name of the University nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE UNIVERSITY AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED.  IN NO EVENT SHALL THE UNIVERSITY OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
SUCH DAMAGE.

------------------------------------------------------------------------
'''
# Script to translate MEG co-registrations from older MNE pipelines
# in the form of _trans.fif files and transforms in raw MEG data
# into a format that can be interpreted by the MNE BIDS analysis pipeline.
# Approximate fiducial positions have been estimated on the freesurfer
# fsaverage head, and are then translated to the individual BIDs T1 and
# written to the associated json file, then transformed to the MEG data
# 'head' coordinate system as used in MNE python and written in the
# metatdata of OPM fif files. The transform between mri and meg systems
# can then derived from these perfectly corresponding coordinates
# in the MNE BIDS analysis pipeline.

import numpy as np
import copy
import argparse

# Approximate distance from Rubidium chamber to external tip of
# sensor housing that makes contact with head
sensor_length = 6e-3


def make_bids_json_fids(xfm, trans):
    # Start with some approximate fiducial positions on the FSaverage head
    # and use the Freesurfer-generated talairach.xfm to transform them to
    # MRI subject space. Then use the YORC-generated transform to get them
    # to the MNE participant 'head' space.

    import open3d as o3d
    import mne
    from mne.transforms import _read_fs_xfm

    # Some approximate locations for nasion, LPA, RPA in FSaverage space
    fsavg_fids = [[0.9, 80.0, -45.5], [78., -27., -61.5], [-78., -27., -61.5]]
    # Make fid cloud
    fid_cloud = o3d.geometry.PointCloud()
    fid_cloud.points = o3d.utility.Vector3dVector(fsavg_fids)
    # Read in talairach xfm from Freesurfer
    tal_trans = mne.transforms._read_fs_xfm(xfm)
    # Invert so that we can transform estimated
    # fiducials from Talairach into participant T1
    ital_trans = np.linalg.inv(tal_trans[0])
    # Apply transform to get template fids
    # in T1 coordinates
    ifid_cloud = copy.deepcopy(fid_cloud)
    ifid_cloud.transform(ital_trans)
    mri_fids = np.asarray(ifid_cloud.points)

    # Transform from T1 to standard/CTF
    X3 = mne.read_trans(trans)
    # transform in fif file is in m, here we are in mm, so
    # need to scale the translation part of the transform
    X3['trans'][0:3, 3] = np.multiply(X3['trans'][0:3, 3], 1000.)
    ctf_cloud = copy.deepcopy(ifid_cloud)
    ctf_cloud.transform(X3['trans'])
    meg_fids = np.asarray(ctf_cloud.points)
    # Now alignment is done, convert coordinates to mm for fif
    meg_fids = np.divide(meg_fids, 1000.)

    return meg_fids, mri_fids


def write_ouput(datafile, meg_fids):
    import open3d as o3d
    import mne

    # Write the results to file
    raw = mne.io.read_raw_fif(datafile, 'default', preload=True)
    dev_head_t = raw.info['dev_head_t']

    # Generate head points from sensor locations to use as digitization points
    head_points = []
    for chan in raw.info['chs']:
        head_points.append([chan['loc'][0] - chan['loc'][9] * sensor_length,
                            chan['loc'][1] - chan['loc'][10] * sensor_length,
                            chan['loc'][2] - chan['loc'][11] * sensor_length])
    head_points = np.array(head_points)

    # Make dig points into cloud
    head_point_cloud = o3d.geometry.PointCloud()
    head_point_cloud.points = o3d.utility.Vector3dVector(head_points)
    # Transform to head reference frame
    head_point_cloud.transform(dev_head_t['trans'])
    head_points = np.asarray(head_point_cloud.points)

    # Add digitization points generated from sensor positions
    montage = mne.channels.make_dig_montage(hsp=head_points,
                                            nasion=meg_fids[0],
                                            lpa=meg_fids[1],
                                            rpa=meg_fids[2],
                                            coord_frame='head')
    raw.set_montage(montage)
    raw.save(datafile, overwrite=True)

    return datafile


def write_fids(mri_fids, T1):
    import json
    import nibabel as nib
    from mne.transforms import apply_trans

    # Get the mm to vox transfrom from the T1 nifti
    img = nib.load(T1)
    t1_meta = img.header
    affine = t1_meta.get_sform()
    # Apply the transform to get the mri-space fiducial positions
    # in voxels
    inv_affine = np.linalg.inv(affine)
    mri_fids_vx = np.round(apply_trans(inv_affine, mri_fids))
    # Convert numpy to python ints for json handling
    mri_fids_vx = getattr(mri_fids_vx, "tolist", lambda: mri_fids_vx)()
    # Get the content of the existing bids T1 json file
    json_file = T1.split('.')[0] + '.json'
    with open(json_file) as f:
        t1_json_dat = json.load(f)
    # Add in the fiducial points
    t1_json_dat['AnatomicalLandmarkCoordinates'] = {"NAS": [mri_fids_vx[0][0],
                                                            mri_fids_vx[0][1],
                                                            mri_fids_vx[0][2]],
                                                    "LPA": [mri_fids_vx[1][0],
                                                            mri_fids_vx[1][1],
                                                            mri_fids_vx[1][2]],
                                                    "RPA": [mri_fids_vx[2][0],
                                                            mri_fids_vx[1][1],
                                                            mri_fids_vx[2][2]]}

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(t1_json_dat, f, ensure_ascii=False, indent=4)

    return json_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-xfm", "--talairach_transform", help="Transform between T1w and FS average space generated by FreeSurfer", required=True)
    parser.add_argument("-T1", "--subject_T1", help="T1 weighted MRI image as used in BIDS", required=True)
    parser.add_argument("-tr", "--trans_file", help="MNE _trans.fif file containing MRI->head transform", required=True)
    parser.add_argument("-m", "--megdata", help="MEG data to generate the transform into", nargs='+', required=True)

    args = parser.parse_args()

    xfm = args.talairach_transform
    T1 = args.subject_T1
    trans = args.trans_file

    [meg_fids, mri_fids] = make_bids_json_fids(xfm, trans)

    print("\nMatching fiducials written to:")
    for MEG_data in args.megdata:
        outfile = write_ouput(MEG_data, meg_fids)
        print(outfile)

    mrifidfile = write_fids(mri_fids, T1)
    print(mrifidfile)


if __name__ == '__main__':
    main()
