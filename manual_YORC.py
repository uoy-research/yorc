#!/usr/bin/env python3

'''
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

import numpy as np
import copy
from sys import exit, argv
import argparse
import mne
import os
import pathlib

# Approximate distance from Rubidium chamber to external tip of
# sensor housing that makes contact with head

sensor_length = 6e-3


def usage():
    print(f"""Usage: {os.path.basename(__file__)} --outsidemesh --insidemesh --mriscalp --megdata

    -om, --outsidemesh, LIDAR scan of head outside the MEG Helmet)
    -im, --insidemesh,  LIDAR scan of head inside the MEG Helmet)
    -s,  --mriscalp,    MRI scalp surface from Freesurfer)
    -m,  --megdata,     MEG data file(s) to generate the transform into
    (multiple files space-seperated))
    """)

    exit(0)


def pick_points(pcd):
    import open3d as o3d
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(width=1000, height=1000)
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    return vis.get_picked_points()

def vis_controls():
    print("\nControls:")
    print("  Shift + Left-Click:  Add a point.")
    print("  Shift + Right-Click: Remove a point.")
    print("  Hit q when all points are made.\n")

def head_to_helmet(source_in, landmarks):
    print("\nSetting up...")
    import open3d as o3d
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    # Align manually selected points from a LIDAR scan with
    # known landmark coordinates

    # Open stl files and convert to 100k point clouds so that we can use point-wise registration
    print("Converting STL to PointCloud...")
    mesh = o3d.io.read_triangle_mesh(source_in)
    source_cloud = mesh.sample_points_poisson_disk(100000)

    # Known positions of stickers in helmet reference frame
    rsticker_pillars = np.zeros([7, 3])
    rsticker_pillars[0] = [102.325, 0.221, 16.345]
    rsticker_pillars[1] = [92.079, 66.226, -27.207]
    rsticker_pillars[2] = [67.431, 113.778, -7.799]
    rsticker_pillars[3] = [-0.117, 138.956, -5.576]
    rsticker_pillars[4] = [-67.431, 113.778, -7.799]
    rsticker_pillars[5] = [-92.079, 66.226, -27.207]
    rsticker_pillars[6] = [-102.325, 0.221, 16.345]

    landmarks = np.asarray(landmarks)-1

    # Make into cloud
    rst_cloud = o3d.geometry.PointCloud()
    rst_cloud.points = o3d.utility.Vector3dVector(rsticker_pillars)

    # Make the true pillar landmarks blue so we can see them relative to red cloud
    rst_cloud.paint_uniform_color([0, 0, 1])

    # Get anchor points
    vis_controls()
    print("Select the Helmet Labels from left to right.")

    # Display visualiser
    red_points = pick_points(source_cloud)

    # Define which sticker-pillar points correspond to which selected points
    corr = np.zeros((len(landmarks), 2))
    for ii, lm in enumerate(landmarks):
        corr[ii, 0] = lm  # So users don't have to deal with zero-indexing
    corr[:, 1] = red_points

    # Calculate transform based on anchor points alone
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    trans_init = p2p.compute_transformation(rst_cloud, source_cloud,
                                            o3d.utility.Vector2iVector(corr))
    X1 = trans_init
    # Have a look at anchor-based registration
    test = copy.deepcopy(rst_cloud)
    test.transform(trans_init)
    # Print errors on anchor-point registration
    print("\nLandmark co-registration Errors:")
    for ii, lm in enumerate(landmarks):
        print("%.3f mm " % np.linalg.norm(test.points[lm] - source_cloud.points[int(corr[ii][1])]))

    return X1



def head_to_standard(target_in):
    import open3d as o3d
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    # Transform head-outside into CTF-like coordinate system
    # We don't really need to have this intermediate coordinate space.
    # In principle, we could just use MEG device and MRI spaces but
    # this might prove useful during analysis as it ape the way that
    # legacy systems dealt with transforms.
    mesh = o3d.io.read_triangle_mesh(target_in)
    print("\nSetting up next step...")
    target_cloud = mesh.sample_points_poisson_disk(100000)

    # Get anchor points
    print("\nIn order, please select: ")
    print("  - right pre-auricular")
    print("  - left pre-auricular")
    print("  - nasion\n")

    anat_points = pick_points(target_cloud)
    R_aur = target_cloud.points[anat_points[0]]
    nas = target_cloud.points[anat_points[1]]
    L_aur = target_cloud.points[anat_points[2]]

    # Get position of CTF-style origin in original LIDAR data
    origin = R_aur + (R_aur - L_aur) / 2.

    # Define anatomical points in 'standard' space to align with
    standard = np.zeros([3, 3])

    # right pre-auricular on -ve y-axis
    standard[0] = [0, -np.linalg.norm(R_aur - L_aur)/2., 0]

    # left pre-auricular on +ve y-axis
    standard[1] = [0, np.linalg.norm(R_aur - L_aur)/2., 0]

    # Nasion on x-axis
    standard[2] = [np.linalg.norm(origin-nas), 0, 0]

    # Make into cloud
    standard_cloud = o3d.geometry.PointCloud()
    standard_cloud.points = o3d.utility.Vector3dVector(standard)

    # Define which LIDAR points correspond to which standard points
    corr = np.zeros((3, 2))
    corr[:, 0] = anat_points
    corr[:, 1] = [0, 1, 2]

    # Calculate transform and apply
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    trans_init = p2p.compute_transformation(target_cloud, standard_cloud,
                                            o3d.utility.Vector2iVector(corr))
    target_cloud.transform(trans_init)

    return(trans_init, target_cloud)


def head_to_head(standard_trans, target_cloud, source_in):
    import open3d as o3d
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    print("\nSetting up next step...")
    mesh = o3d.io.read_triangle_mesh(source_in)
    source_cloud = mesh.sample_points_poisson_disk(100000)

    # Get anchor points
    print("\nIn order, please select: ")
    print("  - Right eye")
    print("  - Left eye")
    print("  - Tip of nose\n")
    source_points = pick_points(source_cloud)

    print("\nIn order, please select: ")
    print("  - Right eye")
    print("  - Left eye")
    print("  - Tip of nose\n")
    target_points = pick_points(target_cloud)

    # create array of corresponding points
    corr = np.zeros((len(source_points), 2))
    corr[:, 0] = source_points
    corr[:, 1] = target_points[0:len(source_points)]

    # Calculate transform based on anchor points alone
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    trans_init = p2p.compute_transformation(source_cloud, target_cloud,
                                            o3d.utility.Vector2iVector(corr))

    # Crop both clouds around the target points so that we don't have any
    # contributions to the refinement from to far away from the face.
    source_COM = np.zeros(3)
    target_COM = np.zeros(3)
    for i in source_points:
        source_COM += source_cloud.points[i]
    source_COM = np.divide(source_COM, len(source_points))
    for i in target_points:
        target_COM += target_cloud.points[i]
    target_COM = np.divide(target_COM, len(target_points))
    radius = 50  # mm
    points = np.asarray(target_cloud.points)

    # Calculate distances to center, set new points
    distances = np.linalg.norm(points - target_COM, axis=1)
    target_crop = o3d.geometry.PointCloud()
    target_crop.points = o3d.utility.Vector3dVector(points[distances <= radius])
    radius = 50  # mm
    points = np.asarray(source_cloud.points)

    # Calculate distances to center, set new points
    distances = np.linalg.norm(points - source_COM, axis=1)
    source_crop = o3d.geometry.PointCloud()
    source_crop.points = o3d.utility.Vector3dVector(points[distances <= radius])

    # Remove some points that aren't part of the contiguous face surface
    # Find clusters contiguous with the anchor points to do that.
    labels = np.array(
            source_crop.cluster_dbscan(eps=5.0, min_points=10, print_progress=True))
    labels = list(labels)
    most_common_label = max(labels, key=labels.count)
    pts = np.asarray(source_crop.points)
    source_crop.points = o3d.utility.Vector3dVector(pts[np.where(labels == most_common_label)[0]])

    # Refine registration with ICP
    threshold = 2.00
    result_icp = o3d.pipelines.registration.registration_icp(
        source_crop, target_crop, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=8000))
    source_crop_cop = copy.deepcopy(source_crop)
    source_crop_cop.transform(result_icp.transformation)
    print("\nPreview...\n - Press q to contiunue")
    o3d.visualization.draw_geometries([source_crop_cop, target_crop], width=1000, height=1000)
    source_cloudX2 = copy.deepcopy(source_cloud)
    source_cloudX2.transform(result_icp.transformation)
    print("Preview...\n - Press q to contiunue")
    o3d.visualization.draw_geometries([source_cloudX2, target_cloud], width=1000, height=1000)
    X2 = result_icp.transformation
    return X2


def head_to_mri(target_cloud, source_in):
    import open3d as o3d
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    print("\nSetting up MRI surface...")
    mesh = o3d.io.read_triangle_mesh(source_in)
    source_cloud = mesh.sample_points_poisson_disk(100000)

    # Get anchor points

    print("\nIn order, please select: ")
    print("  - Right eye")
    print("  - Left eye")
    print("  - Naison\n")
    source_points = pick_points(source_cloud)
    print("\nIn order, please select: ")
    print("  - Right eye")
    print("  - Left eye")
    print("  - Naison\n")
    target_points = pick_points(target_cloud)

    # Create array of corresponding points
    corr = np.zeros((len(source_points), 2))
    corr[:, 0] = source_points
    corr[:, 1] = target_points[0:len(source_points)]

    # Calculate transform based on anchor points alone
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    trans_init = p2p.compute_transformation(source_cloud, target_cloud,
                                            o3d.utility.Vector2iVector(corr))

    # Crop both clouds around the target points so that we don't have any
    # contributions to the refinement from to far away from the face.
    source_COM = np.zeros(3)
    target_COM = np.zeros(3)
    for i in source_points:
        source_COM += source_cloud.points[i]
    source_COM = np.divide(source_COM, len(source_points))
    for i in target_points:
        target_COM += target_cloud.points[i]
    target_COM = np.divide(target_COM, len(target_points))
    radius = 50  # mm
    points = np.asarray(target_cloud.points)

    # Calculate distances to center, set new points
    distances = np.linalg.norm(points - target_COM, axis=1)
    target_crop = o3d.geometry.PointCloud()
    target_crop.points = o3d.utility.Vector3dVector(points[distances <= radius])
    radius = 50  # mm
    points = np.asarray(source_cloud.points)

    # Calculate distances to center, set new points
    distances = np.linalg.norm(points - source_COM, axis=1)
    source_crop = o3d.geometry.PointCloud()
    source_crop.points = o3d.utility.Vector3dVector(points[distances <= radius])

    # Refine registration with ICP
    threshold = 2.00
    result_icp = o3d.pipelines.registration.registration_icp(
        source_crop, target_crop, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=8000))
    source_crop_cop = copy.deepcopy(source_crop)
    source_crop_cop.transform(result_icp.transformation)
    print("\nPreview\nPress q to continue")
    o3d.visualization.draw_geometries([source_crop_cop, target_crop], width=1000, height=1000)
    source_cloudX3 = copy.deepcopy(source_cloud)
    source_cloudX3.transform(result_icp.transformation)
    print("\nPreview\nPress q to continue")
    o3d.visualization.draw_geometries([source_cloudX3, target_cloud], width=1000, height=1000)
    X3 = copy.deepcopy(result_icp.transformation)
    return(source_cloudX3, X3)


def check_headpoints(mri_cloud, datafile, X21):
    import open3d as o3d
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    raw = mne.io.read_raw_fif(datafile, 'default', preload=False)

    # Check results with sensor position based 'digitization points'
    # Load headpoints
    head_points = []
    for chan in raw.info['chs']:
        head_points.append([chan['loc'][0]-chan['loc'][9]*sensor_length,
                            chan['loc'][1]-chan['loc'][10]*sensor_length,
                            chan['loc'][2]-chan['loc'][11]*sensor_length])
    head_points = np.array(head_points)
    head_points = head_points*1000

    # Make into cloud
    head_point_cloud = o3d.geometry.PointCloud()
    head_point_cloud.points = o3d.utility.Vector3dVector(head_points)

    # Make the points blue
    head_point_cloud.paint_uniform_color([0, 0, 1])
    head_point_cloud.transform(X21)

    # Visual check
    print("\nPreview of sensor locations over structural scan")
    o3d.visualization.draw_geometries([mri_cloud, head_point_cloud], width=1000, height=1000)

    # Calculate distances between sensor points and scalp and report median value
    mins = []
    for hp in head_point_cloud.points:
        distances = np.linalg.norm(hp-mri_cloud.points, axis=1)
        mins.append(np.min(distances))
    mins = np.asarray(mins)
    print("Median sensor-scalp distance %.3f mm " % np.median(mins))
    return 0


def write_ouput(datafile, X21, X3):
    # Write the results to file
    raw = mne.io.read_raw_fif(datafile, 'default', preload=True)
    dev_head_t = mne.transforms.Transform("meg", "head", trans=None)
    dev_head_t['trans'] = X21.copy()

    # Metres for mne, mm in open3D
    # so translation elements of transform have to be re-scaled
    # but rotation is fine
    dev_head_t['trans'][0:3, 3] = np.divide(dev_head_t['trans'][0:3, 3], 1000)
    raw.info.update(dev_head_t=dev_head_t)
    raw.save(datafile, overwrite=True)

    # Write seperate trans file for MRI->head transform
    mri_head_t = mne.transforms.Transform("mri", "head", trans=None)
    mri_head_t['trans'] = X3.copy()
    mri_head_t['trans'][0:3, 3] = np.divide(mri_head_t['trans'][0:3, 3], 1000)
    outfile = datafile.split('.fif')[0] + '_trans.fif'
    mne.write_trans(outfile, mri_head_t, overwrite=True)
    return outfile

    def is_file(file):
        if not os.path.isfile(file):
            print(f'Error with {file}.')
            return false
        else:
            return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-om", "--outside_mesh", help="LIDAR scan of head outside the MEG Helmet", required = True)
    parser.add_argument("-im", "--inside_mesh", help="LIDAR scan of head inside the MEG Helmet", required=True)
    parser.add_argument("-s", "--mri_scalp", help="MRI scalp surface from Freesurfer", required=True)
    parser.add_argument("-m", "--megdata", help="MEG data to generate the transform into", nargs='+', required=True)
    parser.add_argument("-lm", "--landmarks", help="Landmarks to use for helmet registration", type=int, nargs='+')

    args = parser.parse_args()

    helmet_mesh = args.inside_mesh
    head_mesh = args.outside_mesh
    mri_cloud = args.mri_scalp

    # Check these files exist.
    for f in [helmet_mesh, head_mesh, mri_cloud]:
        if not os.path.isfile(f):
            print(f'Error opening {f}')
            exit(1)

    if args.landmarks is not None:
        landmarks = args.landmarks
    else:
        landmarks = [1, 2, 3, 4, 5, 6, 7]

    X1 = head_to_helmet(helmet_mesh, landmarks)
    [standard_trans, standard_head] = head_to_standard(head_mesh)
    X2 = head_to_head(standard_trans, standard_head, helmet_mesh)
    X21 = np.dot(X2, X1)
    [mri_cloud, X3] = head_to_mri(standard_head, mri_cloud)

    for MEG_data in args.megdata:
        check_headpoints(mri_cloud, MEG_data, X21)
        outfile = write_ouput(MEG_data, X21, X3)

    print("\nTransforms written to:")
    print(MEG_data)
    print(outfile)


if __name__ == '__main__':
    main()
