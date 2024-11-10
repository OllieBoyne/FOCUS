"""To avoid conflicts with opening Open3D in the main thread (conflicts with PyMeshLab),
interact with Open3D entirely within this script"""

import open3d
import trimesh
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('pts_obj', type=str)
parser.add_argument('neighbours', type=int)
parser.add_argument('std_ratio', type=float)

def filter_by_outliers(pts_obj, neighbours=20, std_ratio=2.0):
    """Remove statistical outliers from a point cloud."""

    pts_orig = trimesh.load(pts_obj).vertices
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pts_orig)
    pcd, ind = pcd.remove_statistical_outlier(
        nb_neighbors=neighbours, std_ratio=std_ratio
    )

    return np.isin(np.arange(len(pts_orig)), ind)


if __name__ == '__main__':
    args = parser.parse_args()
    mask = filter_by_outliers(args.pts_obj, args.neighbours, args.std_ratio)
    print(''.join(['1' if m else '0' for m in mask]), end='')