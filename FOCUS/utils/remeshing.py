import trimesh
import numpy as np
import pymeshlab  # NOTE: Importing this and open3D together cause some errors on M1 mac with poisson reconstruction.
import tempfile
from scipy.spatial import Delaunay
from time import perf_counter
import subprocess
import sys

import warnings
import os


def _setup_meshlab():
    """Needed for meshlab. May interfere with certain Open3D functionality, haven't tested"""
    if os.environ.get("KMP_DUPLICATE_LIB_OK", "True"):
        warnings.warn(
            "Turning off KMP_DUPLICATE_LIB_OK for Meshlab. May interfere with Open3D."
        )
        os.environ["KMP_DUPLICATE_LIB_OK"] = "False"

    # TODO: Check whether this is always needed. But setting this to False seems to fix the issue on Macbook M1.
    os.environ['OMP_NUM_THREADS'] = '1'

def add_floor(points):
    """Add a grid of points within the footprint of the point cloud to force the reconstruction to be watertight."""

    Xp, Yp, Zp = points.T

    N = 100
    X, Y = np.meshgrid(
        np.linspace(Xp.min(), Xp.max(), N), np.linspace(Yp.min(), Yp.max(), N)
    )
    floor_points = np.stack([X, Y, X * 0], axis=-1).reshape(-1, 3)

    points = np.concatenate([points, floor_points], axis=0)
    return points


def cutoff_above(points, z=0.1) -> (np.ndarray, np.ndarray):
    """Remove all points with a height greater than this z."""
    mask = points[:, 2] < z
    return points[mask], mask

def cutoff_below(points, z=0.0) -> (np.ndarray, np.ndarray):
    """Remove all points with a height less than this z."""
    mask = points[:, 2] > z
    return points[mask], mask

# def filter_outliers(pts, neighbours=100, std_ratio=0.5):
#     pcl = o3d.geometry.PointCloud()
#     pcl.points = o3d.utility.Vector3dVector(pts)
#     pcl, ind = pcl.remove_statistical_outlier(nb_neighbors=neighbours, std_ratio=std_ratio)
#     return np.asarray(pcl.points)


def pymeshlab_remove_outliers(
    points: np.ndarray, probability: float = 0.8, n_neighbours: int = 32
) -> (np.ndarray, np.ndarray):
    # TODO: switch to Open3D's remove_statistical_outlier - better implementation.
    mesh = pymeshlab.Mesh(vertex_matrix=points)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(mesh)

    ms.compute_selection_point_cloud_outliers(
        propthreshold=probability, knearest=n_neighbours
    )
    mask = np.invert(ms.current_mesh().vertex_selection_array())

    return points[mask], mask


def pymeshlab_poisson_reconstruct(points, normals=None, depth=8, iters:int = 8, samplespernode:float = 1.5,
                                  cgdepth:int = 5, pointweight:int = 4,
                                  quality: np.ndarray | None=None) -> trimesh.Trimesh:
    _setup_meshlab()

    mesh = pymeshlab.Mesh(vertex_matrix=points)

    if normals is not None:
        kwargs = {}

        if quality is not None:
            kwargs["v_scalar_array"] = np.clip(quality, a_min=1e-4, a_max=1.0)

        mesh = pymeshlab.Mesh(vertex_matrix=points, v_normals_matrix=normals, **kwargs)

    ms = pymeshlab.MeshSet()
    ms.add_mesh(mesh)
    if normals is None:
        ms.compute_normal_for_point_clouds()

    # Threads > 1 seems to cause occasional failures as it is non-deterministic...
    # TODO: Look at using Quality (confidence = True)
    ms.generate_surface_reconstruction_screened_poisson(
        preclean=True, threads=1, depth=depth, iters=iters, samplespernode=samplespernode,
        cgdepth=cgdepth, pointweight=pointweight, confidence=quality is not None
    )

    mesh = trimesh.Trimesh(vertices=ms.current_mesh().vertex_matrix(), faces=ms.current_mesh().face_matrix())

    return mesh


def poisson_recon_execute(points, normals, depth=6):
    """Run from PoissonRecon executable."""
    executable = "/Users/ollie/Documents/repos/phd-projects/FIND2D/misc/PoissonRecon/Bin/Linux/PoissonRecon"
    with tempfile.NamedTemporaryFile(suffix=".ply") as f:
        trimesh.Trimesh(vertices=points, vertex_normals=normals).export(f.name)
        args = [
            "--in",
            f.name,
            "--depth",
            str(depth),
            "--out",
            f.name,
            "--verbose",
            "--bType",
            "3",
        ]

        subprocess.run([executable] + args, check=True)
        return trimesh.load(f.name)


def delaunay_triangulation(points) -> trimesh.Trimesh:
    tri = Delaunay(points, furthest_site=True)
    return trimesh.Trimesh(vertices=points, faces=tri.simplices)


def remove_large_faces(mesh, face_area=0.01):
    """Remove edges that are too long."""
    face_areas = mesh.area_faces
    mesh = mesh.copy()
    mesh.update_faces(face_areas < face_area)
    return mesh


def crop_to_original_pointcloud(mesh: trimesh.Trimesh, original_points: np.ndarray, padding=0.01) -> trimesh.Trimesh:
    """Remove all faces that are not within a bbox containing original_points (+ padding)."""
    # Remove faces outside bounding box + padding
    min_bbox = original_points.min(axis=0) - padding
    max_bbox = original_points.max(axis=0) + padding
    reconstr_points = np.array(mesh.vertices)
    points_in_bbox = np.all(
        (reconstr_points >= min_bbox) & (reconstr_points <= max_bbox), axis=-1
    )
    faces_mask = points_in_bbox[mesh.faces].all(axis=1)
    mesh = mesh.copy()
    mesh.update_faces(faces_mask)
    return mesh

def keep_pointcloud_island(mesh: trimesh.Trimesh, points: np.ndarray) -> trimesh.Trimesh:
    """Given a mesh and a point cloud, keep only the island which the most points are nearest to."""
    return min(mesh.split(only_watertight=False), key=lambda x: trimesh.proximity.ProximityQuery(x).vertex(points)[0].mean())

def open3d_remove_outliers(points: np.ndarray, neighbours=20, std_ratio=2.0):
    """Call Open3D script separately to avoid import conflicts with PyMeshLab."""

    script_loc = __file__.replace('remeshing.py', 'o3d_outlier_removal.py')

    with tempfile.NamedTemporaryFile(suffix='.obj') as tf:
        pcl = trimesh.PointCloud(points)
        tf.close()
        pcl.export(tf.name)

        # Run script, capture STDOUT of 0/1, convert to numpy mask.
        mask = subprocess.run([sys.executable, script_loc, tf.name, str(neighbours), str(std_ratio)], capture_output=True, check=True).stdout
        mask = np.frombuffer(mask, dtype=np.uint8) == 49

    return points[mask], mask


if __name__ == "__main__":
    raw_mesh = trimesh.load("test.obj").vertices

    raw_mesh = cutoff_above(raw_mesh, z=0.15)
    pymeshlab_poisson_reconstruct(raw_mesh, None).show()
