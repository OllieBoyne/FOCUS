import trimesh
import numpy as np
import pymeshlab  # NOTE: Importing this and open3D together cause some errors on M1 mac with poisson reconstruction.
import tempfile
import subprocess
import sys

import warnings
import os


def _setup_meshlab():
    """Needed for meshlab. May interfere with certain Open3D functionality."""
    if os.environ.get("KMP_DUPLICATE_LIB_OK", "True"):
        print("Turning off KMP_DUPLICATE_LIB_OK for Meshlab. May interfere with Open3D.")
        os.environ["KMP_DUPLICATE_LIB_OK"] = "False"

    # TODO: Check whether this is always needed. But setting this to False seems to fix the issue on Macbook M1.
    os.environ['OMP_NUM_THREADS'] = '1'

def cutoff_above(points, z=0.1) -> (np.ndarray, np.ndarray):
    """Remove all points with a height greater than this z."""
    mask = points[:, 2] < z
    return points[mask], mask


def pymeshlab_poisson_reconstruct(points, normals=None, depth=8, iters:int = 8, samplespernode:float = 1.5,
                                  cgdepth:int = 5, pointweight:int = 4,
                                  quality: np.ndarray | None=None,
                                  colors: np.ndarray | None = None) -> trimesh.Trimesh:
    _setup_meshlab()

    if colors is not None and colors.shape[-1] != 4:
        # Add alpha channel if not present
        colors = np.concatenate([colors, np.ones((len(colors), 1))], axis=-1)

    kwargs = {'vertex_matrix': points, 'v_color_matrix': colors}

    if normals is not None:
        kwargs["v_normals_matrix"] = normals

    if quality is not None:
        kwargs["v_scalar_array"] = np.clip(quality, a_min=1e-4, a_max=1.0)

    mesh = pymeshlab.Mesh(**kwargs)

    ms = pymeshlab.MeshSet()
    ms.add_mesh(mesh)
    if normals is None:
        ms.compute_normal_for_point_clouds()

    # Threads > 1 seems to cause occasional failures as it is non-deterministic...
    ms.generate_surface_reconstruction_screened_poisson(
        preclean=True, threads=1, depth=depth, iters=iters, samplespernode=samplespernode,
        cgdepth=cgdepth, pointweight=pointweight, confidence=quality is not None
    )

    vertex_colors = None
    if colors is not None:
        vertex_colors = ms.current_mesh().vertex_color_matrix()
        vertex_colors = (vertex_colors * 255).astype(np.uint8)

    tmesh = trimesh.Trimesh(vertices=ms.current_mesh().vertex_matrix(), faces=ms.current_mesh().face_matrix(),
                            vertex_colors=vertex_colors)

    return tmesh


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

def crop_to_plane(mesh: trimesh.Trimesh, origin: np.ndarray, normal: np.ndarray) -> trimesh.Trimesh:
    """Remove all vertices (and corresponding faces) that are not on the positive side of the plane."""
    # Remove faces on the wrong side of the plane.
    points = np.array(mesh.vertices)
    points_in_plane = np.dot(points - origin, normal) > 0

    vertex_mapping = -np.ones(len(mesh.vertices))
    new_to_old = np.where(points_in_plane)[0]
    vertex_mapping[new_to_old] = np.arange(len(new_to_old))

    # Mapping of new indices to old indices
    faces_mask = points_in_plane[mesh.faces].all(axis=1)

    new_faces = mesh.faces[faces_mask]
    new_faces = vertex_mapping[new_faces]

    vertex_colors = mesh.visual.vertex_colors[points_in_plane]

    new_mesh = trimesh.Trimesh(vertices=points[points_in_plane], faces=new_faces, vertex_colors=vertex_colors)
    return new_mesh

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
