# from FOCUS import camera
import numpy as np

from FOCUS.matching.view import FusionView
from FOCUS.matching import match
from FOCUS.fusion import fused_point_cloud
from FOCUS.utils import visualize

from FOCUS.data.dataset import View
import json
import torch

# from FOCUS.fusion import visualize
from FOCUS.utils import triangulation
from FOCUS.utils import remeshing
from FOCUS.utils import camera
from pathlib import Path
import dataclasses

from time import perf_counter

from FOCUS.fusion.hyperparameters import FusionHyperparameters
import trimesh

def fuse(views: [View], output_folder: Path, hyperparameters: FusionHyperparameters = FusionHyperparameters()):
    """Apply fusion to a list of views."""

    time_start = perf_counter()

    output_folder.mkdir(exist_ok=True, parents=True)

    # Save hyperparameters as JSON
    with open(output_folder / "hyperparameters.json", "w") as f:
        json.dump(dataclasses.asdict(hyperparameters), f)

    new_views = []
    for i, view in enumerate(views):
        v = FusionView.from_view(view, idx=i)
        new_views.append(v)

    views = new_views

    cameras = camera.Camera(
        R=torch.stack([torch.from_numpy(v.R) for v in views]),
        T=torch.stack([torch.from_numpy(v.T) for v in views]),
        size=torch.stack([torch.tensor(v.image_shape) for v in views]),
        focal_length=torch.stack([torch.tensor([v.f]) for v in views]),
    )

    projection_matrices = triangulation.cameras_to_projection_matrix(cameras)

    # Build an initial sample of correspondences.
    correspondences = match.find_matches(views, samples_per_image=hyperparameters.samples_per_image,
                                         max_dist=hyperparameters.toc_correspondence_threshold,
                                         subpixel_scaling=hyperparameters.nn_upsampling_factor)

    # Create point cloud
    point_cloud = fused_point_cloud.FusedPointCloud(correspondences, len(views))
    point_cloud.filter_correspondences_by(lambda x: x.can_triangulate, "Triangulate")

    point_cloud.triangulate(projection_matrices)

    if hyperparameters.is_world_space:
        point_cloud.cutoff_points(hyperparameters.mesh_cutoff_heights[1], "above")
        point_cloud.cutoff_points(hyperparameters.mesh_cutoff_heights[0], "below")

    else:

        # Some extra filtering steps found to improve performance on non-world space data.
        point_cloud.remove_outliers(std_ratio=5.0)
        point_cloud.remove_outliers_by_centroid_distance(frac=0.01)

        # Need to align the mesh to world space (using TOC).
        T, a, cost = trimesh.registration.procrustes(point_cloud.points_3d, point_cloud.toc, reflection=True)

        # Remove reflection if it exists.
        if np.linalg.det(T[:3, :3]) < 0:
            # re-reflect along y
            refl_y = np.eye(4)
            refl_y[1, 1] = -1
            T = refl_y @ T

        point_cloud.apply_transform(T)

        # visualize.show_pointcloud_with_normals(point_cloud.points_3d, np.array(point_cloud.normals), colors=point_cloud.toc)

    point_cloud.remove_outliers(std_ratio=5.0)

    # Export point cloud
    pcl_as_mesh = point_cloud.to_trimesh()
    pcl_as_mesh.export(output_folder / "points.obj", include_normals=True, include_color=True)

    np.savez(output_folder / "points.npz", points=point_cloud.points_3d, normals=point_cloud.normals)
    # visualize.show_pointcloud_with_normals(point_cloud.points_3d, np.array(point_cloud.normals), colors=point_cloud.toc)

    # Perform Poisson reconstruction, and export mesh.
    mesh = remeshing.pymeshlab_poisson_reconstruct(
        point_cloud.points_3d,
        normals=np.array(point_cloud.normals)
        if hyperparameters.screened_poisson_use_calculated_normals
        else None,
        **hyperparameters.screened_poisson_parameters,
    )

    mesh = remeshing.crop_to_original_pointcloud(
        mesh, point_cloud.points_3d, padding=hyperparameters.bbox_cropping_padding
    )
    mesh = remeshing.keep_pointcloud_island(mesh, point_cloud.points_3d)
    mesh.export(output_folder / "mesh.obj")

    # End process
    time_end = perf_counter()
    metadata = {"time_elapsed": time_end - time_start}
    with open(output_folder / "metadata.json", "w") as f:
        json.dump(metadata, f)