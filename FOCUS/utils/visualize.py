import numpy as np
from matplotlib import pyplot as plt
import torch

from FOCUS.matching.correspondence import Correspondence
from FOCUS.data.view import View
import trimesh


def normal_xyz2rgb(xyz):
    mask = np.all(xyz == 0, axis=-1)
    rgb = (xyz + 1) / 2
    # rgb[..., [0, 2]] = 1 - rgb[..., [0, 2]] # invert X & Z
    if mask is not None:
        rgb[mask] = 0

    return rgb


def plot_correspondences(
    views: list[View], correspondences: list[Correspondence], show_normals=False
):
    H, W = views[0].toc_image.shape[:2]
    ims = np.hstack([view.toc_image for view in views])
    normal_image_space = np.hstack(
        [normal_xyz2rgb(view.normal_image) for view in views]
    )
    normal_world_space = np.hstack(
        [normal_xyz2rgb(view.normal_world) for view in views]
    )
    ims = ims
    if show_normals:
        ims = np.vstack([ims, normal_image_space, normal_world_space])
    plt.imshow(ims)
    for correspondence in correspondences:
        npairs = len(correspondence._points_2d) - 1
        # draw line from source to sink for each pair.
        for p in range(npairs):
            (x1, y1), (x2, y2) = (
                correspondence._points_2d[0],
                correspondence._points_2d[p + 1],
            )
            idx1, idx2 = correspondence._idxs[0], correspondence._idxs[p + 1]
            plt.plot(
                (x1 + W * idx1, x2 + W * idx2), (y1, y2), c=correspondence.toc_value
            )
            # plt.scatter([x1 + W * idx1, x2 + W * idx2], [y1, y2], c='r', s=0.4)

    plt.axis("Off")
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    plt.show()


def reproject_correspondences(
    views: list[View],
    correspondences: list[Correspondence],
    projection_matrices: np.ndarray | torch.Tensor,
    first_n: int | None = 1000,
):
    """Plot the reprojection of correspondences onto views."""

    if first_n is not None:
        correspondences = correspondences[:first_n]

    H, W = views[0].toc_image.shape[:2]

    NCOLS = 5
    fig, axs = plt.subplots(ncols=NCOLS, nrows=len(views))
    [ax.axis("off") for ax in axs.ravel()]

    for i in range(len(views)):
        axs[i, 0].imshow(views[i].toc_image)
        for j in range(2, NCOLS):
            axs[i, j].imshow(
                views[i].toc_image * 0.01
            )  # black background of correct size.

        axs[i, 1].imshow(
            normal_xyz2rgb(views[i].normal_image) * views[i].mask[..., None]
        )

    for correspondence in correspondences:
        reproj_points = correspondence.reproject(projection_matrices)
        error_per_point = torch.norm(
            reproj_points - torch.tensor(correspondence._points_2d), dim=-1
        )
        normal = correspondence.get_normal()

        for idx, error, (x, y) in zip(
            correspondence._idxs, error_per_point, reproj_points
        ):
            axs[idx, 2].scatter(x, y, color=correspondence.toc_value, s=1)
            axs[idx, 3].scatter(x, y, c=error, s=1, cmap="Reds", vmin=0, vmax=10)

            # Convert world normal to local space.
            normal_local = np.dot(views[idx].R, normal)
            axs[idx, 4].scatter(x, y, color=normal_xyz2rgb(normal_local), s=1)

            # Show error
            # axs[idx, 4].scatter(x, y, s=1, c=np.rad2deg(np.arccos(np.dot(correspondence._normals[1], correspondence._normals[0]))), vmin=0, vmax=90, cmap="Reds")

    plt.axis("Off")
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    plt.show()


def show_pointcloud_with_normals(points: np.ndarray, normals: np.ndarray, scale=0.005, colors=None):
    pcl = trimesh.PointCloud(points, colors=colors)
    vec = np.column_stack((points, points + (normals * scale)))
    path = trimesh.load_path(vec.reshape((-1, 2, 3)))
    trimesh.Scene([pcl, path]).show(smooth=False)

def show_two_pointclouds_with_correspondences(points1: np.ndarray, points2: np.ndarray, frac=0.05):
    N = len(points1)
    assert N == len(points2)
    on, off = np.ones(N), np.zeros(N)

    pcl1 = trimesh.PointCloud(points1, colors=np.stack([on, off, off], axis=-1))
    pcl2 = trimesh.PointCloud(points2, colors=np.stack([off, on, off], axis=-1))

    step = int(1/frac)
    vec = np.column_stack((points1, points2))[::step]
    path = trimesh.load_path(vec.reshape((-1, 2, 3)))

    trimesh.Scene([pcl1, pcl2, path]).show(smooth=False)