import numpy as np
from FOCUS.matching.correspondence import Correspondence
from typing import Callable, Literal
import FOCUS.utils.remeshing as remeshing
from FOCUS.utils import triangulation
import torch
import trimesh

class FusedPointCloud:
    """Collection of correspondences, for batched processing operations."""

    def __init__(self, correspondences: list[Correspondence], num_views: int):
        self._correspondences = correspondences
        self._mask = np.ones(len(correspondences), dtype=bool)

        self._points_3d = np.zeros((len(correspondences), 3))
        self._points_2d = np.zeros((len(correspondences), num_views, 2))
        self._reproj_points_2d = np.zeros((len(correspondences), num_views, 2))
        self._view_mask = np.zeros((len(correspondences), num_views))

        self._transform = np.eye(4)


    @property
    def correspondences(self) -> list[Correspondence]:
        return [c for c, m in zip(self._correspondences, self._mask) if m]

    @property
    def normals(self):
        return [self._transform[:3, :3] @ c.get_normal() for c, m in zip(self._correspondences, self._mask) if m]

    @property
    def points_3d(self):
        return self._points_3d[self._mask]

    @property
    def toc(self):
        return [c.toc_value for c, m in zip(self._correspondences, self._mask) if m]

    @property
    def colors(self):
        return [c.get_average_color() for c, m in zip(self._correspondences, self._mask) if m]

    def filter_correspondences_by(self, function: Callable, key: str = ""):
        previous_length = sum(self._mask)

        for i in np.nonzero(self._mask)[0]:
            self._mask[i] = function(self._correspondences[i])

        new_length = sum(self._mask)

        s = f"[{key}]" if key else ""
        s += f" Filter {previous_length} -> {new_length} ({new_length - previous_length})"
        print(s)

    def cutoff_points(self, cutoff_height: float, direction: Literal["above", "below"]):
        previous_length = sum(self._mask)
        if direction == "above":
            self._mask *= self._points_3d[:, 2] < cutoff_height
        else:
            self._mask *= self._points_3d[:, 2] > cutoff_height
        new_length = sum(self._mask)

        print(f"[Cutoff {direction} {cutoff_height}] Filter {previous_length} -> {new_length} ({new_length - previous_length})")

    def cutoff_points_by_plane(self, origin: np.ndarray, normal: np.ndarray):
        """Keep only points in positive plane."""
        previous_length = sum(self._mask)
        distances = np.dot(self._points_3d - origin, normal)
        self._mask *= distances > 0 # Keep points above plane
        new_length = sum(self._mask)

        print(f"[Cutoff plane] Filter {previous_length} -> {new_length} ({new_length - previous_length})")

    def remove_outliers(self, neighbours=20, std_ratio=2.0):
        new_points, mask = remeshing.open3d_remove_outliers(self.points_3d, neighbours=neighbours, std_ratio=std_ratio)

        previous_length = sum(self._mask)
        self._mask[self._mask] = mask
        new_length = sum(self._mask)

        print(f"[Outliers] Filter {previous_length} -> {new_length} ({new_length - previous_length}).")

    def remove_outliers_by_centroid_distance(self, frac: float = 0.01):
        """Remove outliers by distance from the centroid."""
        previous_length = sum(self._mask)
        centroid = np.mean(self.points_3d, axis=0)
        distances = np.linalg.norm(self.points_3d - centroid, axis=-1)
        mask = distances < np.percentile(distances, 100 * (1-frac))
        self._mask[self._mask] = mask
        new_length = sum(self._mask)

        print(f"[Outliers by centroid] Filter {previous_length} -> {new_length} ({new_length - previous_length}).")

    def apply_transform(self, T: np.ndarray):
        self._transform = T @ self._transform
        self._points_3d = trimesh.transform_points(self._points_3d, T)

    def filter_by_reprojection_error(self, threshold: float):
        previous_length = sum(self._mask)
        reproj_error = np.linalg.norm(self._reproj_points_2d - self._points_2d, axis=-1)  # C x N
        reproj_error = (reproj_error * self._view_mask).sum(axis=-1) / self._view_mask.sum(axis=-1)
        self._mask *= reproj_error < threshold
        new_length = sum(self._mask)

        print(f"[Reprojection error] Filter {previous_length} -> {new_length} ({new_length - previous_length}).")

    def triangulate(self, projection_matrices: torch.Tensor):
        N, C = len(projection_matrices), sum(self._mask)

        correspondences = self.correspondences
        points_2d = np.zeros((C, N, 2))
        mask = np.zeros((C, N))
        for i in range(C):
            views = correspondences[i].idxs
            mask[i, views] = 1
            points_2d[i, views] = correspondences[i].points_2d

        self._points_2d[self._mask] = points_2d
        self._points_3d[self._mask] = triangulation.batch_triangulation(projection_matrices.numpy(), points_2d, mask)
        self._view_mask[self._mask] = mask

        for c, p3d in zip(correspondences, self._points_3d[self._mask]):
            c.point3d = p3d

        # Reproject all correspondences (even outside of mask)
        p3d_h = np.pad(self._points_3d, ((0, 0), (0, 1)), constant_values=1)
        p2d_h = (projection_matrices[None, :] @ p3d_h[:, None, :, None])[..., 0]
        self._reproj_points_2d = p2d_h[..., :2] / p2d_h[..., 2:]

    def to_trimesh(self) -> trimesh.Trimesh:
        return trimesh.Trimesh(
            vertices=self.points_3d, faces=[], vertex_normals=self.normals, vertex_colors=self.toc
        )