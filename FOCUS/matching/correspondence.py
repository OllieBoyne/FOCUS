import numpy as np
import torch

from FOCUS.utils import triangulation


class Correspondence:
    """Collect correspondences along views, and triangulate."""

    def __init__(
        self,
        toc_value: np.ndarray,
        point_2d: tuple[float, float],
        idx: int,
        normal: tuple[float, float, float] | None = None,
        toc_unc: tuple[float, float, float] | None = None,
        norm_unc: float | None = None,
        rgb: tuple[float, float, float] | None = None,
    ):
        self._points_2d = []
        self._idxs = []
        self._normals = []
        self._toc_unc = []
        self._norm_unc = []
        self._rgbs = []

        self.toc_value = toc_value
        self._tocs = []
        self.add_observation(
            point_2d,
            idx,
            toc=toc_value,
            normal=normal,
            toc_unc=toc_unc,
            norm_unc=norm_unc,
            rgb=rgb,
        )

        self.point3d = None

    def add_observation(
        self,
        point_2d: tuple[float, float],
        idx: int,
        toc: tuple[float, float, float] | None = None,
        normal: tuple[float, float, float] | None = None,
        toc_unc: tuple[float, float, float] | None = None,
        norm_unc: float | None = None,
        rgb: tuple[float, float, float] | None = None,
    ) -> None:
        self._points_2d.append(point_2d)
        self._idxs.append(idx)
        self._tocs.append(toc)
        self._normals.append(normal)
        self._toc_unc.append(toc_unc)
        self._norm_unc.append(norm_unc)
        self._rgbs.append(rgb)

    def triangulate(self, projection_matrices) -> np.ndarray:
        p3d = triangulation.triangulate(
            projection_matrices[self._idxs], torch.tensor(self._points_2d)
        ).numpy()
        self.point3d = p3d
        return p3d

    def reproject(self, projection_matrices):
        """Reproject the 3D point to 2D."""
        if self.point3d is None:
            self.triangulate(projection_matrices)

        p3d_h = torch.cat([torch.from_numpy(self.point3d), torch.ones(1)]).float()
        p2d_h = projection_matrices[self._idxs] @ p3d_h
        p2d = p2d_h[:, :2] / p2d_h[:, 2:]
        return p2d

    def reprojection_error(self, projection_matrices):
        """Calculate the reprojection error of the 3D point."""
        p2d = self.reproject(projection_matrices)
        p2d_gt = torch.tensor(self._points_2d)
        error = torch.norm(p2d_gt - p2d, dim=-1).mean()
        return error

    @property
    def can_triangulate(self):
        return self.num_points >= 2

    @property
    def num_points(self):
        return len(self._points_2d)

    @property
    def idxs(self):
        return self._idxs

    @property
    def points_2d(self):
        return np.array(self._points_2d)

    def gather_points(self, idxs: list[int]):
        """Return points only by idxs"""
        return [self._points_2d[self._idxs.index(i)] for i in idxs]

    def get_normal_with_uncertainty(self):
        """Fuse normals."""
        weighted_normal_average = (
            (np.array(self._normals) / np.array(self._norm_unc))
        ).sum(axis=0)
        return weighted_normal_average / np.linalg.norm(weighted_normal_average)

    def get_normal(self):
        # mean_normal = np.mean(self._normals, axis=0)

        x, y, z = np.array(self._normals).T
        theta = np.arccos(z).mean()
        phi = (np.sign(y) * np.arccos(x / np.sqrt(x**2 + y**2))).mean()

        angular_mean_normal = np.array(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
        )

        return angular_mean_normal

        # return mean_normal / np.linalg.norm(mean_normal)
        # return self._normals[0]

    def average_uncertainty(self):
        return np.mean([np.linalg.norm(unc) for unc in self._toc_unc])

    def get_average_color(self):
        return np.mean(self._rgbs, axis=0)



def points_to_array(
    correspondences: list[Correspondence], num_views=2
) -> (np.ndarray, list[int]):
    "Collect correspondences into a single array. Requires a fixed number of views per correspondence." ""
    if any(c.num_points != num_views for c in correspondences):
        raise ValueError(f"All correspondences must have the {num_views} views")

    # Get all unique indices
    idxs = set(i for c in correspondences for i in c._idxs)
    assert (
        len(idxs) == num_views
    ), f"Expected {num_views} views but found {len(idxs)} indices: {idxs}"

    # Sort correspondences by index
    idxs = sorted(idxs)

    return np.array([c.gather_points(idxs) for c in correspondences])
