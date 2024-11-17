import cv2

from FOCUS.utils import camera
import torch
import numpy as np


def _homogeneous_to_euclidean(
    points: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Converts homogeneous points to euclidean

    Args:
        points numpy array or torch tensor of shape (N, M + 1): N homogeneous points of dimension M

    Returns:
        numpy array or torch tensor of shape (N, M): euclidean points
    """
    if isinstance(points, np.ndarray):
        return (points.T[:-1] / points.T[-1]).T
    elif torch.is_tensor(points):
        return (points.transpose(1, 0)[:-1] / points.transpose(1, 0)[-1]).transpose(
            1, 0
        )
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")


def cameras_to_projection_matrix(cameras: camera.Camera) -> torch.Tensor:
    """Converts a Camera object to a projection matrix.

    Args:
        cameras: Cameras object length N

    Returns:
        torch.Tensor of shape (N, 3, 4): projection matrices
    """
    projection_matrices = (
        cameras.get_world_to_pix_transform().get_matrix().permute(0, 2, 1)
    )

    # Need to convert PyTorch3D's row arrangement to the typical projection approach.
    projection_matrices = projection_matrices[:, [0, 1, 3], :]

    return projection_matrices


def triangulate(
    projection_matrices: torch.Tensor, points_2d: torch.Tensor, confidences=None
) -> torch.Tensor:
    """Triangulate a point in 3D based on 2D correspondences.

    Uses DLT, adapted from: https://github.com/karfly/learnable-triangulation-pytorch/blob/master/mvn/utils/multiview.py

    Args:
        projection_matrices: [N x 3 x 4] projection matrices.
        points_2d: 2D points in pixel coordinates, [N x 2]
        confidences: Optional weighting per observation [N]
    Returns:
        3D point in world coordinates [3]
    """

    num_views = len(projection_matrices)
    if confidences is None:
        confidences = torch.ones(
            num_views, dtype=torch.float32, device=points_2d.device
        )

    A = projection_matrices[:, 2:3].expand(num_views, 2, 4) * points_2d.view(
        num_views, 2, 1
    )
    A -= projection_matrices[:, :2]
    A *= confidences.view(-1, 1, 1)

    u, s, vh = torch.linalg.svd(A.view(-1, 4))

    point_3d_homo = -vh[3, :]
    return _homogeneous_to_euclidean(point_3d_homo.unsqueeze(0))[0]


def batch_triangulation(projection_matrices: np.ndarray, points_2d: np.ndarray, mask: np.ndarray):
    """Batch triangulate over C correspondences in N views.

    Args:
        projection_matrices: [N x 3 x 4] Projection matrices.
        points_2d: 2D points in pixel coordinates, [C x N x 2]
        mask: [C x N] of which correspondences are used in which views
    Returns:
        3D point in world coordinates [3]
    """

    N = len(projection_matrices)
    C = len(mask)

    points_3d_homogenous = np.zeros((C, 4))

    for i in range(C):
        view_idxs = np.nonzero(mask[i])[0]

        A1 = np.broadcast_to(projection_matrices[view_idxs, 2:3], (len(view_idxs), 2, 4))
        A2 = points_2d[i, view_idxs, :, None]
        A = A1 * A2

        A -= projection_matrices[view_idxs, :2]

        u, s, vh = np.linalg.svd(A.reshape(-1, 4))

        points_3d_homogenous[i] = -vh[3, :]

    return _homogeneous_to_euclidean(points_3d_homogenous)
