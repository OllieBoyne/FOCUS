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


def triangulate_ransac(
    projection_matrices: torch.Tensor,
    points_2d: torch.Tensor,
    confidences=None,
    num_iterations=25,
    sample_size: int = 3,
    inlier_threshold_pixels=3.0,
):
    # TODO: speed-up and test this.

    if len(projection_matrices) <= sample_size:
        return triangulate(projection_matrices, points_2d, confidences)

    best_num_inliers = 0
    best_inlier_ids = ()

    if confidences is None:
        confidences = torch.ones(
            len(projection_matrices), dtype=torch.float32, device=points_2d.device
        )

    for i in range(num_iterations):
        sample = np.random.choice(len(projection_matrices), sample_size, replace=False)
        point_3d = triangulate(
            projection_matrices[sample], points_2d[sample], confidences[sample]
        )

        # Project.
        p3d_h = torch.cat([point_3d, torch.ones(1)])
        p2d_h = projection_matrices @ p3d_h
        p2d = p2d_h[:, :2] / p2d_h[:, 2:]

        p2d_gt = torch.tensor(points_2d)
        error = torch.norm(p2d_gt - p2d, dim=-1)
        inliers = np.argwhere(error < inlier_threshold_pixels).ravel()

        if len(inliers) > best_num_inliers:
            best_num_inliers = len(inliers)
            best_inlier_ids = inliers.tolist()

    if best_num_inliers < 2:
        best_inlier_ids = range(len(projection_matrices))

    # Recalculate fit with all inliers.
    return triangulate(
        projection_matrices[best_inlier_ids],
        points_2d[best_inlier_ids],
        confidences[best_inlier_ids],
    )


def triangulate(
    projection_matrices: torch.Tensor, points_2d: torch.Tensor, confidences=None
) -> torch.Tensor:
    """Triangulate a point in 3D based on 2D correspondences.

    Uses DLT, adapted from: https://github.com/karfly/learnable-triangulation-pytorch/blob/master/mvn/utils/multiview.py

    Args:
        cameras: N projection matrices.
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
    """Proj Matrix, N x 3 x 4
    Points_2d, C x N x 2
    mask: C x N, which views are used for which points.
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


def triangulate_no_extrinsics(
    points_view_1: np.ndarray | torch.Tensor,
    points_view_2: np.ndarray | torch.Tensor,
    K,
):
    """Given many image correspondences, recover relative camera poses + triangulate all points.

    Args:
        points_view_1: [N x 2] points from view 1.
        points_view_2: [N x 2] points from view 2.
        K: Joint intrinsic matrix [3 x 3]
    """

    if isinstance(points_view_1, torch.Tensor):
        points_view_1 = points_view_1.cpu().detach().numpy()

    if isinstance(points_view_2, torch.Tensor):
        points_view_2 = points_view_2.cpu().detach().numpy()

    E, E_mask = cv2.findEssentialMat(points_view_1, points_view_2, K, method=cv2.RANSAC)

    # Return mask of points used in RANSAC for E.
    E_mask = E_mask.ravel().astype(bool)

    retval, R, T, mask, triangulated_points_h = cv2.recoverPose(
        E, points_view_1, points_view_2, K, distanceThresh=0.01
    )

    # Convert from homogenous to 3D points.
    triangulated_points = (triangulated_points_h[:3] / triangulated_points_h[3]).T

    # reproject points
    # P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    # P2 = K @ np.hstack((R, T))
    # reproj_view_1 = cv2.convertPointsFromHomogeneous((P1 @ triangulated_points_h).T).reshape(-1, 2)
    # reproj_view_2 = cv2.convertPointsFromHomogeneous((P2 @ triangulated_points_h).T).reshape(-1, 2)

    return dict(
        retval=retval, R=R, T=T, mask=E_mask, triangulatedPoints=triangulated_points
    )


def compute_fundamental_matrix(P1, P2):
    """Compute the fundamental matrix from two projection matrices.

    Args:
        P1: Projection matrix 1 [3 x 4]
        P2: Projection matrix 2 [3 x 4]

    Returns:
        Fundamental matrix [3 x 3]
    """
    _, _, V1t = np.linalg.svd(P1)
    _, _, V2t = np.linalg.svd(P2)
    c1 = V1t[-1]
    c2 = V2t[-1]
    c1 /= c1[-1]
    c2 /= c2[-1]

    # Compute the epipole e2 in the second view
    e2 = P2 @ c1
    e2x = np.array([[0, -e2[2], e2[1]], [e2[2], 0, -e2[0]], [-e2[1], e2[0], 0]])

    # Compute the fundamental matrix F
    F = e2x @ P2 @ np.linalg.pinv(P1)
    return F

def point_to_epipolar_line(F, p1):
    """Compute epipolar lines from a fundamental matrix.

    F: fundamental matrix 3x3 or N x 3 x 3
    p1: points in first view [N x 2]
    returns: lines in second view [N x 3] (ax + by + c = 0)"""
    if F.ndim == 2:
        F = F[None]

    return (F @ np.hstack((p1, np.ones((len(p1), 1))))[..., None])[..., 0]