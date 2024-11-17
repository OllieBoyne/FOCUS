import torch

nn = torch.nn
import numpy as np
from typing import Union
from pytorch3d.renderer.cameras import PerspectiveCameras, get_ndc_to_screen_transform
from pytorch3d.transforms import Transform3d

# TODO: Simplify, numpy approach.

# The camera reference frame we use is:
# 	(X, Y, Z) = (left, forward, up)
# PyTorch3D uses:
# 	(X, Y, Z) = (left, up, forward)


_default_T = np.array([0, 0, 3])
_default_focal_length = 500
_default_size = (256, 256)


def _to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.float()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    if isinstance(x, (int, float)):
        return torch.tensor([x]).float()

    raise ValueError(f'Cannot convert {type(x)} to torch tensor')


class Camera(PerspectiveCameras):
    def __init__(self,
                 T: Union[torch.Tensor, np.ndarray] = _default_T,
                 R: Union[torch.Tensor, np.ndarray] = None,
                 size: Union[torch.Tensor, np.ndarray, list, tuple] = _default_size,
                 focal_length: Union[torch.Tensor, np.ndarray, float] = _default_focal_length):
        """
		Basic projection camera model. Supports batching.

		:param T: [(B), 3] Camera position in world space
		:param R: [(B), 3, 3] Rotation matrix.
		:param size: [(B), 2] Image size in pixels - H x W
		:param focal_length: [(B)] Field of view in degrees
		"""

        # cast focal_length, size to tensors as need to manage dtypes
        focal_length = _to_tensor(focal_length)
        size = _to_tensor(size)

        principal_point = size[..., [1, 0]] / 2  # principal point (x, y)

        super().__init__(focal_length=focal_length,
                         principal_point=principal_point,
                         R=R,
                         T=T,
                         image_size=size,
                         in_ndc=False)

    def project(self, points: torch.Tensor, eps=1e-6, pixel=False) -> torch.Tensor:
        """Project points into the image space, either NDC or pixel

		:param points: [... x 3] tensor of 3D points
		:param eps: small value to prevent division by zero
		:param pixel: if True, return pixel coordinates instead of image coordinates

		:returns: [ N x ... x 2 ] 2D positions, N is number of views.
		"""
        if pixel:
            return self.transform_points_screen(points, eps=eps)[..., :2]

        return self.transform_points_ndc(points, eps=eps)[..., :2]

    def get_world_to_pix_transform(self, with_xyflip=True) -> Transform3d:
        """Get transform from world to pixel coordinates"""
        assert not self.in_ndc(), "Camera must *not* be in NDC space"
        t1 = self.get_full_projection_transform()
        t2 = self.get_ndc_camera_transform()
        t3 = get_ndc_to_screen_transform(self, with_xyflip=with_xyflip, image_size=self.get_image_size())
        return t1.compose(t2).compose(t3)


    def __getitem__(self, item: int | list[int]):
        if isinstance(item, int):
            item = [item]

        camera = Camera(self.T[item], self.R[item], self.image_size[item], self.focal_length[item])
        return camera.to(self.device)