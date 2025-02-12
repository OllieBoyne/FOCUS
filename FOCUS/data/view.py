import numpy as np

from FOCUS.utils import normals as normals_utils

class View:

    rgb: np.ndarray
    norm_rgb: np.ndarray
    toc_rgb: np.ndarray
    norm_unc: np.ndarray
    toc_unc: np.ndarray
    mask: np.ndarray
    norm_xyz: np.ndarray

    calibration_data: dict
    key: str = None

    _nn = None

    def __init__(self,
                 rgb: np.ndarray,
                norm_rgb: np.ndarray,
                toc_rgb: np.ndarray,
                toc_unc: np.ndarray,
                norm_unc: np.ndarray,
                mask: np.ndarray,
                calibration_data: dict,
                key: str = None,
                idx: int = None,
                ):

        self.rgb = rgb
        self.norm_rgb = norm_rgb
        self.toc_rgb = toc_rgb
        self.toc_unc = toc_unc
        self.norm_unc = norm_unc
        self.mask = mask
        self.calibration_data = calibration_data
        self.key = key
        self.idx = idx
        self._nn = None

        norm_xyz = normals_utils.rgb2xyz(self.norm_rgb)

        norm_xyz[..., -1] *= -1  # invert Z
        self.norm_xyz = norm_xyz
        self.norm_xyz[mask] = (
            norm_xyz[mask] / np.linalg.norm(norm_xyz[mask], axis=-1)[..., None]
        )

        # Calculate normal in world frame
        self.normal_world = ((self.R @ self.norm_xyz.reshape(-1, 3).T).T).reshape(
            *self.norm_xyz.shape
        )
        self.normal_world[mask] = (
            self.normal_world[mask]
            / np.linalg.norm(self.normal_world[mask], axis=-1)[..., None]
        )
        self.normal_world *= mask[..., None]

    @property
    def R(self) -> np.ndarray:
        return self.calibration_data.get("R", np.eye(3))

    @property
    def T(self) -> np.ndarray:
        return self.calibration_data["T"]

    @property
    def cx(self) -> float:
        return self.calibration_data["cx"]

    @property
    def cy(self) -> float:
        return self.calibration_data["cy"]

    @property
    def f(self) -> float:
        return self.calibration_data["f"]

    @property
    def size(self) -> tuple[int, int]:
        return self.calibration_data["width"], self.calibration_data["height"]

    @property
    def image_shape(self) -> tuple[int, int]:
        """Return the image shape in (W, H) format."""
        return self.size[1], self.size[0]

    @property
    def toc_nn(self):
        if self._nn is None:
            from sklearn import neighbors
            self._nn = neighbors.NearestNeighbors(
                n_neighbors=1, n_jobs=-1, algorithm="kd_tree"
            ).fit(self.toc_rgb.reshape(-1, 3))
        return self._nn

    @property
    def mask_coverage(self) -> float:
        """Return the mask coverage of the view."""
        w, h = self.mask.shape
        return np.sum(self.mask) / (w * h)