from time import perf_counter

import cv2
import numpy as np
import torch
from sklearn import neighbors

from FOCUS.utils import sampler
from FOCUS.matching.correspondence import Correspondence
from FOCUS.utils import normals, triangulation
from FOCUS.data.dataset import View

import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()})
from FOCUS.utils import rgb_matcher


class FusionView:
    """Viewpoint from a camera."""

    toc_image: np.ndarray
    toc_unc_image: np.ndarray
    normal_image: np.ndarray
    normal_world: np.ndarray
    mask: np.ndarray
    idx: int

    _F = {}  # fundamental matrices to all other views.

    def __init__(
        self,
        toc_image,
        toc_unc_image,
        normal_image,
        normal_uncertainty,
        mask,
        idx: int,
        calibration_data: dict,
    ):

        self.toc_image = toc_image
        self.toc_unc_image = toc_unc_image
        self._toc_image_upsampled = None

        self.normal_uncertainty = normal_uncertainty

        normal_image[..., -1] *= -1  # invert Z
        self.normal_image = normal_image
        self.normal_image[mask] = (
            normal_image[mask] / np.linalg.norm(normal_image[mask], axis=-1)[..., None]
        )

        self.mask = mask
        self.idx = idx
        self.calibration_data = calibration_data

        # Calculate normal in world frame
        # TODO: Experimental, currently not working.
        self.normal_world = ((self.R @ normal_image.reshape(-1, 3).T).T).reshape(
            *normal_image.shape
        )
        self.normal_world[mask] = (
            self.normal_world[mask]
            / np.linalg.norm(self.normal_world[mask], axis=-1)[..., None]
        )
        self.normal_world *= mask[..., None]

        self._nn = None

    @classmethod
    def from_view(cls, view: View, idx: int = 0):
        return cls(
            view.toc_rgb,
            view.toc_unc,
            view.norm_xyz,
            view.norm_unc,
            view.mask,
            idx,
            calibration_data=view.calibration_data,
        )

    @property
    def R(self) -> np.ndarray:
        return self.calibration_data["R"]

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
        return (self.size[1], self.size[0])

    @property
    def nn(self):
        if self._nn is None:
            # Scale up image for subpixel matching.
            self._nn = neighbors.NearestNeighbors(
                n_neighbors=1, n_jobs=-1, algorithm="kd_tree"
            ).fit(self.toc_image.reshape(-1, 3))
        return self._nn

    def nearest_neighbour_match(
        self,
        correspondences: list[Correspondence],
        max_dist=0.02,
        subpixel_scaling: int = 1,
    ):
        other_correspondences = [c for c in correspondences if c._idxs[0] != self.idx]
        other_tocs = np.array([c.toc_value for c in other_correspondences])

        dist, ind = self.nn.kneighbors(other_tocs)

        # Keep matches that are likely to resolve in error < max_dist for subpixel (for speed).
        valid_matches = dist < 5 * max_dist

        H, W = self.toc_image.shape[:2]

        # Batch subsampling
        x_batch, y_batch, sampled_toc_batch = self._subsample_toc_batch(
            ind.flatten() % W,
            ind.flatten() // W,
            other_tocs,
            width=1,
            scale_factor=subpixel_scaling,
        )

        errors = np.linalg.norm(sampled_toc_batch - other_tocs, axis=-1)

        for n in np.nonzero(valid_matches)[0]:
            c = other_correspondences[n]
            idx = ind[n].item()

            x_int, y_int = idx % W, idx // W
            x_float, y_float, sampled_toc = x_batch[n], y_batch[n], sampled_toc_batch[n]

            error = errors[n]
            if error > max_dist:
                continue

            pixel_value = (x_float, y_float)
            c.add_observation(
                pixel_value,
                self.idx,
                toc=sampled_toc,
                normal=self.normal_world[y_int, x_int],
                toc_unc=self.toc_unc_image[y_int, x_int],
                norm_unc=self.normal_uncertainty[y_int, x_int],
            )

    def _subsample_toc_batch(
        self,
        x: np.ndarray,
        y: np.ndarray,
        color: np.ndarray,
        width: int = 1,
        scale_factor: int = 1,
    ):
        """Run _subsample_toc in batch."""

        # TODO: minor accuracy checks, especially for edge cases.

        N = len(x)
        H, W = self.toc_image.shape[:2]

        roi_size = (1 + 2 * width) * scale_factor

        corners = np.stack(
            [(x - width) * scale_factor, (y - width) * scale_factor], axis=-1
        ).astype(np.int32)

        # TODO: review a way to perhaps make this more accurate, or just ignore edge cases.
        corners[:, 0] = np.clip(corners[:, 0], 0, W * scale_factor - roi_size)
        corners[:, 1] = np.clip(corners[:, 1], 0, H * scale_factor - roi_size)

        upsampled_img = cv2.resize(
            self.toc_image,
            None,
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_LINEAR,
        )

        best_match = rgb_matcher.rgb_match(upsampled_img, corners, roi_size, roi_size, color)

        # calculate offset of new position relative to x, y
        x_roi_start, y_roi_start = corners.T

        x_in_window = best_match % roi_size
        y_in_window = best_match // roi_size

        x_new_upsampled = x_roi_start + x_in_window
        y_new_upsampled = y_roi_start + y_in_window

        x_new = x_new_upsampled / scale_factor
        y_new = y_new_upsampled / scale_factor

        return x_new, y_new, upsampled_img[y_new_upsampled, x_new_upsampled]

    def get_F(self, other_idx, projection_matrices):
        if other_idx in self._F:
            return self._F[other_idx]

        F = triangulation.compute_fundamental_matrix(
            projection_matrices[other_idx].numpy(),
            projection_matrices[self.idx].numpy(),
        )
        self._F[other_idx] = F
        return F

    def sample_in_mask(self, n_samples) -> list[Correspondence]:
        # TODO: make not PyTorch dependent?
        mask_batch = torch.from_numpy(self.mask).unsqueeze(0)
        samples = sampler.samples_in_mask(mask_batch, n_samples)

        toc_image_samples = sampler.sample_image(
            torch.from_numpy(self.toc_image).unsqueeze(0), samples
        )[0].numpy()

        normal_world_samples = sampler.sample_image(
            torch.from_numpy(self.normal_world).unsqueeze(0), samples
        )[0].numpy()

        toc_unc_samples = sampler.sample_image(
            torch.from_numpy(self.toc_unc_image).unsqueeze(0), samples
        )[0].numpy()

        norm_unc_samples = sampler.sample_image(
            torch.from_numpy(self.normal_uncertainty).unsqueeze(0), samples
        )[0].numpy()

        correspondences = []
        samples = samples.squeeze(0)
        for n in range(len(samples)):
            correspondence = Correspondence(
                toc_image_samples[n],
                samples[n].tolist(),
                self.idx,
                normal=normal_world_samples[n],
                toc_unc=toc_unc_samples[n],
                norm_unc=norm_unc_samples[n],
            )
            correspondences.append(correspondence)

        return correspondences
