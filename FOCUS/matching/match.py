"""Find matches across Views."""
import FOCUS.data.view
from FOCUS.matching import correspondence

import cv2
import numpy as np
import torch

from FOCUS.utils import sampler
from FOCUS.matching.correspondence import Correspondence
from FOCUS.data.view import View

import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()})
from FOCUS.utils import rgb_matcher

def find_matches(views: [FOCUS.data.view.View], num_correspondences: int,
                 max_dist: float = 0.002,
                 subpixel_scaling: int = 8
                 ) -> [correspondence.Correspondence]:
    """Find matches across Views."""

    correspondences: list[correspondence.Correspondence] = []

    torch.manual_seed(10) # TODO: add seed to hyperparameters

    samples_per_image = num_correspondences // len(views)
    for view in views:
        correspondences += sample_in_mask(view, samples_per_image)

    for view in views:
        nearest_neighbour_match(
            view,
            correspondences,
            max_dist=max_dist,
            subpixel_scaling=subpixel_scaling
        )

    return correspondences


def nearest_neighbour_match(
    view: View,
    correspondences: list[Correspondence],
    max_dist=0.02,
    subpixel_scaling: int = 1,
):
    other_correspondences = [c for c in correspondences if c._idxs[0] != view.idx]
    other_tocs = np.array([c.toc_value for c in other_correspondences])

    dist, ind = view.toc_nn.kneighbors(other_tocs)

    # Keep matches that are likely to resolve in error < max_dist for subpixel (for speed).
    valid_matches = dist < 5 * max_dist

    H, W = view.toc_rgb.shape[:2]

    # Batch subsampling
    x_batch, y_batch, sampled_toc_batch = _subsample_toc_batch(
        view,
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
            view.idx,
            toc=sampled_toc,
            normal=view.normal_world[y_int, x_int],
            toc_unc=view.toc_unc[y_int, x_int],
            norm_unc=view.norm_unc[y_int, x_int],
            rgb=view.rgb[y_int, x_int] if view.rgb is not None else None,
        )

def _subsample_toc_batch(
    view: View,
    x: np.ndarray,
    y: np.ndarray,
    color: np.ndarray,
    width: int = 1,
    scale_factor: int = 1,
):
    """Run _subsample_toc in batch."""

    # TODO: minor accuracy checks, especially for edge cases.

    N = len(x)
    H, W = view.toc_rgb.shape[:2]

    roi_size = (1 + 2 * width) * scale_factor

    corners = np.stack(
        [(x - width) * scale_factor, (y - width) * scale_factor], axis=-1
    ).astype(np.int32)

    # TODO: review a way to perhaps make this more accurate, or just ignore edge cases.
    corners[:, 0] = np.clip(corners[:, 0], 0, W * scale_factor - roi_size)
    corners[:, 1] = np.clip(corners[:, 1], 0, H * scale_factor - roi_size)

    upsampled_img = cv2.resize(
        view.toc_rgb,
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

def sample_in_mask(view: View, n_samples) -> list[Correspondence]:
    # TODO: make not PyTorch dependent?
    mask_batch = torch.from_numpy(view.mask).unsqueeze(0)
    samples = sampler.samples_in_mask(mask_batch, n_samples)

    rgb_image_samples = sampler.sample_image(
        torch.from_numpy(view.rgb).unsqueeze(0), samples
    )[0].numpy()

    toc_image_samples = sampler.sample_image(
        torch.from_numpy(view.toc_rgb).unsqueeze(0), samples
    )[0].numpy()

    normal_world_samples = sampler.sample_image(
        torch.from_numpy(view.normal_world).unsqueeze(0), samples
    )[0].numpy()

    toc_unc_samples = sampler.sample_image(
        torch.from_numpy(view.toc_unc).unsqueeze(0), samples
    )[0].numpy()

    norm_unc_samples = sampler.sample_image(
        torch.from_numpy(view.norm_unc).unsqueeze(0), samples
    )[0].numpy()

    correspondences = []
    samples = samples.squeeze(0)
    for n in range(len(samples)):
        correspondence = Correspondence(
            toc_image_samples[n],
            samples[n].tolist(),
            view.idx,
            normal=normal_world_samples[n],
            toc_unc=toc_unc_samples[n],
            norm_unc=norm_unc_samples[n],
            rgb=rgb_image_samples[n],
        )
        correspondences.append(correspondence)

    return correspondences
