"""A dataset which loads in images, inference results, and calibration into a single,
    unified format."""

import functools
from time import perf_counter

import cv2
from pathlib import Path
import numpy as np
import json
import os

from FOCUS.utils import normals as normals_utils

import dataclasses

@dataclasses.dataclass
class View:
    rgb: np.ndarray

    norm_rgb: np.ndarray
    toc_rgb: np.ndarray
    norm_unc: np.ndarray
    toc_unc: np.ndarray
    mask: np.ndarray
    norm_xyz: np.ndarray

    calibration_data: dict

    def __post_init__(self):
        self.norm_xyz = normals_utils.rgb2xyz(self.norm_rgb)

# @functools.lru_cache(maxsize=100)
def load_view(directory: Path) -> View:
    """Expects a directory with all relevant information for a single image."""

    image_data = _load_image_data(directory)
    inference_data = _load_inference_data(directory)
    calibration_data = _load_calibration_data(directory)

    return View(**image_data, **inference_data, calibration_data=calibration_data)

def load_views(directory: Path, ignore_list=('colmap', 'frames')) -> [View]:
    """Load in all views from a directory."""
    return [load_view(directory / d) for d in sorted(os.listdir(directory)) if os.path.isdir(directory / d) and d not in ignore_list]

def _load_image_data( directory):
    return {'rgb': _get_image(directory, "rgb")}

def _load_inference_data(directory):
    norm_rgb = _get_image(directory, "normal", num_channels=3)
    norm_unc = _get_image(directory, "norm_unc", num_channels=1)
    toc_rgb = _get_image(directory, "toc", num_channels=3)
    toc_unc = _get_image(directory, "toc_unc", num_channels=3)
    mask = _get_image(directory, "mask", num_channels=1)[..., 0]

    mask = mask > 0

    norm_xyz = normals_utils.rgb2xyz(norm_rgb)
    norm_unc = norm_unc / 255.0

    return {'norm_rgb': norm_rgb, 'toc_rgb': toc_rgb, 'toc_unc': toc_unc, 'norm_unc': norm_unc, 'mask': mask, 'norm_xyz': norm_xyz}

def _load_calibration_data(directory: Path):
    filepath = directory / "colmap.json"

    if not filepath.exists():
        return {}

    with open(filepath) as f:
        data = json.load(f)

    return {
        'R': np.array(data['R']).T, # Transpose to convert from COLMAP to PyTorch3D coordinate system.
        'T': np.array(data['T']),
        'cx': data['cx'],
        'cy': data['cy'],
        'f': data['f'],
        'k': data['k'],
        'width': data['width'],
        'height': data['height']
    }


def _get_image(directory:Path, filename:str, extensions:[str]=('png',), num_channels:int=3):
    """Load in an image from a directory."""

    filepath = None
    for ext in extensions:
        path = directory / f"{filename}.{ext}"
        if path.exists():
            filepath = path
            break

    if filepath is None:
        raise FileNotFoundError(f"Could not find {filename} in {directory}")

    img = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)

    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA if img.shape[-1] == 4 else cv2.COLOR_BGR2RGB)
    return img.astype(np.float32)[..., :num_channels] / 255.0
