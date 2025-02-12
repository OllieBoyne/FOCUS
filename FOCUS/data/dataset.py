"""A dataset which loads in images, inference results, and calibration into a single,
    unified format."""

import cv2
from pathlib import Path
import numpy as np
import json
import os

from FOCUS.data.view import View


def load_view(directory: Path) -> View:
    """Expects a directory with all relevant information for a single image."""

    image_data = _load_image_data(directory)
    inference_data = _load_inference_data(directory)
    calibration_data = _load_calibration_data(directory)

    key = directory.name

    return View(**image_data, **inference_data, calibration_data=calibration_data, key=key)

def load_views(directory: Path, ignore_list=('colmap', 'frames', 'videos', 'logs'), min_coverage: float = 0.02) -> [View]:
    """Load in all views from a directory.

    Args:
        directory: Directory containing all views.
        ignore_list: List of directories to ignore.
        min_coverage: Minimum coverage (mask area / image area) of a view to be included.
    """
    idx = 0
    views = []
    rejected_views = []
    for file in sorted(os.listdir(directory)):
        if os.path.isdir(directory / file) and file not in ignore_list:
            view = load_view(directory / file)

            if view.mask_coverage < min_coverage:
                rejected_views.append(view)
                continue

            view.idx = idx
            views.append(view)
            idx += 1

    if len(rejected_views) > 0:
        print(f"Rejected {len(rejected_views)} views with coverage < {min_coverage}.")


    return views

def _load_image_data( directory):
    return {'rgb': _get_image(directory, "rgb")}

def _load_inference_data(directory):
    norm_rgb = _get_image(directory, "normal", num_channels=3)
    norm_unc = _get_image(directory, "norm_unc", num_channels=1)
    toc_rgb = _get_image(directory, "toc", num_channels=3)
    toc_unc = _get_image(directory, "toc_unc", num_channels=3)
    mask = _get_image(directory, "mask", num_channels=1)[..., 0]

    mask = mask > 0

    return {'norm_rgb': norm_rgb, 'toc_rgb': toc_rgb, 'toc_unc': toc_unc, 'norm_unc': norm_unc, 'mask': mask}

def _load_calibration_data(directory: Path):
    filepath = directory / "colmap.json"

    if not filepath.exists():
        return {}

    with open(filepath) as f:
        data = json.load(f)

    return {
        'R': np.array(data['R']).T, # Transpose to convert from COLMAP to PyTorch3D coordinate system.
        'T': np.array(data['T']),
        'C': np.array(data['C']),
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
