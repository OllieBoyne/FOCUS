from pathlib import Path
import os
import numpy as np
import cv2

IMAGE_FORMATS = ('.png', '.jpg', '.jpeg')

def load_images_from_dir(directory: Path, image_formats: tuple[str] = IMAGE_FORMATS) -> (list[np.ndarray], list[str]):
    """Load all images from a directory."""
    imgs, keys = [], []
    for f in sorted(os.listdir(directory)):
        if f.endswith(image_formats):
            key = os.path.splitext(f)[0]
            img = cv2.cvtColor(cv2.imread(os.path.join(directory, f)), cv2.COLOR_BGR2RGB)
            imgs.append(img)
            keys.append(key)

    return imgs, keys