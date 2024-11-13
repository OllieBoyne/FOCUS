from pathlib import Path
import os
import numpy as np
import cv2

IMAGE_FORMATS = ('.png', '.jpg', '.jpeg')

def load_images_from_dir(directory: Path, image_formats: tuple[str] = IMAGE_FORMATS) -> (np.ndarray, list[str]):
    imgs, keys = [], []
    for f in os.listdir(directory):
        if f.endswith(image_formats):
            key = os.path.splitext(f)[0]
            img = cv2.cvtColor(cv2.imread(os.path.join(directory, f)), cv2.COLOR_BGR2RGB)
            imgs.append(img)
            keys.append(key)

    return np.array(imgs), keys