import cv2
import numpy as np

def resize_preserve_aspect(img, targ_size,
            interpolation=cv2.INTER_LINEAR, return_transform=False):
    """Scale an image to a target size,
    padding with zeros to preserve aspect ratio"""

    targ_W, targ_H = targ_size

    H, W = img.shape[:2]
    if H > W:
        new_H, new_W = targ_H, int(W * targ_H / H)
    else:
        new_H, new_W = int(H * targ_W / W), targ_W

    img = cv2.resize(img, (new_W, new_H), interpolation=interpolation)

    pad_H = (targ_H - new_H) // 2
    pad_W = (targ_W - new_W) // 2

    out_shape = (targ_H, targ_W) + img.shape[2:]
    out = np.zeros(out_shape, dtype=img.dtype)
    out[pad_H:pad_H+new_H, pad_W:pad_W+new_W] = img


    return out