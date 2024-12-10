# Apply masks.

import cv2
import numpy as np
import os

out_dir = '../synth'

for f in os.listdir('mask'):

    _mask = None

    for ftype in ['mask', 'normals', 'rgb', 'toc']:

        os.makedirs(os.path.join(out_dir, ftype), exist_ok=True)

        img = cv2.imread(os.path.join(ftype, f))

        if ftype == 'mask':
            _mask = img.mean(axis=-1) > 200

        out_img = np.ones((img.shape[0], img.shape[1], 4), dtype=np.uint8) * 255
        out_img[:, :, :3] = img
        if ftype != 'rgb':
            out_img[:, :, 3] = 255 * _mask.astype(int)

        cv2.imwrite(os.path.join(out_dir, ftype, f), out_img)