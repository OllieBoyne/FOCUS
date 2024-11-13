"""Given an input video, produce videos of TOC, normals etc."""

import argparse
import tempfile
import cv2
from FOCUS.toc_prediction.predict import predict_toc
import numpy as np
import imageio
from pathlib import Path
from collections import defaultdict

parser = argparse.ArgumentParser(description="Predict TOC from video.")

parser.add_argument("--video_path", type=Path, help="Path to video file.")
parser.add_argument("--output_folder", type=Path, help="Path to output folder.")
parser.add_argument("--model_path", type=Path, help="Path to predictor model.", default='data/toc_model/resnet50_v12_t=44/model_best.pth')

parser.add_argument('--fps', type=int, default=60, help='Frames per second of output video.')

FTYPES = ('rgb', 'toc', 'normal', 'norm_unc', 'toc_unc', 'mask')

def run(args):

    args.output_folder.mkdir(exist_ok=True, parents=True)

    frames = imageio.mimread(args.video_path, memtest=False)
    out_frames = defaultdict(list)

    with tempfile.TemporaryDirectory() as tempdir:
        keys = [f'{i:06d}' for i in range(len(frames))]
        predict_toc(imgs=frames, model=args.model_path, out_dir=Path(tempdir),
                                 img_keys=keys, device=None)

        for key in keys:
            for ftype in FTYPES:
                frame = cv2.imread(f'{tempdir}/{key}/{ftype}.png')
                out_frames[ftype].append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    for ftype in FTYPES:
        imageio.mimwrite(args.output_folder / f'{ftype}.mp4', out_frames[ftype], fps=args.fps)

if __name__ == "__main__":
    args = parser.parse_args()
    run(args)