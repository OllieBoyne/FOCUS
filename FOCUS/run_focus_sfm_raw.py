"""Run FOCUS-SfM from raw images, no other input."""

import argparse

from FOCUS.fusion import fuse
from FOCUS.data.dataset import load_views
from FOCUS.data import io

from FOCUS.toc_prediction import predict as predict_toc_module
from FOCUS.calibration import run_colmap
import cv2
from tqdm import tqdm

from pathlib import Path

parser = argparse.ArgumentParser(description="Fuse a set of views into a point cloud.")
parser.add_argument("--img_dir", type=Path, help="Path to directory of images.")
parser.add_argument("--output_folder", type=Path, help="Path to output folder.")

# From video.
parser.add_argument('--video_path', type=Path, default=None, help='Path to video file.')
parser.add_argument('--num_frames', type=int, help='Number of frames to extract from video.', default=20)

parser.add_argument('--colmap_exe', default='colmap', help='Path to COLMAP executable.')
parser.add_argument('--toc_model_path', type=Path, help='Path to predictor model.', default='data/toc_model/resnet50_v12_t=44/model_best.pth')

# TODO: Add hyperparams

def _frames_from_video(args):
    image_dir = args.output_folder / 'frames'
    image_dir.mkdir(exist_ok=True, parents=True)

    cap = cv2.VideoCapture(str(args.video_path))
    num_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = num_video_frames // args.num_frames
    with tqdm(total=args.num_frames, desc="Extracting frames") as pbar:
        for i in range(args.num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(str(image_dir / f'{i:06d}.png'), frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_step)
            pbar.update(1)

    cap.release()

if __name__ == "__main__":
    args = parser.parse_args()

    image_dir = args.img_dir

    if args.video_path is not None:
        _frames_from_video(args)
        image_dir = args.output_folder / 'frames'

    images, keys = io.load_images_from_dir(image_dir)

    predict_toc_module.predict_toc(images, args.toc_model_path,
                args.output_folder, keys
                )

    run_colmap.run_colmap(image_dir, args.output_folder, colmap_exe=args.colmap_exe,
                          predictions_folder=args.output_folder)

    views = load_views(args.output_folder)

    hyperparams = fuse.FusionHyperparameters(is_world_space=False)
    fuse.fuse(views, args.output_folder, hyperparameters=hyperparams)