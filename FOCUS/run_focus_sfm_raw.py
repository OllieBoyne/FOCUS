"""Run FOCUS-SfM from raw images, no other input."""

import argparse
import sys

from FOCUS.fusion import fuse
from FOCUS.data.dataset import load_views
from FOCUS.data import io

from FOCUS.toc_prediction import predict as predict_toc_module
from FOCUS.calibration import run_colmap
import subprocess
import shutil
import os

from pathlib import Path

parser = argparse.ArgumentParser(description="Fuse a set of views into a point cloud.")
parser.add_argument("--img_dir", type=Path, help="Path to directory of images.")
parser.add_argument("--output_folder", type=Path, help="Path to output folder.")

# From video.
parser.add_argument('--video_path', type=Path, default=None, help='Path to video file.')
parser.add_argument('--num_frames', type=int, help='Number of frames to extract from video (-1 = all).', default=20)

parser.add_argument('--colmap_exe', default='colmap', help='Path to COLMAP executable.')
parser.add_argument('--toc_model_path', type=Path, help='Path to predictor model.', default='data/toc_model/resnet50_v12_t=44/model_best.pth')

# TODO: Add hyperparams

def _frames_from_video(args):

    # Check FFMPEG exists.
    if shutil.which('ffmpeg') is None:
        raise FileNotFoundError("FFMPEG not found. Please install FFMPEG to extract frames from video.")

    image_dir = args.output_folder / 'frames'
    image_dir.mkdir(exist_ok=True, parents=True)

    frame_count_cmd = f"ffmpeg -i {args.video_path} -map 0:v:0 -c copy -f null - 2>&1 | grep 'frame=' | awk '{{print $2}}'"
    total_frames = int(os.popen(frame_count_cmd).read().strip())
    desired_frames = args.num_frames if args.num_frames != -1 else total_frames
    interval = max(1, round(total_frames / desired_frames))

    args = ['ffmpeg', '-i', str(args.video_path),
            "-vf", f"select='not(mod(n,{interval}))'",
            '-vsync', 'vfr', f'{image_dir}/%06d.png']

    subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # If too many frames added, remove some.
    output_frames = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    for frame in output_frames[desired_frames:]:
        os.remove(image_dir / frame)


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