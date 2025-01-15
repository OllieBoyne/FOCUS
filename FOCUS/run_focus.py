"""Run FOCUS-SfM from raw images, no other input."""

import argparse

import subprocess
import shutil
import os
import sys
from tqdm import tqdm
import cv2
from pathlib import Path

from FOCUS.data import io

from FOCUS.toc_prediction import predict as predict_toc_module
from FOCUS.calibration import run_colmap
from FOCUS.utils.image import resize_preserve_aspect

from FOCUS.fusion import fuse
from FOCUS.data.dataset import load_views
from FOCUS.fusion.hyperparameters import FusionHyperparameters

from FOCUS.optim import optim
from FOCUS.optim.hyperparameters import OptimHyperparameters

parser = argparse.ArgumentParser(description="Reconstruct a 3D model from views.")

parser.add_argument('--method', type=str, default='sfm', choices=['sfm', 'o'],
                    help='Method to use for reconstruction (sfm or o).')

parser.add_argument('--make_predictions', action='store_true', help='Predictions do not exist: make predictions for TOC and normals.')

parser.add_argument("--source_folder", type=Path, help="Directory of source images/predictions.")

parser.add_argument("--output_folder", type=Path, help="Target output folder.")
parser.add_argument("--overwrite", action='store_true', help="Overwrite output folder if it exists.")

# Use these for extracting from video.
parser.add_argument('--video_path', type=Path, default=None, help='Path to video file.')
parser.add_argument('--num_frames', type=int, help='Number of frames to extract from video (-1 = all).', default=20)

# COLMAP params.
parser.add_argument('--num_colmap_matches', type=int, default=2500, help='Number of matches to use for COLMAP.')
parser.add_argument('--colmap_exe', default='colmap', help='Path to COLMAP executable.')

parser.add_argument('--toc_model_path', type=Path, help='Path to predictor model.', default='data/toc_model/densedepth_toc_predictor.pth')

parser.add_argument('--render', action='store_true', help='Render the output meshes.')
parser.add_argument('--produce_videos', action='store_true', help='Produce videos of TOC, normals etc.')

method = parser.parse_known_args()[0].method

if method == 'o':
    OptimHyperparameters.add_to_argparse(parser)
else:
    FusionHyperparameters.add_to_argparse(parser)

render_meshes_script = 'FOCUS/vis/render_meshes.py'
produce_videos_script = 'FOCUS/vis/video_to_predictions.py'

def _frames_from_video(args):

    # Check FFMPEG exists.
    if shutil.which('ffmpeg') is None:
        raise FileNotFoundError("FFMPEG not found. Please install FFMPEG to extract frames from video.")

    # Check video exists.
    if not args.video_path.exists():
        raise FileNotFoundError(f"Video file {args.video_path} does not exist.")

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

    # Resize frames to target size.
    for frame in output_frames[:desired_frames]:
        img = cv2.imread(str(image_dir / frame))
        img = resize_preserve_aspect(img, (480, 640))
        cv2.imwrite(str(image_dir / frame), img)

class Runner:
    def __init__(self, logdir: Path):
        self.logdir = logdir
        self._original_stdout = sys.stdout
        self._pbar: tqdm = None
        self.logfile = None

    def __call__(self, title, desc):
        self._title = title
        self._desc = desc
        return self

    def __enter__(self):

        logfile = logdir / f'{self._title}.txt'
        self.logfile = open(logfile, 'w')

        sys.stdout = self.logfile
        # with  as pbar:
        self._pbar = tqdm([1], desc=self._desc)
        return self

    def __exit__(self, *args):
        self._pbar.update()
        self._pbar.close()
        sys.stdout = self._original_stdout
        self.logfile.close()


if __name__ == "__main__":
    args = parser.parse_args()

    source_folder = args.source_folder

    if args.output_folder.exists():
        if args.overwrite:
            shutil.rmtree(args.output_folder)
        else:
            raise FileExistsError(f"Output folder {args.output_folder} already exists. Run with --overwrite to force overwrite.")

    args.output_folder.mkdir(parents=True, exist_ok=True)

    logdir = args.output_folder / 'logs'
    logdir.mkdir(exist_ok=True)
    runner = Runner(logdir)

    if args.video_path is not None:
        assert args.make_predictions, "Cannot extract frames from video without making predictions. Use flag --make_predictions."
        with runner('Video', 'Extracting frames from video'):
            _frames_from_video(args)
        source_folder = args.output_folder / 'frames'

    if args.make_predictions:
        images, keys = io.load_images_from_dir(source_folder)

        predict_toc_module.predict_toc(images, args.toc_model_path,
                    args.output_folder, keys
        )

        run_colmap.run_colmap(source_folder, args.output_folder, colmap_exe=args.colmap_exe,
                              predictions_folder=args.output_folder,
                              num_correspondences=args.num_colmap_matches)

    else:
        # Copy over predictions.
        shutil.copytree(source_folder, args.output_folder, dirs_exist_ok=True)

    if args.method == 'o':
        with runner('Optim', 'Running Optim'):
            hyperparams = OptimHyperparameters.from_args(args, is_world_space=not args.make_predictions)
            views = load_views(args.output_folder)
            optim.optim(views, args.output_folder, hyperparameters=hyperparams)

    else:
        with runner('FOCUS-SfM', 'Running FOCUS-SfM'):
            hyperparams = FusionHyperparameters.from_args(args, is_world_space=not args.make_predictions)
            views = load_views(args.output_folder)
            fuse.fuse(views, args.output_folder, hyperparameters=hyperparams)

    if args.render:
        with runner('Render', 'Rendering meshes'):

            command = [sys.executable, render_meshes_script, '--input_directory', str(args.output_folder)]

            processes = [subprocess.Popen(command + ['--use_color', '0'], stdout=runner.logfile)]
            if method == 'sfm':
                processes.append(subprocess.Popen(command + ['--use_color', '1'], stdout=runner.logfile))

            # Wait for all rendering processes to complete
            for process in processes:
                process.wait()

    if args.produce_videos:
        with runner('Videos', 'Producing videos'):

            command = [sys.executable, produce_videos_script, '--predictions_folder', str(args.output_folder), '--output_folder', str(args.output_folder / 'videos')]

            if method == 'o':
                command += ['--no_mesh_color']

            subprocess.run(command)
