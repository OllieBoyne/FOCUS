"""Run FOCUS-SfM from raw images, no other input."""

import argparse

from FOCUS.fusion import fuse
from FOCUS.data.dataset import load_views
from FOCUS.data import io

from FOCUS.toc_prediction import predict as predict_toc_module
from FOCUS.calibration import run_colmap
import subprocess
import shutil
import os
import sys
from tqdm import tqdm
from contextlib import contextmanager

from pathlib import Path

parser = argparse.ArgumentParser(description="Fuse a set of views into a point cloud.")
parser.add_argument('--make_predictions', action='store_true', help='Predictions do not exist: make predictions for TOC and normals.')

parser.add_argument("--source_folder", type=Path, help="Directory of source images/predictions.")

parser.add_argument("--output_folder", type=Path, help="Target output folder.")
parser.add_argument("--overwrite", action='store_true', help="Overwrite output folder if it exists.")

# Use these for extracting from video.
parser.add_argument('--video_path', type=Path, default=None, help='Path to video file.')
parser.add_argument('--num_frames', type=int, help='Number of frames to extract from video (-1 = all).', default=20)

parser.add_argument('--colmap_exe', default='colmap', help='Path to COLMAP executable.')
parser.add_argument('--toc_model_path', type=Path, help='Path to predictor model.', default='data/toc_model/resnet50_v12_t=44/model_best.pth')

parser.add_argument('--render', action='store_true', help='Render the output meshes.')
parser.add_argument('--produce_videos', action='store_true', help='Produce videos of TOC, normals etc.')

render_meshes_script = 'FOCUS/vis/render_meshes.py'
produce_videos_script = 'FOCUS/vis/video_to_predictions.py'

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
                              predictions_folder=args.output_folder)

    else:
        # Copy over predictions.
        shutil.copytree(source_folder, args.output_folder, dirs_exist_ok=True)

    with runner('Loader', 'Loading views'):
        views = load_views(args.output_folder)

    # Assume calibration is world space if using existing predictions, not otherwise.
    hyperparams = fuse.FusionHyperparameters(is_world_space=not args.make_predictions)

    with runner('FOCUS-SfM', 'Running FOCUS-SfM'):
        fuse.fuse(views, args.output_folder, hyperparameters=hyperparams)

    if args.render:
        with runner('Render', 'Rendering meshes'):
            subprocess.run([sys.executable, render_meshes_script, '--input_directory', str(args.output_folder)],
                           stdout=runner.logfile)

    if args.produce_videos:
        with runner('Videos', 'Producing videos'):
            subprocess.run([sys.executable, produce_videos_script, '--predictions_folder', str(args.output_folder),
                            '--output_folder', str(args.output_folder / 'videos')])
