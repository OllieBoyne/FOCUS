"""Given an input video, produce videos of TOC, normals etc."""

import argparse
import tempfile
import cv2
from FOCUS.toc_prediction.predict import predict_toc
from FOCUS.data.dataset import load_views
import numpy as np
import imageio
from pathlib import Path
from collections import defaultdict
import subprocess
import tempfile

parser = argparse.ArgumentParser(description="Predict TOC from video.")

parser.add_argument("--video_path", type=Path, help="Path to video file.")
parser.add_argument('--predictions_folder', type=Path, help="Existing folder containing predictions.")

parser.add_argument("--output_folder", type=Path, help="Path to output folder.")
parser.add_argument("--model_path", type=Path, help="Path to predictor model.", default='data/toc_model/resnet50_v12_t=44/model_best.pth')

parser.add_argument('--fps', type=int, default=60, help='Frames per second of output video.')

FTYPES = ('rgb', 'toc', 'normal', 'norm_unc', 'toc_unc', 'mask', 'mesh_no_color', 'mesh_color')
STACK_FTYPES = ('rgb', 'toc', 'normal', 'mesh_no_color', 'mesh_color')

def overlay_text_on_video(input_video, text, output_video):
    command = [
        "ffmpeg",
        "-i", input_video,
        "-vf", f"drawtext=text='{text}':fontcolor=white:fontsize=50:x=(w-text_w)/2:y=10",
        "-codec:a", "copy",
        output_video
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def hstack_videos(output_loc, videos):
    args = ['ffmpeg', '-y']
    for i, video in enumerate(videos):
        args += ['-i', video]

    args += ['-filter_complex', f"{''.join([f'[{i}:v]' for i in range(len(videos))])}hstack=inputs={len(videos)}[v]", '-map', '[v]', output_loc]

    subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # ALso convert to gif.
    gif_loc = output_loc.with_suffix('.gif')
    args = ['ffmpeg', '-y', '-i', output_loc, '-vf', 'fps=25,scale=640:-1', gif_loc]
    subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def run(args):

    args.output_folder.mkdir(exist_ok=True, parents=True)

    out_frames = defaultdict(list)

    if args.predictions_folder is None:
        frames = imageio.mimread(args.video_path, memtest=False)

        with tempfile.TemporaryDirectory() as tempdir:
            keys = [f'{i:06d}' for i in range(len(frames))]
            predict_toc(imgs=frames, model=args.model_path, out_dir=Path(tempdir),
                                     img_keys=keys, device=None)

            for key in keys:
                for ftype in FTYPES:
                    frame = cv2.imread(f'{tempdir}/{key}/{ftype}.png')
                    out_frames[ftype].append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    else:
        views = load_views(args.predictions_folder)
        keys = [v.key for v in views]

        for key in keys:
            for ftype in FTYPES:
                frame = cv2.imread(f'{args.predictions_folder}/{key}/{ftype}.png')
                if frame is None:
                    print(f'Error reading {args.predictions_folder}/{key}/{ftype}.png')
                    continue
                out_frames[ftype].append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    for ftype in FTYPES:
        imageio.mimwrite(args.output_folder / f'{ftype}.mp4', out_frames[ftype], fps=args.fps)

    # Stack files.
    temp_vids = []
    for ftype in STACK_FTYPES:
        temp_vid = tempfile.mktemp(suffix='.mp4')
        overlay_text_on_video(args.output_folder / f'{ftype}.mp4', ftype.upper(), temp_vid)
        temp_vids.append(temp_vid)

    hstack_videos(args.output_folder / 'stacked.mp4', temp_vids)

if __name__ == "__main__":
    args = parser.parse_args()
    run(args)