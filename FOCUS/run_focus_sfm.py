import argparse

from FOCUS.fusion.fuse import fuse
from FOCUS.data.dataset import load_views

from pathlib import Path

parser = argparse.ArgumentParser(description="Fuse a set of views into a point cloud.")
parser.add_argument("--view_dir", type=Path, help="Path to views directory.")
parser.add_argument("--output_folder", type=Path, help="Path to output folder.")

# TODO: Add hyperparams

if __name__ == "__main__":
    args = parser.parse_args()

    views = load_views(args.view_dir)
    fuse(views, args.output_folder)