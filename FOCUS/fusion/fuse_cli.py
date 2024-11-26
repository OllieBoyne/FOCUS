"""Run a fusion prediction on a folder. Expects a folder containing predictions on each view, and will output fusion
results into the same folder."""

from pathlib import Path
import argparse

from FOCUS.fusion import fuse
from FOCUS.data.dataset import load_views
from FOCUS.fusion.hyperparameters import FusionHyperparameters

parser = argparse.ArgumentParser(description="Fuse a set of views into a point cloud.")
parser.add_argument('--folder', type=Path, help='Folder containing views.')
FusionHyperparameters.add_to_argparse(parser)

def main(args: argparse.Namespace):
    hyperparams = FusionHyperparameters.from_args(args)
    views = load_views(args.folder)
    fuse.fuse(views, args.folder, hyperparameters=hyperparams)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)