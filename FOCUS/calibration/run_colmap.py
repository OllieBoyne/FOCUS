"""Run COLMAP Bundle Adjustment on a set of images."""

from pathlib import Path
import os
import subprocess
from tqdm import tqdm
import json

from FOCUS.calibration.colmap_format_conversion import colmap2pytorch3d
from FOCUS.calibration.custom_matches_colmap import toc_matches_to_database

ACCEPTED_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')

def _check_colmap(colmap_exe: str = 'colmap'):
    try:
        subprocess.check_call([colmap_exe, 'help'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
    except FileNotFoundError:
        raise FileNotFoundError(f'`{colmap_exe}` does not run. Make sure COLMAP is installed, and either added to PATH, or the `colmap_exe` argument points to the correct location.')

def run_colmap(image_dir: Path, output_dir: Path, colmap_exe: str = 'colmap',
               predictions_folder: Path = None,
               num_correspondences: int = 2500) -> None:
    """
    :param image_dir: Directory containing images.
    :param output_dir: Output directory for COLMAP data.
    :param colmap_exe: Path to COLMAP executable.
    :param predictions_folder: If given, use these for custom matches. Otherwise, use COLMAP feature extraction & matching.
    :param num_correspondences: For custom matches, how many correspondences to use.
    """

    _check_colmap(colmap_exe)

    workspace_dir = output_dir / 'colmap'
    workspace_dir.mkdir(exist_ok=True)

    database_path = os.path.join(workspace_dir, 'database.db')

    sparse_dir = workspace_dir / 'sparse'
    sparse_dir.mkdir(exist_ok=True)

    img_ids = []

    commands = {}
    if predictions_folder is not None:
        # Use custom matches.
        img_ids = toc_matches_to_database(predictions_folder, workspace_dir, num_correspondences=num_correspondences)
        image_dir = predictions_folder # Switch to image_dir being the predicted images (so that sizing is consistent
                                       # with predictions).

    else:

        img_ids = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(ACCEPTED_IMAGE_EXTENSIONS)]

        commands['feature_extractor'] = {
                'image_path': image_dir,
                'database_path': database_path,
                'ImageReader.single_camera': '1'
            }
        commands['exhaustive_matcher'] = {'database_path': database_path}

    commands['mapper'] = {'database_path': database_path, 'image_path': image_dir, 'output_path': sparse_dir,
               'Mapper.ba_refine_focal_length': '1'}

    with tqdm(commands.items()) as progress_bar:
        for method, command in progress_bar:
            progress_bar.set_description(f'COLMAP: Running `{method}`')

            args = [colmap_exe, method]
            for k, v in command.items():
                args += [f'--{k}', str(v)]

            # Run subprocess check call, saving output to log file.
            logfile = workspace_dir / f'{method}_log.txt'
            subprocess.check_call(args, stdout=open(logfile, 'w'), stderr=subprocess.STDOUT, shell=True)

    # Convert to PyTorch3D format
    output_data = colmap2pytorch3d(str(sparse_dir))

    with open(output_dir / 'colmap.json', 'w') as f:
        json.dump(output_data, f)

    # Export per view.
    failed_views = []
    for view in img_ids:
        view_data = {**output_data['camera']}

        for i in output_data['images']:
            if i['pth'].startswith(view):
                view_data['R'] = i['R']
                view_data['T'] = i['T']
                view_data['C'] = i['C']
                break
        else:
            failed_views.append(view)
            continue

        view_dir = output_dir / view
        view_dir.mkdir(exist_ok=True)
        with open(view_dir / f'colmap.json', 'w') as f:
            json.dump(view_data, f)

    if failed_views:
        raise ValueError(f'Failed to calibrate {len(failed_views)} / {len(img_ids)} images. '
                      'Try increasing --num_colmap_matches.')

