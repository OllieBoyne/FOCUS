"""Run COLMAP Bundle Adjustment on a set of images."""

from pathlib import Path
import os
import subprocess
from tqdm import tqdm
import json

from FOCUS.calibration.colmap_format_conversion import colmap2pytorch3d

ACCEPTED_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')

def _check_colmap(colmap_exe: str = 'colmap'):
    try:
        subprocess.check_call(['colmap', 'help'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        raise FileNotFoundError(f'`{colmap_exe}` does not run. Make sure COLMAP is installed, and either added to PATH, or the `colmap_exe` argument points to the correct location.')

def run_colmap(image_dir: Path, output_dir: Path, colmap_exe: str = 'colmap'):

    _check_colmap(colmap_exe)

    workspace_dir = output_dir / 'colmap'
    workspace_dir.mkdir(exist_ok=True)

    database_path = os.path.join(workspace_dir, 'database.db')

    sparse_dir = workspace_dir / 'sparse'
    sparse_dir.mkdir(exist_ok=True)


    commands = {
        'feature_extractor': {
            'image_path': image_dir,
            'database_path': database_path,
            'ImageReader.single_camera': '1'
        },
        'exhaustive_matcher': {'database_path': database_path},
        		'mapper': {'database_path': database_path, 'image_path': image_dir, 'output_path': sparse_dir,
               'Mapper.ba_refine_focal_length': '1'},
    }

    with tqdm(commands.items()) as progress_bar:
        for method, command in progress_bar:
            progress_bar.set_description(f'COLMAP: Running `{method}`')

            args = [colmap_exe, method]
            for k, v in command.items():
                args += [f'--{k}', v]

            # Run subprocess check call, saving output to log file.
            logfile = workspace_dir / f'{method}_log.txt'
            subprocess.check_call(args, stdout=open(logfile, 'w'), stderr=subprocess.STDOUT)

    # Convert to PyTorch3D format
    output_data = colmap2pytorch3d(str(sparse_dir))

    with open(output_dir / 'colmap.json', 'w') as f:
        json.dump(output_data, f)

    # Export per view.
    img_ids = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(ACCEPTED_IMAGE_EXTENSIONS)]
    for view in img_ids:
        view_data = {**output_data['camera']}

        for i in output_data['images']:
            if i['pth'].startswith(view):
                view_data['R'] = i['R']
                view_data['T'] = i['T']
                view_data['C'] = i['C']
                break
        else:
            raise ValueError(f'No matching image for view {view} found in COLMAP data.')

        view_dir = output_dir / view
        view_dir.mkdir(exist_ok=True)
        with open(view_dir / f'colmap.json', 'w') as f:
            json.dump(view_data, f)


if __name__ == '__main__':
    src = '/Users/ollie/Library/CloudStorage/OneDrive-UniversityofCambridge/FIND2D/data/Foot3D/mono3d_v11_t=56/0035'
    output_dir = Path('/Users/ollie/Documents/repos/phd-projects/FOCUS/data/dummy_data_pred')

    img_dir = os.path.join(src, 'rgb')
    run_colmap(img_dir, output_dir)

