"""Render fitting result from each viewpoint using Blender(synth)"""

import os, sys
import blendersynth as bsyn
from argparse import ArgumentParser
from tempfile import mkdtemp
import shutil

parser = ArgumentParser(description="Render fitting result from each viewpoint using Blender(synth)")
parser.add_argument("--input_directory", type=str, help="Path to the input directory")

# TODO: Transfer this functionality into BlenderSynth
if not bsyn.is_blender_running():
    args = parser.parse_args()

else:
    argv = sys.argv[sys.argv.index('--')+1:]
    args, _ = parser.parse_known_args(argv)

bsyn.run_this_script(open_blender=False, **vars(args))

# Note this has to be called after run_this_script due to sys path handling.
from FOCUS.data.dataset import load_views
from pathlib import Path

import mathutils

# Camera initial rotation = [180, 0, 180] - to match PyTorch3D format.
p2b = pytoch3d_2_blender_rotmat = mathutils.Matrix([
    [-1, 0, 0],
    [0, 1, 0],
    [0, 0, -1]
])

def main(input_directory):

    bsyn.render.set_cycles_samples(10)

    # Set up world lighting
    bsyn.world.set_color((1.0, 1.0, 1.0))

    mesh_path = os.path.join(input_directory, 'mesh.obj')
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh {mesh_path} does not exist.")

    mesh = bsyn.Mesh.from_obj(mesh_path)
    mesh.rotation_euler = (0, 0, 0)

    views = load_views(Path(input_directory))

    H, W, _ = views[0].rgb.shape
    bsyn.render.set_resolution(W, H)

    cameras = []

    tmp_output_dir = mkdtemp()

    comp = bsyn.Compositor()

    comp.define_output('Image', tmp_output_dir, file_name='mesh')

    for i, view in enumerate(views):

        camera = bsyn.Camera.create(view.key)

        R = view.calibration_data['R']
        T = view.calibration_data['T']
        C = view.calibration_data['C']

        focal_length_pixels = view.calibration_data['f']
        sensor_width = camera.object.data.sensor_width
        focal_length_mm = focal_length_pixels * sensor_width / H
        camera.focal_length = focal_length_mm

        rot_mat = mathutils.Matrix(R.tolist()) @ p2b

        camera.location = C
        camera.rotation_euler = rot_mat.to_euler('XYZ')

        cameras.append(camera)

    comp.render(camera=cameras)

    # Move to output directories.
    for view in views:
        src_path = os.path.join(tmp_output_dir, f"{view.key}_mesh.png")
        dst_path = os.path.join(input_directory, view.key, "mesh.png")
        shutil.copy(Path(src_path), Path(dst_path))



if __name__ == "__main__":
    main(input_directory=args.input_directory)
