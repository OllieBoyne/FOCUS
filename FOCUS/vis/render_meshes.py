"""Render fitting result from each viewpoint using Blender(synth)"""

import os
import blendersynth as bsyn

parser = bsyn.ArgumentParser(description="Render fitting result from each viewpoint using Blender(synth)")
parser.add_argument("--input_directory", type=str, help="Path to the input directory")
parser.add_argument('--use_color', type=int, default=1, help='1 for use vertex color, 0 otherwise.')
args = parser.parse_args()

bsyn.run_this_script(open_blender=False, **vars(args))

# Note this has to be called after run_this_script due to sys path handling.
from FOCUS.data.dataset import load_views
from pathlib import Path
import numpy as np
import cv2

import mathutils

# Camera initial rotation = [180, 0, 180] - to match PyTorch3D format.
p2b = pytoch3d_2_blender_rotmat = mathutils.Matrix([
    [-1, 0, 0],
    [0, 1, 0],
    [0, 0, -1]
])

def main(input_directory, use_color=True, out_key='mesh_color'):
    """Render all views, save to output directory.

    :param input_directory: Path to the input directory.
    :param use_color: If True, use vertex-color if it exists.
    :param out_key: Output key for the rendered mesh.
    """

    bsyn.render.set_cycles_samples(2)

    # Set up world lighting
    bsyn.world.set_transparent(True)
    bsyn.world.set_color((1.0, 1.0, 1.0))
    bsyn.world.set_intensity(1.0 if use_color else 0.7)

    mesh_path = os.path.join(input_directory, 'mesh.obj')
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh {mesh_path} does not exist.")

    mesh = bsyn.Mesh.from_obj(mesh_path)
    mesh.rotation_euler = (0, 0, 0)


    # Default color.
    mesh.material.shader.inputs['Base Color'].default_value = (0.1, 0.3, 0.85, 1.0)

    # Assign vertex-color if it exists
    if use_color:
        if 'Color' in mesh.obj.data.attributes:
            new_node = mesh.material.nodes.new('ShaderNodeVertexColor')
            new_node.layer_name = 'Color'
            mesh.material.links.new(new_node.outputs['Color'], mesh.material.shader.inputs['Base Color'])

    # To ensure consistent lighting, we will scale the whole scene to fit the average dimension of the mesh to 1m.
    sf = 1 / np.mean(mesh.dimensions)
    mesh.scale = (sf, sf, sf)
    mesh.location = sf * mesh.location

    views = load_views(Path(input_directory))

    H, W, _ = views[0].rgb.shape
    bsyn.render.set_resolution(W, H)

    cameras = []

    comp = bsyn.Compositor()

    comp.define_output('Image', name='mesh')

    for i, view in enumerate(views):

        camera = bsyn.Camera.create(view.key)

        R = view.calibration_data['R']
        C = view.calibration_data['C'] * sf

        focal_length_pixels = view.calibration_data['f']
        sensor_width = camera.object.data.sensor_width
        focal_length_mm = focal_length_pixels * sensor_width / H
        camera.focal_length = focal_length_mm

        rot_mat = mathutils.Matrix(R.tolist()) @ p2b

        camera.location = C
        camera.rotation_euler = rot_mat.to_euler('XYZ')

        cameras.append(camera)

    # Spotlight, for non-color only.
    # Pick the central most camera, add a spotlight to it.
    cameras_centroid = mathutils.Vector(np.mean([c.location for c in cameras], axis=0))
    closest_camera_idx = min(range(len(cameras)), key=lambda i: (cameras[i].location - cameras_centroid).length)

    if not use_color:
        light = bsyn.Light.create('SPOT', intensity=200)
        constr = light.object.constraints.new(type='COPY_TRANSFORMS')
        constr.target = cameras[closest_camera_idx].object

        light.object.data.spot_size = 1.22 # ~70 degrees

    render_result = comp.render(camera=cameras)

    # Move to output directories.
    for view in views:
        src_path = render_result.get_render_path('mesh', view.key)
        dst_path = os.path.join(input_directory, view.key, f"{out_key}.png")

        # Load and place over the original image.
        original_image = (view.rgb * 255).astype(np.uint8)
        rendered_image = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
        rendered_image = cv2.cvtColor(rendered_image, cv2.COLOR_BGRA2RGBA)

        new_image = original_image.copy()
        mask = rendered_image[..., -1] > 0
        new_image[mask] = rendered_image[..., :3][mask]

        cv2.imwrite(dst_path, cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA))


if __name__ == "__main__":
    main(input_directory=args.input_directory, use_color=args.use_color, out_key='mesh_color' if args.use_color else 'mesh_no_color')
