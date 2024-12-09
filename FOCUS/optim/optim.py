"""FOCUS-O, optimization of FIND model to TOC images, implementation."""

import torch
import numpy as np

import trimesh
from time import perf_counter

from tqdm import tqdm

from FOCUS.utils.torch_utils import get_device
from FOCUS.data.view import View
from FOCUS.utils import camera, sampler, geometry

from FOCUS.matching import match
from FOCUS.fusion import fused_point_cloud
from FOCUS.utils import triangulation

from FOCUS.optim.hyperparameters import OptimHyperparameters

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils", "FOUND"))
from FOCUS.utils.FOUND.FOUND import model as found_model


from matplotlib import pyplot as plt
import json

from pathlib import Path
from pytorch3d.transforms import euler_angles_to_matrix, Transform3d

# TODO: Handle footedness.

def _trimesh_transform_to_pytorch3d(T):

	scale, shear, angles, translate, persp = trimesh.transformations.decompose_matrix(T)

	# Trimesh returns as sxyz, need to convert to rxyz for pytorch3d.
	euler_rot = trimesh.transformations.euler_from_matrix(trimesh.transformations.euler_matrix(*angles, 'sxyz'), 'rxyz')

	return translate, scale, np.array(euler_rot)

def _register(views, cameras, template_verts: np.ndarray):
	"""For cases in which the cameras are not in world space, initialize a good registration for the mesh."""
	correspondences = match.find_matches(views, num_correspondences=500,
										 max_dist=0.002,
										 subpixel_scaling=8)

	point_cloud = fused_point_cloud.FusedPointCloud(correspondences, len(views))
	point_cloud.triangulate(triangulation.cameras_to_projection_matrix(cameras).cpu())
	point_cloud.remove_outliers(neighbours=50, std_ratio=1.0)

	min_bounds, max_bounds = np.min(template_verts, axis=0), np.max(template_verts, axis=0)

	source_points = point_cloud.toc * (max_bounds - min_bounds) + min_bounds
	target_points = point_cloud.points_3d

	T, a, cost = trimesh.registration.procrustes(source_points, target_points, reflection=True)

	translate, scale, euler = _trimesh_transform_to_pytorch3d(T)

	return {'translate': translate, 'scale': np.abs(np.mean(scale)), 'euler': euler, 'source_points': source_points, 'target_points': target_points}

def _sample_views(views, num_samples_per_view, device=None):
	"""Sample points on the TOC images."""
	pixel_values = []
	toc_values = []
	toc_unc_values = []
	for view in views:
		mask_batch = torch.from_numpy(view.mask).unsqueeze(0)
		samples = sampler.samples_in_mask(mask_batch, num_samples_per_view)

		toc_image_samples = sampler.sample_image(torch.from_numpy(view.toc_rgb).unsqueeze(0), samples)
		toc_unc_image_samples = sampler.sample_image(torch.from_numpy(view.toc_unc).unsqueeze(0), samples)

		pixel_values.append(samples)
		toc_values.append(toc_image_samples)
		toc_unc_values.append(toc_unc_image_samples)

	pixel_values = torch.concatenate(pixel_values).to(device)
	toc_values = torch.concatenate(toc_values).to(device)
	toc_unc_values = torch.concatenate(toc_unc_values).to(device)

	toc_values.requires_grad = True  # Necessary for uncertainty propagation.

	return pixel_values, toc_values, toc_unc_values

def optim(views: list[View], output_folder: Path, hyperparameters: OptimHyperparameters = OptimHyperparameters()):

	time_start = perf_counter()

	device = get_device()
	model = found_model.FIND("data/find_nfap").to(device)

	output_folder.mkdir(exist_ok=True, parents=True)

	template_verts = model.model.template_verts[0]
	min_bounds, max_bounds = torch.min(template_verts, dim=0)[0], torch.max(template_verts, dim=0)[0]

	cameras =  camera.Camera(
		R=torch.stack([torch.from_numpy(v.R) for v in views]).float(),
		T=torch.stack([torch.from_numpy(v.T) for v in views]).float(),
		size=torch.stack([torch.tensor(v.image_shape) for v in views]),
		focal_length=torch.stack([torch.tensor([v.f]) for v in views]),
	).to(device)

	# Precompute pixel transform to save time.
	project_transform = cameras.get_world_to_pix_transform()

	# Initialize registration parameters with quick alignment to triangulated points.
	reg = _register(views, cameras, template_verts.cpu().detach().numpy())
	with torch.no_grad():
		model.trans.data = torch.tensor(reg['translate'], device=device, dtype=torch.float32).unsqueeze(0)
		model.scale.data *= reg['scale']
		model.rot.data = torch.tensor(reg['euler'], device=device, dtype=torch.float32).unsqueeze(0)

	def _optimize(optim, num_iters):
		losses = []

		pixel_values, toc_values, toc_unc_values = _sample_views(views, hyperparameters.num_samples_per_view, device=device)

		with tqdm(np.arange(num_iters)) as tqdm_it:
			for i in tqdm_it:

				optim.zero_grad()

				# Convert to un-normalized FIND space.
				x = toc_values * (max_bounds - min_bounds) + min_bounds

				x_flat = x.reshape(1, -1, 3)
				res = model.model(x_flat, shapevec=model.shapevec, posevec=model.posevec,
								  texvec=model.texvec)

				offsets = res['disp']

				euler_rot = model.rot
				R = euler_angles_to_matrix(euler_rot, 'XYZ')

				T = Transform3d(device=model.scale.device).scale(model.scale).rotate(R).translate(model.trans)
				X = T.transform_points(x_flat + offsets)
				X = X.reshape(len(views), -1, 3)

				reproj_points = project_transform.transform_points(X, eps=1e-6)[..., :2]

				# Calculate Jacobian
				J = torch.zeros(len(views), hyperparameters.num_samples_per_view, 2, 3, device=toc_values.device, dtype=toc_values.dtype)

				# Set up gradients for each output component
				for i in range(2):
					grad_outputs = torch.zeros_like(reproj_points)  # Initialize gradient outputs
					grad_outputs[:, :, i] = 1  # Set 1 for the current dimension i

					# Compute gradients with respect to toc_values
					J_i = \
					torch.autograd.grad(outputs=reproj_points, inputs=toc_values, grad_outputs=grad_outputs, retain_graph=True,
										create_graph=False)[0]

					# Assign the computed gradients to the appropriate slice of the Jacobian matrix
					J[:, :, i, :] = J_i


				covariance_matrix_toc = torch.diag_embed(toc_unc_values ** 2)
				covariance_matrix_reproj_points = J @ covariance_matrix_toc @ J.transpose(-1, -2)
				reproj_points_variance = torch.diagonal(covariance_matrix_reproj_points, dim1=-2, dim2=-1)

				# Normalize all to image space.
				reproj_error = reproj_points - pixel_values

				image_size = torch.tensor(views[0].image_shape, device=reproj_points.device)
				reproj_points_variance_normalized = reproj_points_variance / (image_size ** 2)

				reproj_error_normalized = reproj_error / image_size.unsqueeze(0)

				if hyperparameters.use_uncertainty:
					loss = torch.norm(reproj_error_normalized / (reproj_points_variance_normalized.detach() ** .5), dim=-1).mean()
					loss = 0.01 * loss  # Weighting.

				else:
					loss = torch.norm(reproj_error_normalized, dim=-1).mean()
				loss.backward()

				losses.append(loss.item())

				optim.step()

				tqdm_it.set_description(f"Reproj error: {torch.norm(reproj_error, dim=-1).mean().item():.2f} pix.")

		return losses


	optimizers = [torch.optim.SGD(model.params['reg'], lr=0.002),
				  torch.optim.Adam(model.params["deform"] + model.params['reg'], lr=1e-3),
				  ]

	num_iters = [500, 500]

	its_so_far = 0
	for i in range(len(optimizers)):
		losses = _optimize(optimizers[i], num_iters[i])
		plt.semilogy(np.arange(num_iters[i]) + its_so_far, losses)
		its_so_far += num_iters[i]

	final_model = model()
	out_mesh = geometry.pytorch3d_meshes_to_trimesh(final_model)
	out_mesh.export(output_folder / "mesh.obj")

	time_end = perf_counter()
	metadata = {'runtime (s)': time_end - time_start}
	with open(output_folder / "metadata.json", 'w') as f:
		json.dump(metadata, f)

	hyperparameters.save(output_folder / "hyperparameters.json")

	plt.savefig(output_folder / "loss.png")

if __name__ == '__main__':
	from FOCUS.data.dataset import load_views
	# v = load_views(Path('data/dummy_data_pred'))
	# v = load_views(Path('data/dummy_data/0035_mono3d_v11_t=56'))
	v = load_views(Path('/Users/ollie/Documents/repos/phd-projects/FOCUS/exp/tmp/focus_o_demo'))
	o = Path("exp/tmp/focus_o_demo")
	h = OptimHyperparameters(is_world_space=False)
	optim(v, o, h)