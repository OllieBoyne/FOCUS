"""Utils for pixel sampling"""

import numpy as np
import torch
from torch.nn import functional as F


# TODO: tidy up to remove torch dependency.

def boundary_mask(mask: torch.BoolTensor):
	assert mask.ndim == 3, "Mask must be 3D, [B x H x W]"

	kernel = torch.ones(1, 1, 3, 3, device=mask.device)
	summed_neighbors = F.conv2d(mask.float().unsqueeze(1), kernel, padding=1)
	max_sum = kernel.sum()

	summed_neighbors = summed_neighbors.squeeze(1)
	boundary_pixels = (summed_neighbors < max_sum) & (mask > 0)

	return boundary_pixels

def samples_in_mask(mask: torch.Tensor, n_samples:int):
	"""Given a 2D mask, return the (x, y) positions of N samples inside the mask, subpixel.
	
	:param mask: [B x H x W] mask
	:param n_samples: number of samples to return
	"""
	B = mask.shape[0]

	if mask.dtype != torch.bool:
		mask = torch.gt(mask, 0.0)

	internal_mask = torch.bitwise_xor(mask, boundary_mask(mask)) # remove boundary pixels
	batched_samples = torch.zeros(B, n_samples, 2, device=mask.device).float()

	for b in range(B):
		non_zero_coords = torch.nonzero(internal_mask[b])
		idxs = torch.randint(0, non_zero_coords.shape[0], (n_samples,))
		selected_coords = non_zero_coords[idxs]
		noise = torch.rand(n_samples, 2).to(mask.device) - 0.5 # for subpixel sampling
		batched_samples[b] = selected_coords[:, [1, 0]] + noise

	return batched_samples

def sample_image(image: torch.Tensor, samples: torch.Tensor):
	"""Given (subpixel) samples (x,y), sample image at these points, return samples.

	:image: [B x H x W x C] image
	:samples: [B x N x 2] (x,y) coordinates in pixel space
	:returns: [B x N x C] sampled image

	Currently uses simple linear interpolation
	"""

	B, H, W = image.shape[:3]

	b = torch.arange(B)[:, None]
	x, y = samples.permute(2, 0, 1)

	# clip to image bounds
	x = torch.clip(x, 0, W - 1)
	y = torch.clip(y, 0, H - 1)

	xmin, ymin = torch.floor(x).long(), torch.floor(y).long()
	xmax, ymax = xmin + 1, ymin + 1

	# to manage edge effects
	xmaxclipped = torch.clip(xmax, 0, W - 1)
	ymaxclipped = torch.clip(ymax, 0, H - 1)

	q11 = image[b, ymin, xmin]
	q12 = image[b, ymin, xmaxclipped]
	q21 = image[b, ymaxclipped, xmin]
	q22 = image[b, ymaxclipped, xmaxclipped]


	fy1 = (xmax - x).unsqueeze(-1) * q11 + (x - xmin).unsqueeze(-1) * q12
	fy2 = (xmax - x).unsqueeze(-1) * q21 + (x - xmin).unsqueeze(-1) * q22
	fxy = (ymax - y).unsqueeze(-1) * fy1 + (y - ymin).unsqueeze(-1) * fy2

	return fxy

class Sampler():
	"""Class for sampling from a given image(s)"""
	
	def __call__(self, mask: torch.Tensor, imgs: dict, N:int = 1000, normalize:list=None) -> dict:
		"""
		Samples N points within the masks.
		Returns sampled values for each point in image

		:param mask: [N x H x W]
		:param imgs: dict of name : [N x H x W x C] images
		:param N: number of samples
		:param normalize: any inputs in `imgs` to normalize to unit length

		:returns: dict of name : [N x C] sampled images; pixel_values : [N x 2] sampled pixel values
		"""

		pixel_values = samples_in_mask(mask, N)

		sampled_imgs = {name: sample_image(img, pixel_values) for name, img in imgs.items()}
		sampled_imgs['pixel_values'] = pixel_values

		if normalize is not None:
			if not isinstance(normalize, list):
				normalize = [normalize]
				
			for name in normalize:
				if name in sampled_imgs:
					sampled_imgs[name] = torch.nn.functional.normalize(sampled_imgs[name], dim=-1)

		return sampled_imgs