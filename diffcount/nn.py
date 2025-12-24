"""
Various utilities for neural networks.
"""

import math

import torch as th
import torch.nn as nn
from inspect import isfunction
import torch.nn.functional as F

VAE_DOWNSCALE_FACTOR = 8
TILED_BBOXES_PADDING_VALUE = -1.0

def disabled_train(self, mode=True):
	"""Overwrite model.train with this function to make sure train/eval mode
	does not change anymore."""
	return self


def count_params(model, verbose=False):
	total_params = sum(p.numel() for p in model.parameters())
	if verbose:
		print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
	return total_params


def expand_dims_like(x, y):
    while x.dim() != y.dim():
        x = x.unsqueeze(-1)
    return x


def exists(val):
	return val is not None


def default(val, d):
	if exists(val):
		return val
	return d() if isfunction(d) else d


def conv_nd(dims, *args, **kwargs):
	"""
	Create a 1D, 2D, or 3D convolution module.
	"""
	if dims == 1:
		return nn.Conv1d(*args, **kwargs)
	elif dims == 2:
		return nn.Conv2d(*args, **kwargs)
	elif dims == 3:
		return nn.Conv3d(*args, **kwargs)
	raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
	"""
	Create a linear module.
	"""
	return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
	"""
	Create a 1D, 2D, or 3D average pooling module.
	"""
	if dims == 1:
		return nn.AvgPool1d(*args, **kwargs)
	elif dims == 2:
		return nn.AvgPool2d(*args, **kwargs)
	elif dims == 3:
		return nn.AvgPool3d(*args, **kwargs)
	raise ValueError(f"unsupported dimensions: {dims}")


def zero_module(module):
	"""
	Zero out the parameters of a module and return it.
	"""
	for p in module.parameters():
		p.detach().zero_()
	return module


def scale_module(module, scale):
	"""
	Scale the parameters of a module and return it.
	"""
	for p in module.parameters():
		p.detach().mul_(scale)
	return module


def freeze(module):
	for param in module.parameters():
		param.requires_grad = False
	return module


def mean_flat(tensor):
	"""
	Take the mean over all non-batch dimensions.
	"""
	return tensor.mean(dim=list(range(1, len(tensor.shape))))


def possibly_vae_encode(x, vae=None, single_ch=False):
	if vae is not None:
		_, ch, _, _ = x.shape
		if ch == 1:
			x = x.repeat(1, 3, 1, 1)
			if single_ch:
				x[:, 1:, :, :] = -1.0
		x = vae.encode(x).latent_dist.sample() * vae.config.scaling_factor
	return x


def possibly_vae_decode(z, vae=None, clip_decoded=False):
	if vae is not None:
		z = vae.decode(z / vae.config.scaling_factor).sample
	if clip_decoded:
		z = z.clamp(-1, 1)
	return z


def torch_to(x, *args, **kwargs):
	if isinstance(x, th.Tensor):
		return x.to(*args, **kwargs)
	if isinstance(x, dict):
		return {k: torch_to(v, *args, **kwargs) for k, v in x.items()}
	if isinstance(x, (list, tuple)):
		return [torch_to(v, *args, **kwargs) for v in x]
	return x


def timestep_embedding(timesteps, dim, max_period=10000):
	"""
	Create sinusoidal timestep embeddings.

	:param timesteps: a 1-D Tensor of N indices, one per batch element.
					  These may be fractional.
	:param dim: the dimension of the output.
	:param max_period: controls the minimum frequency of the embeddings.
	:return: an [N x dim] Tensor of positional embeddings.
	"""
	half = dim // 2
	freqs = th.exp(
		-math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
	).to(device=timesteps.device)
	args = timesteps[:, None].float() * freqs[None]
	embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
	if dim % 2:
		embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
	return embedding


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
	

def box_cxcywh_to_xyxy(x):
	x_c, y_c, w, h = x.unbind(-1)
	b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
		 (x_c + 0.5 * w), (y_c + 0.5 * h)]
	return th.stack(b, dim=-1)


def fast_train_nms(density_map, sort=False, validate=False, custom_batch_thresh=None):
	B, C, _, _ = density_map.shape  # B, 1, H, W

	# maxpool instead of scikit local peak
	pooled = F.max_pool2d(density_map, 3, 1, 1)
	# medians over batch
	if validate:
		batch_thresh = th.max(density_map.reshape(B, -1), dim=-1).values.view(B, C, 1, 1) / 8
	else:
		batch_thresh = th.median(density_map.reshape(B, -1), dim=-1).values.view(B, C, 1, 1)

	if custom_batch_thresh is not None:
		batch_thresh = custom_batch_thresh

	# binary mask of selected boxes
	mask = (pooled == density_map) & (density_map > batch_thresh)
	return mask


def boxes_with_scores(density_map, tlrb, sort=False, validate=False, custom_batch_thresh=None):
	B, C, _, _ = density_map.shape  # B, 1, H, W
	mask = fast_train_nms(density_map=density_map, validate=validate, custom_batch_thresh=custom_batch_thresh)

	# need this for loop to have the same output structure
	# can be vectorized otherwise
	out_batch = []
	ref_points_batch = []
	for i in range(B):
		# select the masked density maps and box offsets
		bbox_scores = density_map[i, mask[i]]
		ref_points = mask[i].nonzero()[:, -2:]

		# normalize center locations
		bbox_centers = ref_points / th.tensor(mask.shape[2:], device=mask.device)

		# select masked box offsets, permute to keep channels last
		tlrb_ = tlrb[i].permute(1, 2, 0)
		bbox_offsets = tlrb_[mask[i].permute(1, 2, 0).expand_as(tlrb_)].reshape(-1, 4)

		# vectorised calculation of the boxes = [ref_points_transposed[1] / ...] in original
		sign = th.tensor([-1, -1, 1, 1], device=mask.device)
		bbox_xyxy = bbox_centers.flip(-1).repeat(1, 2) + sign * bbox_offsets

		# sort by bbox score if needed -- this matches the original
		if sort:
			perm = th.argsort(bbox_scores, descending=True)
			bbox_scores = bbox_scores[perm]
			bbox_xyxy = bbox_xyxy[perm]
			ref_points = ref_points[perm]

		# Discard degenerate boxes
		# valid = (
		# 	(bbox_xyxy[:, 2:] >= bbox_xyxy[:, :2]).all(dim=1) & 
		# 	(bbox_xyxy[:, :2] >= 0.0).all(dim=1) &
		# 	(bbox_xyxy[:, 2:] <= 1.0).all(dim=1)
		# )
		valid = (bbox_xyxy[:, 2:] >= bbox_xyxy[:, :2]).all(dim=1)
		bbox_xyxy = bbox_xyxy[valid]
		bbox_scores = bbox_scores[valid]
		ref_points = ref_points[valid]

		out_batch.append({
			"pred_boxes": bbox_xyxy.unsqueeze(0),
			"box_v": bbox_scores.unsqueeze(0)
		})
		ref_points_batch.append(ref_points.T)

	return out_batch, ref_points_batch, mask