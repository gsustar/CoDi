import math
import torch
import torch.nn.functional as F
import warnings

from torch import nn, einsum
from inspect import isfunction
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from transformers.models.deformable_detr.modeling_deformable_detr import (
	MultiScaleDeformableAttention,
	load_cuda_kernels, 
	multi_scale_deformable_attention, 
	MultiScaleDeformableAttentionFunction,
	DeformableDetrEncoder,
)
from transformers.utils import is_torch_cuda_available, is_ninja_available

from . import logger

def exists(val):
	return val is not None


def uniq(arr):
	return{el: True for el in arr}.keys()


def default(val, d):
	if exists(val):
		return val
	return d() if isfunction(d) else d


def max_neg_value(t):
	return -torch.finfo(t.dtype).max


def init_(tensor):
	dim = tensor.shape[-1]
	std = 1 / math.sqrt(dim)
	tensor.uniform_(-std, std)
	return tensor


# feedforward
class GEGLU(nn.Module):
	def __init__(self, dim_in, dim_out):
		super().__init__()
		self.proj = nn.Linear(dim_in, dim_out * 2)

	def forward(self, x):
		x, gate = self.proj(x).chunk(2, dim=-1)
		return x * F.gelu(gate)


class FeedForward(nn.Module):
	def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
		super().__init__()
		inner_dim = int(dim * mult)
		dim_out = default(dim_out, dim)
		project_in = nn.Sequential(
			nn.Linear(dim, inner_dim),
			nn.GELU()
		) if not glu else GEGLU(dim, inner_dim)

		self.net = nn.Sequential(
			project_in,
			nn.Dropout(dropout),
			nn.Linear(inner_dim, dim_out)
		)

	def forward(self, x):
		return self.net(x)


def zero_module(module):
	"""
	Zero out the parameters of a module and return it.
	"""
	for p in module.parameters():
		p.detach().zero_()
	return module


class CrossAttention(nn.Module):

	def __init__(
		self,
		query_dim,
		context_dim=None,
		heads=8,
		dim_head=64,
		dropout=0.0,
		topk=-1,
	):
		super().__init__()
		inner_dim = dim_head * heads
		context_dim = default(context_dim, query_dim)

		self.scale = dim_head**-0.5
		self.heads = heads
		self.topk = topk

		self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
		self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
		self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

		self.to_out = nn.Sequential(
			nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
		)

	def forward(
		self,
		x,
		context=None,
		mask=None,
	):
		h = self.heads
		q = self.to_q(x)
		context = default(context, x)
		k = self.to_k(context)
		v = self.to_v(context)

		q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

		if self.topk > 0:
			# Compute scaled QK attention matrix
			dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
			# Select topk values for each row
			top_dots, top_inds = dots.topk(self.topk, dim=-1, sorted=False)
			# Softmax on topk in each row
			top_attn = F.softmax(top_dots, dim=-1).to(dots)
			# Zero out non-topk values
			attn = dots.zero_().scatter_(-1, top_inds, top_attn)
			# Matmul with V
			out = torch.matmul(attn, v)
		else:
			out = F.scaled_dot_product_attention(
				q, k, v, attn_mask=mask
			)  # scale is dim_head ** -0.5 per default

		del q, k, v
		out = rearrange(out, "b h n d -> b n (h d)", h=h)

		return self.to_out(out)
	

class DeformableAttention(nn.Module):
	"""
	Deformable attention as proposed in Deformable DETR. With removed multiscale
	"""

	def __init__(self, d_model, num_heads, n_points=4, disable_custom_kernels=True):
		super().__init__()

		kernel_loaded = MultiScaleDeformableAttention is not None
		if is_torch_cuda_available() and is_ninja_available() and not kernel_loaded and not disable_custom_kernels:
			try:
				load_cuda_kernels()
				logger.log("Loaded CUDA kernel for deformable attention")
			except Exception as e:
				logger.warn(f"Could not load the custom kernel for multi-scale deformable attention: {e}")
		else:
			logger.log("Custom CUDA kernel for deformable attention is disabled")


		if d_model % num_heads != 0:
			raise ValueError(
				f"embed_dim (d_model) must be divisible by num_heads, but got {d_model} and {num_heads}"
			)
		dim_per_head = d_model // num_heads
		# check if dim_per_head is power of 2
		if not ((dim_per_head & (dim_per_head - 1) == 0) and dim_per_head != 0):
			warnings.warn(
				"You'd better set embed_dim (d_model) in DeformableDetrMultiscaleDeformableAttention to make the"
				" dimension of each attention head a power of 2 which is more efficient in the authors' CUDA"
				" implementation."
			)

		self.im2col_step = 64
		self.d_model = d_model
		self.n_heads = num_heads
		self.n_points = n_points
		self.n_levels = 1

		self.sampling_offsets = nn.Linear(d_model, num_heads * self.n_levels * n_points * 2)
		self.attention_weights = nn.Linear(d_model, num_heads * self.n_levels * n_points)
		self.value_proj = nn.Linear(d_model, d_model)
		self.output_proj = nn.Linear(d_model, d_model)

		self.disable_custom_kernels = disable_custom_kernels

		self._reset_parameters()

	def _reset_parameters(self):
		nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
		default_dtype = torch.get_default_dtype()
		thetas = torch.arange(self.n_heads, dtype=torch.int64).to(default_dtype) * (2.0 * math.pi / self.n_heads)
		grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
		grid_init = (
			(grid_init / grid_init.abs().max(-1, keepdim=True)[0])
			.view(self.n_heads, 1, 1, 2)
			.repeat(1, self.n_levels, self.n_points, 1)
		)
		for i in range(self.n_points):
			grid_init[:, :, i, :] *= i + 1
		with torch.no_grad():
			self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
		nn.init.constant_(self.attention_weights.weight.data, 0.0)
		nn.init.constant_(self.attention_weights.bias.data, 0.0)
		nn.init.xavier_uniform_(self.value_proj.weight.data)
		nn.init.constant_(self.value_proj.bias.data, 0.0)
		nn.init.xavier_uniform_(self.output_proj.weight.data)
		nn.init.constant_(self.output_proj.bias.data, 0.0)

	def with_pos_embed(self, tensor, position_embeddings):
		return tensor if position_embeddings is None else tensor + position_embeddings

	def forward(
		self,
		hidden_states,
		attention_mask=None,
		encoder_hidden_states=None,
		position_embeddings=None,
		reference_points=None,
		spatial_shapes=None
	):
		# add position embeddings to the hidden states before projecting to queries and keys
		if position_embeddings is not None:
			hidden_states = self.with_pos_embed(hidden_states, position_embeddings)

		batch_size, num_queries, _ = hidden_states.shape
		batch_size, sequence_length, _ = encoder_hidden_states.shape
		if (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() != sequence_length:
			raise ValueError(
				"Make sure to align the spatial shapes with the sequence length of the encoder hidden states"
			)

		value = self.value_proj(encoder_hidden_states)
		if attention_mask is not None:
			# we invert the attention_mask
			value = value.masked_fill(~attention_mask[..., None], float(0))
		value = value.view(batch_size, sequence_length, self.n_heads, self.d_model // self.n_heads)
		sampling_offsets = self.sampling_offsets(hidden_states).view(
			batch_size, num_queries, self.n_heads, self.n_levels, self.n_points, 2
		)
		attention_weights = self.attention_weights(hidden_states).view(
			batch_size, num_queries, self.n_heads, self.n_levels * self.n_points
		)
		attention_weights = F.softmax(attention_weights, -1).view(
			batch_size, num_queries, self.n_heads, self.n_levels, self.n_points
		)
		# batch_size, num_queries, n_heads, n_levels, n_points, 2
		num_coordinates = reference_points.shape[-1]
		if num_coordinates == 2:
			offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
			sampling_locations = (
				reference_points[:, :, None, :, None, :]
				+ sampling_offsets / offset_normalizer[None, None, None, :, None, :]
			)
		elif num_coordinates == 4:
			sampling_locations = (
				reference_points[:, :, None, :, None, :2]
				+ sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
			)
		else:
			raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}")

		if self.disable_custom_kernels:
			# PyTorch implementation
			output = multi_scale_deformable_attention(value, spatial_shapes, sampling_locations, attention_weights)
		else:
			try:
				# custom kernel
				output = MultiScaleDeformableAttentionFunction.apply(
					value,
					spatial_shapes,
					None,
					sampling_locations,
					attention_weights,
					self.im2col_step,
				)
			except Exception:
				# PyTorch implementation
				output = multi_scale_deformable_attention(value, spatial_shapes, sampling_locations, attention_weights)
		output = self.output_proj(output)

		return output


class BasicTransformerBlock(nn.Module):

	def __init__(
		self,
		dim, 
		n_heads, 
		d_head, 
		dropout=0., 
		context_dim=None, 
		gated_ff=True, 
		use_checkpoint=True,
	):
		super().__init__()
		self.attn1 = CrossAttention(
			query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout
		) # is a self-attention
		self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
		self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
									heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none

		self.norm1 = nn.LayerNorm(dim)
		self.norm2 = nn.LayerNorm(dim)
		self.norm3 = nn.LayerNorm(dim)
		self.use_checkpoint = use_checkpoint

	def forward(self, x, context=None):
		if self.use_checkpoint:
			return checkpoint(self._forward, x, context, use_reentrant=False)
		else:
			return self._forward(x, context)

	def _forward(self, x, context=None):
		x = self.attn1(self.norm1(x)) + x
		x = self.attn2(self.norm2(x), context=context) + x
		x = self.ff(self.norm3(x)) + x
		return x


class EnhancedTransformerBlock(nn.Module):

	def __init__(
		self,
		dim, 
		n_heads, 
		d_head, 
		dropout=0., 
		context_dim=None, 
		gated_ff=True, 
		use_checkpoint=True,
		disable_self_attn=False,
		deformable_self_attn=False,
		enhanced=False,
		nx_enhanced=1,
		topk=-1,
		ctx_x_im=False,
	):
		super().__init__()
		self.disable_self_attn = disable_self_attn
		self.deformable_self_attn = deformable_self_attn
		self.enhanced = enhanced
		self.nx_enhanced = nx_enhanced
		self.ctx_x_im = ctx_x_im

		self.ctx_attns = nn.ModuleList(
			[CrossAttention(query_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout)for _ in range(self.nx_enhanced)]  # is a self-attention
		)
		self.im_attn = DeformableAttention(
			d_model=dim, num_heads=n_heads, disable_custom_kernels=True
		) if self.deformable_self_attn else CrossAttention(
			query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout, topk=topk
		) # is a self-attention if not self.disable_self_attn

		self.im_x_ctx_attn = CrossAttention(query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout)

		if self.ctx_x_im:
			self.ctx_x_im_norm = nn.LayerNorm(context_dim)
			self.ctx_x_im_attn = CrossAttention(query_dim=context_dim, context_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)
		
		self.ff_ctx = FeedForward(context_dim, dropout=dropout, glu=gated_ff)
		self.ff_im = FeedForward(dim, dropout=dropout, glu=gated_ff)

		self.norm1 = nn.LayerNorm(dim)
		self.norm2 = nn.LayerNorm(dim)
		self.norm3 = nn.LayerNorm(dim)

		self.c_norms = nn.ModuleList(
			[nn.LayerNorm(context_dim) for _ in range(self.nx_enhanced)]
		)
		# self.c_norm1 = nn.LayerNorm(context_dim)
		self.use_checkpoint = use_checkpoint

	def forward(self, x, spatial_shapes, context=None):
		if self.use_checkpoint:
			return checkpoint(self._forward, x, context, spatial_shapes, use_reentrant=False)
		else:
			return self._forward(x, spatial_shapes, context)

	def _forward(self, x, context, spatial_shapes):
		if not self.disable_self_attn:
			xn = self.norm1(x)
			if self.deformable_self_attn:
				reference_points = DeformableDetrEncoder.get_reference_points(
					spatial_shapes=spatial_shapes, 
					valid_ratios=torch.ones((1, 1, 2), dtype=torch.float32, device=x.device), 
					device=x.device
				)
				x = self.im_attn(
					hidden_states=xn,
					encoder_hidden_states=xn,
					reference_points=reference_points,
					spatial_shapes=spatial_shapes
				) + x
			else:
				x = self.im_attn(self.norm1(x)) + x

		if self.ctx_x_im and exists(context):
			context = self.ctx_x_im_attn(self.ctx_x_im_norm(context), context=x) + context

		if self.enhanced and exists(context):
			for ctx_attn, c_norm in zip(self.ctx_attns, self.c_norms):
				context = ctx_attn(c_norm(context)) + context

		if exists(context):
			x = self.im_x_ctx_attn(self.norm2(x), context=context) + x
		x = self.ff_im(self.norm3(x)) + x

		return x