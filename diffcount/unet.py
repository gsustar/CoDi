from abc import abstractmethod

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange

from .conditioning import LightRoIAlignExemplarEmbedder
from .attention import EnhancedTransformerBlock, BasicTransformerBlock
from .nn import (
	conv_nd,
	linear,
	avg_pool_nd,
	zero_module,
	timestep_embedding,
	exists,
)


class TimestepBlock(nn.Module):
	"""
	Any module where forward() takes timestep embeddings as a second argument.
	"""

	@abstractmethod
	def forward(self, x, emb):
		"""
		Apply the module to `x` given `emb` timestep embeddings.
		"""


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
	"""
	A sequential module that passes timestep embeddings to the children that
	support it as an extra input.
	"""
	def forward(self, x, emb, context=None, bboxes=None, E_os=None, concat_inter=None):
		for layer in self:
			if isinstance(layer, TimestepBlock):
				x = layer(x, emb)
			elif isinstance(layer, SpatialTransformer):
				x = layer(x, context=context, bboxes=bboxes, E_os=E_os)
			elif isinstance(layer, AtLvlConv):
				x = layer(x, concat_inter)
			else:
				x = layer(x)
		return x


class Upsample(nn.Module):
	"""
	An upsampling layer with an optional convolution.

	:param channels: channels in the inputs and outputs.
	:param use_conv: a bool determining if a convolution is applied.
	:param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
				 upsampling occurs in the inner-two dimensions.
	"""

	def __init__(self, channels, use_conv, dims=2, out_channels=None):
		super().__init__()
		self.channels = channels
		self.out_channels = out_channels or channels
		self.use_conv = use_conv
		self.dims = dims
		if use_conv:
			self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

	def forward(self, x):
		assert x.shape[1] == self.channels
		if self.dims == 3:
			x = F.interpolate(
				x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
			)
		else:
			x = F.interpolate(x, scale_factor=2, mode="nearest")
		if self.use_conv:
			x = self.conv(x)
		return x


class Downsample(nn.Module):
	"""
	A downsampling layer with an optional convolution.

	:param channels: channels in the inputs and outputs.
	:param use_conv: a bool determining if a convolution is applied.
	:param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
				 downsampling occurs in the inner-two dimensions.
	"""

	def __init__(self, channels, use_conv, dims=2, out_channels=None):
		super().__init__()
		self.channels = channels
		self.out_channels = out_channels or channels
		self.use_conv = use_conv
		self.dims = dims
		stride = 2 if dims != 3 else (1, 2, 2)
		if use_conv:
			self.op = conv_nd(
				dims, self.channels, self.out_channels, 3, stride=stride, padding=1
			)
		else:
			assert self.channels == self.out_channels
			self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

	def forward(self, x):
		assert x.shape[1] == self.channels
		return self.op(x)


class ResBlock(TimestepBlock):
	"""
	A residual block that can optionally change the number of channels.

	:param channels: the number of input channels.
	:param emb_channels: the number of timestep embedding channels.
	:param dropout: the rate of dropout.
	:param out_channels: if specified, the number of out channels.
	:param use_conv: if True and out_channels is specified, use a spatial
		convolution instead of a smaller 1x1 convolution to change the
		channels in the skip connection.
	:param dims: determines if the signal is 1D, 2D, or 3D.
	:param use_checkpoint: if True, use gradient checkpointing on this module.
	:param up: if True, use this block for upsampling.
	:param down: if True, use this block for downsampling.
	"""

	def __init__(
		self,
		channels,
		emb_channels,
		dropout,
		out_channels=None,
		use_conv=False,
		use_scale_shift_norm=False,
		dims=2,
		use_checkpoint=False,
		up=False,
		down=False,
	):
		super().__init__()
		self.channels = channels
		self.emb_channels = emb_channels
		self.dropout = dropout
		self.out_channels = out_channels or channels
		self.use_conv = use_conv
		self.use_checkpoint = use_checkpoint
		self.use_scale_shift_norm = use_scale_shift_norm

		self.in_layers = nn.Sequential(
			nn.GroupNorm(32, channels),
			nn.SiLU(),
			conv_nd(dims, channels, self.out_channels, 3, padding=1),
		)

		self.updown = up or down

		if up:
			self.h_upd = Upsample(channels, False, dims)
			self.x_upd = Upsample(channels, False, dims)
		elif down:
			self.h_upd = Downsample(channels, False, dims)
			self.x_upd = Downsample(channels, False, dims)
		else:
			self.h_upd = self.x_upd = nn.Identity()

		self.emb_layers = nn.Sequential(
			nn.SiLU(),
			linear(
				emb_channels,
				2 * self.out_channels if use_scale_shift_norm else self.out_channels,
			),
		)
		self.out_layers = nn.Sequential(
			nn.GroupNorm(32, self.out_channels),
			nn.SiLU(),
			nn.Dropout(p=dropout),
			zero_module(
				conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
			),
		)

		if self.out_channels == channels:
			self.skip_connection = nn.Identity()
		elif use_conv:
			self.skip_connection = conv_nd(
				dims, channels, self.out_channels, 3, padding=1
			)
		else:
			self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

	def forward(self, x, emb):
		"""
		Apply the block to a Tensor, conditioned on a timestep embedding.

		:param x: an [N x C x ...] Tensor of features.
		:param emb: an [N x emb_channels] Tensor of timestep embeddings.
		:return: an [N x C x ...] Tensor of outputs.
		"""
		if self.use_checkpoint:
			return checkpoint(self._forward, x, emb, use_reentrant=False)
		else:
			return self._forward(x, emb)


	def _forward(self, x, emb):
		if self.updown:
			in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
			h = in_rest(x)
			h = self.h_upd(h)
			x = self.x_upd(x)
			h = in_conv(h)
		else:
			h = self.in_layers(x)
		emb_out = self.emb_layers(emb)
		while len(emb_out.shape) < len(h.shape):
			emb_out = emb_out[..., None]
		if self.use_scale_shift_norm:
			out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
			scale, shift = th.chunk(emb_out, 2, dim=1)
			h = out_norm(h) * (1 + scale) + shift
			h = out_rest(h)
		else:
			h = h + emb_out
			h = self.out_layers(h)
		return self.skip_connection(x) + h


class SpatialTransformer(nn.Module):
	"""
	Transformer block for image-like data.
	First, project the input (aka embedding)
	and reshape to b, t, d.
	Then apply standard transformer action.
	Finally, reshape to image
	"""
	def __init__(
		self,
		in_channels,
		n_heads,
		d_head,
		spatial_scale,
		depth=1,
		dropout=0.,
		context_dim=None,
		roi_encode_size=False,
		roi_output_size=1,
		unfold_roi=False,
		enhanced=False,
		nx_enhanced=1,
		disable_self_attn=False,
		deformable_self_attn=False,
		topk=-1,
		roi_skip_linear=False,
		ctx_x_im=False,
	):
		super().__init__()

		if d_head == -1:
			d_head = in_channels // n_heads
		else:
			assert in_channels % d_head == 0, (
				f"q,k,v channels {in_channels} is not divisible by num_head_channels {d_head}"
			)
			n_heads = in_channels // d_head
	
		self.in_channels = in_channels
		inner_dim = n_heads * d_head
		self.norm = nn.GroupNorm(32, in_channels, eps=1e-6, affine=True)
		self.bbox_embed = LightRoIAlignExemplarEmbedder(
			in_channels=inner_dim,
			out_channels=context_dim,
			roi_output_size=roi_output_size,
			spatial_scale=spatial_scale,
			encode_size=roi_encode_size,
			unfold_roi=unfold_roi,
			skip_linear=roi_skip_linear,
		)

		self.proj_in = nn.Conv2d(in_channels,
								 inner_dim,
								 kernel_size=1,
								 stride=1,
								 padding=0)

		self.enhanced = enhanced
		self.transformer_blocks = nn.ModuleList([
			EnhancedTransformerBlock(
				inner_dim, 
				n_heads, 
				d_head, 
				dropout=dropout, 
				context_dim=context_dim,
				disable_self_attn=disable_self_attn,
				deformable_self_attn=deformable_self_attn,
				enhanced=enhanced,
				nx_enhanced=nx_enhanced,
				topk=topk,
				ctx_x_im=ctx_x_im
			) for _ in range(depth)
		])


		self.proj_out = zero_module(nn.Conv2d(inner_dim,
											  in_channels,
											  kernel_size=1,
											  stride=1,
											  padding=0))


	def forward(self, x, context=None, bboxes=None, E_os=None):
		# note: if no context is given, cross-attention defaults to self-attention
		b, c, h, w = x.shape
		spatial_shapes = th.tensor([[h, w]], device=x.device)
		x_in = x
		x = self.norm(x)
		x = self.proj_in(x)
		if bboxes is not None:
			ex_emb = self.bbox_embed(x, bboxes)
			if E_os is not None:
				ex_emb = ex_emb + E_os
			context = (
				ex_emb if context is None else 
				th.cat((context, ex_emb), dim=1)
			)
		x = rearrange(x, 'b c h w -> b (h w) c')
		for block in self.transformer_blocks:
			x = block(x, spatial_shapes, context=context)
		x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
		x = self.proj_out(x)
		return x + x_in


class CountingBranch(nn.Module):
	
	def __init__(self, feat_dims, hidden_dim=64):
		super().__init__()
		self.num_feats = len(feat_dims)
		self.input_dim = int(sum(feat_dims.values()))

		self.avgpool = nn.AdaptiveAvgPool2d(1)
		self.norm = nn.LayerNorm(self.input_dim)
		self.mlp = nn.Sequential(
			nn.Linear(self.input_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, 1)
		)

	def forward(self, feats):
		x = [feats[key] for key in feats]
		x = th.cat([self.avgpool(feats[key]) for key in feats], dim=1)
		x = x.flatten(start_dim=1)
		x = self.norm(x)
		x = self.mlp(x)
		return x


class AtLvlConv(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3):
		super().__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)

	def forward(self, x, concat_at_lvl=None):
		if concat_at_lvl is None:
			return x
		assert x.shape[-2:] == concat_at_lvl.shape[-2:], f"Shape mismatch: {x.shape} vs {concat_at_lvl.shape}"
		x = th.cat((x, concat_at_lvl), dim=1)
		x = self.conv(x)
		return x


class UNetModel(nn.Module):
	"""
	The full UNet model with attention and timestep embedding.

	:param in_channels: channels in the input Tensor.
	:param model_channels: base channel count for the model.
	:param out_channels: channels in the output Tensor.
	:param num_res_blocks: number of residual blocks per downsample.
	:param attention_resolutions: a collection of downsample rates at which
		attention will take place. May be a set, list, or tuple.
		For example, if this contains 4, then at 4x downsampling, attention
		will be used.
	:param dropout: the dropout probability.
	:param channel_mult: channel multiplier for each level of the UNet.
	:param conv_resample: if True, use learned convolutions for upsampling and
		downsampling.
	:param dims: determines if the signal is 1D, 2D, or 3D.
	:param num_classes: if specified (as an int), then this model will be
		class-conditional with `num_classes` classes.
	:param use_checkpoint: use gradient checkpointing to reduce memory usage.
	:param num_heads: the number of attention heads in each attention layer.
	:param num_heads_channels: if specified, ignore num_heads and instead use
							   a fixed channel width per attention head.
	:param num_heads_upsample: works with num_heads to set a different number
							   of heads for upsampling. Deprecated.
	:param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
	:param resblock_updown: use residual blocks for up/downsampling.
	:param use_new_attention_order: use a different attention pattern for potentially
									increased efficiency.
	"""

	def __init__(
		self,
		in_channels,
		model_channels,
		out_channels,
		num_res_blocks,
		attention_resolutions,
		dropout=0,
		channel_mult=(1, 2, 4, 8),
		conv_resample=True,
		dims=2,
		y_dim=None,
		context_dim=None,
		use_checkpoint=False,
		num_heads=1,
		num_head_channels=-1,
		num_heads_upsample=-1,
		use_scale_shift_norm=False,
		resblock_updown=False,
		learn_count=False,
		learn_sigma=False,
		transformer_depth=1,
		initial_ds=1.0,
		st_roi_encode_size=False,
		st_roi_output_size=1,
		st_unfold_roi=False,
		st_skip_linear=False,
		disable_middle_transformer=False,
		disable_self_attentions=None,
		enhanced_spatial_transformer=False,
		nx_enhanced=1,
		deformable_self_attn=False,
		topk=-1,
		num_embeddings=0,
		concat_at_ds=None,
		concat_at_ds_ch=None,
		extra_down_block=False,
	):
		super().__init__()

		if num_heads_upsample == -1:
			num_heads_upsample = num_heads

		self.in_channels = in_channels
		self.model_channels = model_channels
		self.out_channels = out_channels
		self.num_res_blocks = num_res_blocks
		self.attention_resolutions = attention_resolutions
		self.dropout = dropout
		self.channel_mult = channel_mult
		self.conv_resample = conv_resample
		self.context_dim = context_dim
		self.y_dim = y_dim
		self.use_checkpoint = use_checkpoint
		self.num_heads = num_heads
		self.num_head_channels = num_head_channels
		self.num_heads_upsample = num_heads_upsample
		self.learn_count = learn_count
		self.learn_sigma = learn_sigma
		self.concat_at_ds = concat_at_ds
		self.concat_at_ds_ch = concat_at_ds_ch

		if isinstance(transformer_depth, int):
			transformer_depth = len(channel_mult) * [transformer_depth]
		transformer_depth_middle = transformer_depth[-1]

		if disable_self_attentions is not None:
			assert len(disable_self_attentions) == len(channel_mult)

		time_embed_dim = model_channels * 4
		self.time_embed = nn.Sequential(
			linear(model_channels, time_embed_dim),
			nn.SiLU(),
			linear(time_embed_dim, time_embed_dim),
		)

		if y_dim is not None:
			self.y_embed = nn.Sequential(
				nn.Linear(y_dim, time_embed_dim),
				nn.SiLU(),
				nn.Linear(time_embed_dim, time_embed_dim),
			)

		ch = input_ch = int(channel_mult[0] * model_channels)
		self.input_blocks = nn.ModuleList(
			[TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
		)
		self._feature_size = ch
		input_block_chans = [ch]
		ds = 1
		in_channel_mult = channel_mult
		if extra_down_block:
			in_channel_mult.insert(0, 1)
		for level, mult in enumerate(in_channel_mult):
			for _ in range(num_res_blocks):
				layers = []
				if concat_at_ds is not None and ds == concat_at_ds:
					layers.append(
						AtLvlConv(ch + concat_at_ds_ch, ch, kernel_size=3)
					)
				layers.append(
					ResBlock(
						ch,
						time_embed_dim,
						dropout,
						out_channels=int(mult * model_channels),
						dims=dims,
						use_checkpoint=use_checkpoint,
						use_scale_shift_norm=use_scale_shift_norm,
					)
				)

				ch = int(mult * model_channels)
				if ds in attention_resolutions:
					if context_dim is not None and exists(disable_self_attentions):
						disabled_sa = disable_self_attentions[level]
					else:
						disabled_sa = False
					layers.append(
	  					SpatialTransformer(
							ch, 
							n_heads=num_heads, 
							d_head=num_head_channels,
							depth=transformer_depth[level],
							spatial_scale=(initial_ds / ds),
							context_dim=context_dim,
							roi_encode_size=st_roi_encode_size,
							roi_output_size=st_roi_output_size,
							unfold_roi=st_unfold_roi,
							enhanced=enhanced_spatial_transformer,
							nx_enhanced=nx_enhanced,
							disable_self_attn=disabled_sa,
							deformable_self_attn=deformable_self_attn,
							topk=topk,
							roi_skip_linear=st_skip_linear,
							ctx_x_im=(num_embeddings != 0),
						)
					)
				self.input_blocks.append(TimestepEmbedSequential(*layers))
				self._feature_size += ch
				input_block_chans.append(ch)
			if level != len(channel_mult) - 1:
				out_ch = ch
				self.input_blocks.append(
					TimestepEmbedSequential(
						ResBlock(
							ch,
							time_embed_dim,
							dropout,
							out_channels=out_ch,
							dims=dims,
							use_checkpoint=use_checkpoint,
							use_scale_shift_norm=use_scale_shift_norm,
							down=True,
						)
						if resblock_updown
						else Downsample(
							ch, conv_resample, dims=dims, out_channels=out_ch
						)
					)
				)
				ch = out_ch
				input_block_chans.append(ch)
				ds *= 2
				if level == 0 and extra_down_block:
					ds //= 2
				self._feature_size += ch

		self.middle_block = TimestepEmbedSequential(
			ResBlock(
				ch,
				time_embed_dim,
				dropout,
				dims=dims,
				use_checkpoint=use_checkpoint,
				use_scale_shift_norm=use_scale_shift_norm,
			),
			SpatialTransformer(
				ch, 
				n_heads=num_heads, 
				d_head=num_head_channels,
				depth=transformer_depth_middle,
				spatial_scale=(initial_ds / ds),
				context_dim=context_dim,
				roi_encode_size=st_roi_encode_size,
				roi_output_size=st_roi_output_size,
				unfold_roi=st_unfold_roi,
				enhanced=enhanced_spatial_transformer,
				nx_enhanced=nx_enhanced,
				deformable_self_attn=deformable_self_attn,
				topk=topk,
				roi_skip_linear=st_skip_linear,
			)
			if not disable_middle_transformer
            else th.nn.Identity(),
			ResBlock(
				ch,
				time_embed_dim,
				dropout,
				dims=dims,
				use_checkpoint=use_checkpoint,
				use_scale_shift_norm=use_scale_shift_norm,
			),
		)
		self._feature_size += ch

		self.output_blocks = nn.ModuleList([])
		for level, mult in list(enumerate(channel_mult))[::-1]:
			for i in range(num_res_blocks + 1):
				ich = input_block_chans.pop()
				layers = [
					ResBlock(
						ch + ich,
						time_embed_dim,
						dropout,
						out_channels=int(model_channels * mult),
						dims=dims,
						use_checkpoint=use_checkpoint,
						use_scale_shift_norm=use_scale_shift_norm,
					)
				]
				ch = int(model_channels * mult)
				if ds in attention_resolutions:
					if context_dim is not None and exists(disable_self_attentions):
						disabled_sa = disable_self_attentions[level]
					else:
						disabled_sa = False
					layers.append(
	  					SpatialTransformer(
							ch, 
							n_heads=num_heads, 
							d_head=num_head_channels,
							depth=transformer_depth[level],
							spatial_scale=(initial_ds / ds),
							context_dim=context_dim,
							roi_encode_size=st_roi_encode_size,
							roi_output_size=st_roi_output_size,
							unfold_roi=st_unfold_roi,
							enhanced=enhanced_spatial_transformer,
							nx_enhanced=nx_enhanced,
							disable_self_attn=disabled_sa,
							deformable_self_attn=deformable_self_attn,
							topk=topk,
							roi_skip_linear=st_skip_linear,
							ctx_x_im=(num_embeddings != 0),
						)
					)
				if level and i == num_res_blocks:
					out_ch = ch
					layers.append(
						ResBlock(
							ch,
							time_embed_dim,
							dropout,
							out_channels=out_ch,
							dims=dims,
							use_checkpoint=use_checkpoint,
							use_scale_shift_norm=use_scale_shift_norm,
							up=True,
						)
						if resblock_updown
						else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
					)
					ds //= 2
				self.output_blocks.append(TimestepEmbedSequential(*layers))
				self._feature_size += ch


		if learn_count:
			self.feat_extract_list = [1, 4, 7, 10, 13, 17, 20][:len(channel_mult)]
			feat_dims = {
				f"p{layer}": model_channels * mult for layer, mult in zip(self.feat_extract_list, channel_mult)
			}
			self.counting_branch = CountingBranch(feat_dims, hidden_dim=64)

		self.embeddings = None
		if num_embeddings > 0:
			self.embeddings = nn.Parameter(
				nn.init.kaiming_normal_(
					th.empty(num_embeddings, context_dim), mode='fan_out', nonlinearity='relu'
				)
			)

		self.out = nn.Sequential(
			nn.GroupNorm(32, ch),
			nn.SiLU(),
			zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
		)

		self.initialize_weights()

	def initialize_weights(self):
		def _basic_init(module):
			if isinstance(module, nn.Linear):
				th.nn.init.xavier_uniform_(module.weight)
				if module.bias is not None:
					nn.init.constant_(module.bias, 0)
		self.apply(_basic_init)
		
		# Initialize timestep embedding MLP:
		nn.init.normal_(self.time_embed[0].weight, std=0.02)
		nn.init.normal_(self.time_embed[2].weight, std=0.02)


	def _forward(self, x, timesteps, y=None, context=None, bboxes=None, E_os=None, concat_inter=None):
		"""
		Apply the model to an input batch.

		:param x: an [N x C x ...] Tensor of inputs.
		:param timesteps: a 1-D batch of timesteps.
		:param y: an [N] Tensor of labels, if class-conditional.
		:return: an [N x C x ...] Tensor of outputs.
		"""
		xs = []
		de_feats = {}
		dethead_f = None
		emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
		
		if y is not None:
			assert y.shape[0] == x.shape[0]
			emb = emb + self.y_embed(y)

		if self.embeddings is not None:
			_embs = self.embeddings.repeat(x.shape[0], 1, 1)
			context = (
				_embs if context is None else 
				th.cat((context, _embs), dim=1)
			)

		for module in self.input_blocks:
			x = module(x, emb, context=context, bboxes=bboxes, E_os=E_os, concat_inter=concat_inter)
			if dethead_f is None and any(isinstance(x, SpatialTransformer) for x in module):
				dethead_f = x.clone()
			xs.append(x)

		x = self.middle_block(x, emb, context=context, bboxes=bboxes, E_os=E_os, concat_inter=concat_inter)

		for layer, module in enumerate(self.output_blocks):
			x = th.cat([x, xs.pop()], dim=1)
			x = module(x, emb, context=context, bboxes=bboxes, E_os=E_os, concat_inter=concat_inter)

			if self.learn_count and layer in self.feat_extract_list:
				de_feats[f"p{layer}"] = x.clone()

		count = None
		if self.learn_count:
			count = self.counting_branch(de_feats)
		del de_feats
			
		out = self.out(x)
		return dict(out=out, count=count, dethead_f=dethead_f)


	def forward(self, x, t, cond, **kwargs):
		x = th.cat((x, cond.get("concat", th.tensor([]).type_as(x))), dim=1)
		return self._forward(
			x,
			timesteps=t,
			y=cond.get("vector", None),
			context=cond.get("crossattn", None),
			bboxes=cond.get("bboxes", None),
			E_os=cond.get("E_os", None),
			concat_inter=cond.get("concat_inter", None),
		)


	def forward_with_cfg(self, x, t, cond, cfg_scale):
		assert not self.training, "classifier-free guidance is only possible during inference"
		## half = x[: len(x) // 2]
		## combined = th.cat([half, half], dim=0)
		model_out = self.forward(x, t, cond)["out"]
		ch = self.out_channels if not self.learn_sigma else self.out_channels // 2
		eps, rest = model_out[:, :ch], model_out[:, ch:]
		cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
		# There are 2 formulations of CFG floating around. One from the original paper (the uncommented one)
		# and one from the Imagen paper (the commented one).
		# https://github.com/huggingface/diffusers/issues/5882
		half_eps = cond_eps + cfg_scale * (cond_eps - uncond_eps)
		# half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
		eps = th.cat([half_eps, half_eps], dim=0)
		return dict(
			out=th.cat([eps, rest], dim=1),
			count=None
		)