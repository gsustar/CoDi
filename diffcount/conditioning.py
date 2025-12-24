import torch as th
import torch.nn as nn
import numpy as np

from torchvision.transforms.v2 import functional as F
from torch.nn.functional import interpolate

from contextlib import nullcontext
from torchvision.ops import roi_align
from einops import rearrange
from transformers import (
	ViTModel,
	ViTImageProcessor,
	Dinov2Backbone,
	CLIPTokenizer,
	CLIPTextModel,
	Swinv2Model,
	AutoImageProcessor,
)

from .nn import (
	disabled_train, 
	count_params, 
	timestep_embedding, 
	possibly_vae_encode, 
	expand_dims_like,
	avg_pool_nd,
	VAE_DOWNSCALE_FACTOR,
	TILED_BBOXES_PADDING_VALUE,
)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class AbstractEmbModel(nn.Module):
	
	def __init__(
		self, 
		input_keys=None, 
		reference_key=None,
		custom_outkey=None,
		custom_catdim=None,
		ucg_rate=None, 
		is_trainable=False,
		**kwargs
	):
		super().__init__()
		if custom_outkey is not None and custom_outkey != "discard":
			assert custom_catdim is not None, (
				"custom_catdim has to be specified iff custom_outkey is specified"
			)
		self.input_keys = input_keys
		self.reference_key = reference_key
		self.custom_outkey = custom_outkey
		self.custom_catdim = custom_catdim
		self.ucg_rate = ucg_rate
		self.is_trainable = is_trainable



class Conditioner(nn.Module):
	OUTPUT_DIM2KEYS = {2: "vector", 3: "crossattn", 4: "concat"}
	KEY2CATDIM = {"vector": 1, "crossattn": 1, "concat": 1}

	def __init__(self, emb_models):
		super().__init__()

		embedders = []
		for n, embedder in enumerate(emb_models):
			assert isinstance(
				embedder, AbstractEmbModel
			), f"embedder model {embedder.__class__.__name__} has to inherit from AbstractEmbModel"
			if not embedder.is_trainable:
				embedder.train = disabled_train
				for param in embedder.parameters():
					param.requires_grad = False
				embedder.eval()
			print(
				f"Initialized embedder #{n}: {embedder.__class__.__name__} "
				f"with {count_params(embedder, False)} params. Trainable: {embedder.is_trainable}"
			)
			embedders.append(embedder)
		self.embedders = nn.ModuleList(embedders)


	def forward(self, cond, vae=None, force_zero_embeddings=None):
		
		if force_zero_embeddings is None:
			force_zero_embeddings = []

		output = dict()
		log = dict()
		_cond = cond.copy()
		for embedder in self.embedders:
			embedding_context = nullcontext if embedder.is_trainable else th.no_grad
			with embedding_context():
				emb_out = embedder(*[_cond[k] for k in embedder.input_keys])

			geco_pts = None
			if isinstance(emb_out, tuple):
				emb_out, geco_pts, geco_boxes = emb_out

			emb_out = emb_out.float()
			assert isinstance(
				emb_out, th.Tensor
			), f"encoder outputs must be tensors {type(emb_out)}"

			if isinstance(embedder, (SAM2ExemplarMaskEmbedder)):
				log["sam_masks"] = emb_out

			# if isinstance(embedder, (GeCoCenternessEmbedder)):
			# 	log["geco_centerness"] = emb_out

			if embedder.custom_outkey is not None:
				out_key = embedder.custom_outkey
				cat_dim = embedder.custom_catdim
			else:
				out_key = self.OUTPUT_DIM2KEYS[emb_out.dim()]
				cat_dim = self.KEY2CATDIM[out_key]

			if out_key == "concat":
				if not isinstance(embedder, (DINOv2ImageEmbedder, 
											 DINOImageEmbedder, 
											 Swinv2ImageEmbedder, 
											 SAM2ImageEmbedder, 
											 ProjectEmbedder, 
											 RADIOImageEmbedder)):
					emb_out = possibly_vae_encode(emb_out, vae)

			if embedder.reference_key is not None:
				_cond[embedder.reference_key] = emb_out

			if self.training and embedder.ucg_rate > 0.0:
				emb_out = (
					expand_dims_like(
						th.bernoulli(
							(1.0 - embedder.ucg_rate)
							* th.ones(emb_out.shape[0], device=emb_out.device)
						), emb_out,
					) * emb_out
				)
			
			if (
				embedder.input_keys is not None
				and any(k in force_zero_embeddings for k in embedder.input_keys)
			):
				emb_out = th.zeros_like(emb_out)

			if out_key in output:
				output[out_key] = th.cat(
					(output[out_key], emb_out), cat_dim
				)
			else:
				if out_key != "discard":
					output[out_key] = emb_out

			if geco_pts is not None:
				output["geco_pts"] = geco_pts
				output["geco_boxes"] = geco_boxes
		return output, log
	


	def get_unconditional_conditioning(
		self,
		cond,
		ucond=None,
		force_uc_zero_embeddings=None,
		force_cond_zero_embeddings=None,
		vae=None
	):
		assert not self.training, "Conditioner has to be in eval mode to get unconditional conditioning"
		c, _ = self(cond, vae, force_cond_zero_embeddings)
		uc, _ = self(
			cond if ucond is None else ucond, 
			vae, 
			force_uc_zero_embeddings
		)
		return c, uc



class ClassEmbedder(AbstractEmbModel):
	def __init__(self, embed_dim, n_classes=10, add_sequence_dim=False, **kwargs):
		super().__init__(**kwargs)
		self.embedding = nn.Embedding(n_classes, embed_dim)
		self.n_classes = n_classes
		self.add_sequence_dim = add_sequence_dim

	def forward(self, c):
		c = self.embedding(c)
		if self.add_sequence_dim:
			c = c[:, None, :]
		return c


class ImageConcatEmbedder(AbstractEmbModel):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def forward(self, img):
		return img
	

class BBoxAppendEmbedder(AbstractEmbModel):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def forward(self, bboxes):
		return bboxes
	

class ConcatTimestepEmbedderND(AbstractEmbModel):
	"""embeds each dimension independently and concatenates them"""

	def __init__(self, outdim, **kwargs):
		super().__init__(**kwargs)
		self.outdim = outdim

	def forward(self, x):
		while len(x.shape) < 3:
			x = x[..., None]
		assert len(x.shape) == 3
		b, n, dims = x.shape
		x = rearrange(x, "b n d -> (b n d)")
		emb = timestep_embedding(x, dim=self.outdim)
		emb = rearrange(emb, "(b n d) d2 -> b n (d d2)", b=b, n=n, d=dims, d2=self.outdim)
		return emb


class BBoxSizeEmbedder(AbstractEmbModel):

	def __init__(
			self,
			outdim,
			embdim=256,
			**kwargs
		):
		super().__init__(**kwargs)
		self.objectness = nn.Sequential(
			nn.Linear(2, 64),
			nn.ReLU(),
			nn.Linear(64, embdim),
			nn.ReLU(),
			nn.Linear(embdim, outdim)
		)

	def forward(self, bboxes):
		bbox_size = th.stack((
			bboxes[:, :, 3] - bboxes[:, :, 1],
			bboxes[:, :, 2] - bboxes[:, :, 0]), dim=2
		)
		x = self.objectness(bbox_size)
		return x


class FourierBBoxSizeEmbedder(AbstractEmbModel):

	def __init__(
			self,
			outdim,
			embdim=256,
			scale=None,
			**kwargs
		):
		super().__init__(**kwargs)
		if scale is None or scale <= 0.0:
			scale = 1.0

		self.outdim = outdim
		self.scale = scale
		self.embdim = embdim

		self.register_buffer(
			"positional_encoding_gaussian_matrix",
			self.scale * th.randn((2, self.embdim // 2)),
		)
		self.objectness = nn.Sequential(
			nn.Linear(embdim, embdim),
			nn.ReLU(),
			nn.Linear(embdim, embdim),
			nn.ReLU(),
			nn.Linear(embdim, outdim)
		)

	def forward(self, img, bboxes):
		_, _, h, w = img.shape
		bbox_size = th.stack((
			(bboxes[:, :, 3] - bboxes[:, :, 1]) / h,
			(bboxes[:, :, 2] - bboxes[:, :, 0]) / w), dim=2
		)
		bbox_size = 2 * bbox_size - 1
		bbox_size = bbox_size @ self.positional_encoding_gaussian_matrix
		bbox_size = 2 * np.pi * bbox_size
		bbox_size = th.cat([th.sin(bbox_size), th.cos(bbox_size)], dim=-1)
		x = self.objectness(bbox_size)
		return x


class RoIAlignExemplarEmbedder(AbstractEmbModel):

	def __init__(
		self,
		in_channels,
		out_channels,
		roi_output_size=1,
		spatial_scale=0.125,
		remove_sequence_dim=False,
		encode_size=False,
		mlp_ratio=4.0,
		**kwargs
	):
		super().__init__(**kwargs)

		self.in_channels = in_channels
		self.out_channels = out_channels
		self.roi_output_size = roi_output_size
		self.spatial_scale = spatial_scale
		self.remove_sequence_dim = remove_sequence_dim
		self.encode_size = encode_size

		assert out_channels % 2 == 0
		self.bbox_size_embed = BBoxSizeEmbedder(outdim=out_channels)
		self.avgpool = nn.AdaptiveAvgPool2d(1)
		inner_dim = int(mlp_ratio * in_channels)
		self.out = nn.Sequential(
			nn.Linear(in_channels, inner_dim),
			nn.ReLU(),
			nn.Linear(inner_dim, out_channels)
		)
		self.out2 = nn.Linear(2 * out_channels, out_channels)

	def _tile_replicate(self, x, bboxes):
		valid_mask = ~th.all(bboxes == TILED_BBOXES_PADDING_VALUE, dim=2)
		valid_rois = th.stack(bboxes.shape[0] * [x[valid_mask]])
		return valid_rois


	def forward(self, z, bboxes, ds=1.0):
		bs, ch, _, _ = z.shape
		assert ch > 3, "z must be a feature map not an image"
		x = roi_align(
			z, 
			boxes=list(bboxes), 
			output_size=self.roi_output_size, 
			spatial_scale=(self.spatial_scale / ds),
			aligned=True
		)
		if self.roi_output_size > 1:
			x = self.avgpool(x)
		x = x.reshape(bs, -1, x.shape[1])
		x = self.out(x)

		if self.encode_size:
			x = th.cat([x, self.bbox_size_embed(bboxes)], dim=-1)
			x = self.out2(x)

		# if not self.training and bs > 1:
		# 	x = self._tile_replicate(x, bboxes)

		if self.remove_sequence_dim:
			x = x.reshape(bs, -1)
		return x


class LightRoIAlignExemplarEmbedder(AbstractEmbModel):
	
	def __init__(
		self,
		in_channels,
		out_channels,
		roi_output_size=1,
		spatial_scale=0.125,
		remove_sequence_dim=False,
		encode_size=False,
		unfold_roi=False,
		skip_linear=False,
		**kwargs
	):
		super().__init__(**kwargs)

		self.in_channels = in_channels
		self.out_channels = out_channels
		self.roi_output_size = roi_output_size
		self.spatial_scale = spatial_scale
		self.remove_sequence_dim = remove_sequence_dim
		self.encode_size = encode_size
		self.unfold_roi = unfold_roi
		self.skip_linear = skip_linear

		assert out_channels % 2 == 0
		self.bbox_size_embed = BBoxSizeEmbedder(outdim=out_channels)
		self.avgpool = nn.AdaptiveAvgPool2d(1)
		self.out = nn.Linear(in_channels, out_channels)
		self.out2 = nn.Linear(2 * out_channels, out_channels)


	def _cfg_drop(self, x, bboxes):
		# properly handle cfg drop for when this embedder is used inside UNet
		cfg_mask = (~th.all(bboxes == 0, dim=2, keepdims=True)).float()
		cfg_mask = cfg_mask.repeat(1, self.roi_output_size * self.roi_output_size, 1)
		return x * cfg_mask
	

	def _tile_replicate(self, x, bboxes):
		valid_mask = ~th.all(bboxes == TILED_BBOXES_PADDING_VALUE, dim=2)
		valid_rois = th.stack(bboxes.shape[0] * [x[valid_mask]])
		return valid_rois


	def forward(self, z, bboxes, ds=1.0):
		bs, ch, _, _ = z.shape
		assert ch > 3, "z must be a feature map not an image"
		x = roi_align(
			z, 
			boxes=list(bboxes), 
			output_size=self.roi_output_size, 
			spatial_scale=(self.spatial_scale / ds),
			aligned=True
		)

		pooled = self.avgpool(x)
		if self.roi_output_size > 1 and not self.unfold_roi:
			x = pooled

		x = x.reshape(bs, -1, x.shape[1])
		if self.roi_output_size > 1 and self.unfold_roi:
			pooled = pooled.reshape(bs, -1, x.shape[-1])
			th.cat([x, pooled], dim=1)

		if not self.skip_linear:
			x = self.out(x)

		if self.encode_size:
			x = th.cat([x, self.bbox_size_embed(bboxes)], dim=-1)
			x = self.out2(x)

		if self.training:
			x = self._cfg_drop(x, bboxes)

		# if not self.training and bs > 1:
		# 	x = self._tile_replicate(x, bboxes)

		if self.remove_sequence_dim:
			x = x.reshape(bs, -1)
		return x



class SAM2ExemplarMaskEmbedder(AbstractEmbModel):

	def __init__(
		self, 
		checkpoint="facebook/sam2.1-hiera-tiny", 
		score_threshold=0.6, 
		**kwargs
	):
		super().__init__(**kwargs)
		try:
			from sam2.sam2_image_predictor import SAM2ImagePredictor #type: ignore

		except ImportError:
			raise ImportError("sam2 is required for SAM2ImageMaskEmbedder")


		self.checkpoint = checkpoint
		self.score_threshold = score_threshold
		self.predictor = SAM2ImagePredictor.from_pretrained(checkpoint)


	def forward(self, img, bboxes):
		dev = self.predictor.device
		img = img.permute(0, 2, 3, 1).cpu().numpy()
		img = (img + 1.0) / 2.0
		img = [im for im in img]

		with th.autocast(device_type=dev.type, dtype=th.bfloat16):
			self.predictor.set_image_batch(img)
			masks_batch, scores_batch, _ = self.predictor.predict_batch(
				None,
				None, 
				box_batch=bboxes, 
				multimask_output=False
			)
		masks_batch = np.stack(masks_batch).squeeze(axis=2)
		scores_batch = np.stack(scores_batch)[:, :, None]

		masks_batch = th.as_tensor(masks_batch, device=dev, dtype=th.float32)
		scores_batch = th.as_tensor(scores_batch, device=dev, dtype=th.float32)

		mul_ = (scores_batch > self.score_threshold).float()
		masks_batch *= mul_
		return masks_batch * 2 - 1


class ProjectEmbedder(AbstractEmbModel):

	def __init__(
		self,
		in_channels,
		out_channels,
		**kwargs
	):
		super().__init__(**kwargs)
		self.project = nn.Conv2d(
				in_channels,
				out_channels,
				kernel_size=1,
				stride=1,
				padding=0,
				bias=True
			)

	def forward(self, *zs):
		z = th.cat(zs, dim=1)
		assert z.ndim == 4
		return self.project(z)



class SAM2ImageEmbedder(AbstractEmbModel):

	def __init__(
		self, 
		out_channels,
		checkpoint="facebook/sam2-hiera-tiny",
		img_downscale_factor=1/VAE_DOWNSCALE_FACTOR,
		**kwargs
	):
		super().__init__(**kwargs)
		try:
			from sam2.sam2_image_predictor import SAM2ImagePredictor #type: ignore

		except ImportError:
			raise ImportError("sam2 is required for SAM2ImageEmbedder")

		self.checkpoint = checkpoint
		self.predictor = SAM2ImagePredictor.from_pretrained(checkpoint)
		self.img_downscale_factor = img_downscale_factor
		self.out = nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=1, stride=1, bias=True)
		self.freeze()

	def freeze(self):
		self.predictor.model.eval()
		self.predictor.model.train = disabled_train
		for param in self.predictor.model.parameters():
			param.requires_grad = False

	def forward(self, img):
		h, w = img.shape[-2:]
		dev = self.predictor.device
		assert h == w == 1024, "SAM2ImageEmbedder only supports input resolution of 1024"

		img = img.permute(0, 2, 3, 1).cpu().numpy()
		img = (img + 1.0) / 2.0
		img = [im for im in img]

		with th.no_grad():
			with th.autocast(device_type=dev.type, dtype=th.bfloat16):
				self.predictor.set_image_batch(img)

		x = self.predictor._features["image_embed"]
		x = interpolate(x, scale_factor=2, mode='nearest')
		x = self.out(x)
		return x



class DINOv2ImageEmbedder(AbstractEmbModel):

	def __init__(
		self,
		out_channels,
		out_indices="auto",
		checkpoint="facebook/dinov2-small",
		img_downscale_factor=1/VAE_DOWNSCALE_FACTOR,
		resize=False,
		dtype="float16",
		**kwargs
	):
		super().__init__(**kwargs)
		assert checkpoint in ["facebook/dinov2-small", "facebook/dinov2-base", "facebook/dinov2-large", "facebook/dinov2-giant"]
		self.out_channels = out_channels
		self.img_downscale_factor = img_downscale_factor
		self.out_indices = out_indices
		self.resize = resize
		self.dtype = getattr(th, dtype)

		if out_indices == "auto":
			if checkpoint == "facebook/dinov2-small":
				self.out_indices = [2, 5, 8, 11]
			elif checkpoint == "facebook/dinov2-base":
				self.out_indices = [2, 5, 8, 11]
			elif checkpoint == "facebook/dinov2-large":
				self.out_indices = [5, 11, 17, 23]
			elif checkpoint == "facebook/dinov2-giant":
				self.out_indices = [9, 19, 29, 39]

		self.image_processor = AutoImageProcessor.from_pretrained(
			checkpoint, do_resize=False, do_center_crop=False, do_rescale=False, do_convert_rgb=False
		)
		self.backbone = Dinov2Backbone.from_pretrained(
			checkpoint,
			out_indices=self.out_indices,
		)
		self.patch_size = self.backbone.config.patch_size
		assert max(self.out_indices) <= len(self.backbone.encoder.layer) - 1

		in_channels = len(self.out_indices) * self.backbone.config.hidden_size
		self.out = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=True)

		self.freeze()

	def freeze(self):
		self.backbone.eval()
		self.backbone.train = disabled_train
		for name, param in self.backbone.named_parameters():
			param.requires_grad = False

	
	def forward(self, img):
		dev = self.backbone.device
		h, w = img.shape[-2:]
		lat_h, lat_w = (
			int(h * self.img_downscale_factor), 
			int(w * self.img_downscale_factor)
		)
		if self.resize:
			resize_h = lat_h * self.patch_size
			resize_w = lat_w * self.patch_size
			img = F.resize(img, (resize_h, resize_w))

		img = (img + 1.0) / 2.0

		with th.no_grad():
			with th.autocast(device_type=dev.type, dtype=self.dtype):
				inputs = self.image_processor(img, return_tensors="pt").to(img.device)
				x = self.backbone(inputs["pixel_values"])

		# This interpolate has an effect only if self.resize is False
		fs = [
			interpolate(f, (lat_h, lat_w), mode='bilinear', align_corners=True) 
			for f in x.feature_maps
		]
		x = th.cat(fs, dim=1)
		x = self.out(x)
		return x


class DINOImageEmbedder(AbstractEmbModel):

	def __init__(
		self,
		out_channels,
		out_indices=[2, 5, 8, 11],
		checkpoint="facebook/dino-vitb16",
		img_downscale_factor=1/VAE_DOWNSCALE_FACTOR,
		resize=False,
		dtype="float16",
		**kwargs
	):
		super().__init__(**kwargs)
		assert checkpoint in ["facebook/dino-vitb16", "facebook/dino-vitb8"]

		self.img_downscale_factor=img_downscale_factor
		self.out_indices = out_indices
		self.resize = resize
		self.dtype = getattr(th, dtype)

		self.image_processor = ViTImageProcessor.from_pretrained(
			checkpoint, do_resize=False, do_rescale=False
		)
		self.backbone = ViTModel.from_pretrained(
			checkpoint,
			output_hidden_states=True,
		)
		assert max(self.out_indices) <= len(self.backbone.encoder.layer) - 1

		self.patch_size = self.backbone.config.patch_size
		in_channels = self.backbone.config.hidden_size

		self.out = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=True)
		self.freeze()

	def freeze(self):
		self.backbone.eval()
		self.backbone.train = disabled_train
		for name, param in self.backbone.named_parameters():
			param.requires_grad = False
	
	def forward(self, img):
		dev = self.backbone.device
		bs, _, h, w = img.shape
		lat_h, lat_w = (
			int(h * self.img_downscale_factor), 
			int(w * self.img_downscale_factor)
		)
		if self.resize:
			resize_h = lat_h * self.patch_size
			resize_w = lat_w * self.patch_size
			img = F.resize(img, (resize_h, resize_w))

		img = (img + 1.0) / 2.0

		with th.no_grad():
			with th.autocast(device_type=dev.type, dtype=self.dtype):
				inputs = self.image_processor(img, return_tensors="pt").to(img.device)
				x = self.backbone(inputs["pixel_values"], interpolate_pos_encoding=True)
		hidden_states = x.hidden_states[1:] # remove 'stem' stage

		fs = []
		for stage, token_seq in enumerate(hidden_states):
			if stage in self.out_indices:
				f = token_seq[:, 1:] # remove 'cls' token
				f = f.reshape(bs, h // self.patch_size, w // self.patch_size, -1)
				f = f.permute(0, 3, 1, 2).contiguous()
				# This interpolate has an effect only if self.resize is False
				f = interpolate(f, (lat_h, lat_w), mode='nearest')
				fs.append(f)
		x = th.cat(fs, dim=1)

		x = self.out(x)
		return x



class Swinv2ImageEmbedder(AbstractEmbModel):

	def __init__(
		self,
		out_channels,
		checkpoint="microsoft/swinv2-base-patch4-window8-256",
		img_downscale_factor=1/VAE_DOWNSCALE_FACTOR,
		unfreeze=False,
		**kwargs
	):
		super().__init__(**kwargs)
		self.img_downscale_factor=img_downscale_factor
		self.unfreeze = unfreeze
		self.image_processor = AutoImageProcessor.from_pretrained(checkpoint, do_resize=False, do_rescale=False)
		self.backbone = Swinv2Model.from_pretrained(
			checkpoint, 
			output_hidden_states=True
		)
		hs = self.backbone.config.hidden_size
		self.hidden_states_dims = [hs//8, hs//4, hs//2, hs, hs]
		in_channels = sum(self.hidden_states_dims)

		self.downsample = avg_pool_nd(dims=2, kernel_size=2, stride=2)
		self.out = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=True)

		self.backbone_context = nullcontext
		self.freeze()

	def freeze(self):
		if not self.unfreeze:
			self.backbone.eval()
			self.backbone_context = th.no_grad
			self.backbone.train = disabled_train
			for param in self.backbone.parameters():
				param.requires_grad = False

	def forward(self, img):
		h, w = img.shape[-2:]
		lat_h, lat_w = (
			int(h * self.img_downscale_factor), 
			int(w * self.img_downscale_factor)
		)
		img = (img + 1.0) / 2.0

		with self.backbone_context():
			inputs = self.image_processor(img, return_tensors="pt").to(img.device)
			x = self.backbone(inputs["pixel_values"], interpolate_pos_encoding=True)

		fs =[]
		for f in x.reshaped_hidden_states:
			fh, fw = f.shape[-2:]
			if fh > lat_h and fw > lat_w:
				fs.append(self.downsample(f))
			elif fh < lat_h and fw < lat_w:
				fs.append(interpolate(f, (lat_h, lat_w), mode='nearest'))
			else:
				fs.append(f)

		x = th.cat(fs, dim=1)
		x = self.out(x)
		return x


# Ugly monkey patch
import math
from typing import Tuple
import types
def _new_get_pos_embeddings(self, batch_size: int, input_dims: Tuple[int, int]):
		if (self.num_rows, self.num_cols) == input_dims:
			return self.pos_embed

		pos_embed = self.pos_embed.reshape(1, self.num_rows, self.num_cols, -1).permute(0, 3, 1, 2)

		def window_select(pos_embed):
			if input_dims[0] < pos_embed.shape[-2]:
				pos_embed = pos_embed[..., :input_dims[0], :]
			if input_dims[1] < pos_embed.shape[-1]:
				pos_embed = pos_embed[..., :, :input_dims[1]]
			return pos_embed

		if self.cpe_mode:
			if self.training:
				if self.num_video_frames is not None:
					if batch_size % self.num_video_frames != 0:
						raise ValueError(f'Batch size {batch_size} must be divisible by num_video_frames {self.num_video_frames} for CPE mode.')

					batch_size //= self.num_video_frames

				min_scale = math.sqrt(0.1)
				scale = th.rand(batch_size, 1, 1, device=pos_embed.device) * (1 - min_scale) + min_scale
				aspect_min = math.log(3 / 4)
				aspect_max = -aspect_min
				aspect = th.exp(th.rand(batch_size, 1, 1, device=pos_embed.device) * (aspect_max - aspect_min) + aspect_min)

				scale_x = scale * aspect
				scale_y = scale * (1 / aspect)
				scale_xy = th.stack([scale_x, scale_y], dim=-1).clamp_(0, 1)

				pos_xy = th.rand(batch_size, 1, 1, 2, device=pos_embed.device) * (1 - scale_xy)

				lin_x = th.linspace(0, 1, steps=input_dims[1], device=pos_embed.device)[None, None].expand(batch_size, input_dims[0], -1)
				lin_y = th.linspace(0, 1, steps=input_dims[0], device=pos_embed.device)[None, :, None].expand(batch_size, -1, input_dims[1])

				lin_xy = th.stack([lin_x, lin_y], dim=-1)

				grid_xy = lin_xy * scale_xy + pos_xy

				# Convert to [-1, 1] range
				grid_xy.mul_(2).sub_(1)

				pos_embed = F.grid_sample(
					pos_embed.float().expand(batch_size, -1, -1, -1),
					grid=grid_xy,
					mode='bilinear',
					padding_mode='zeros',
					align_corners=True,
				).to(pos_embed.dtype)

				if self.num_video_frames is not None:
					pos_embed = th.repeat_interleave(pos_embed, self.num_video_frames, dim=0)
			else:
				# i_rows, i_cols = input_dims
				# p_rows, p_cols = pos_embed.shape[2:]
				# if i_rows <= p_rows and i_cols <= p_cols:
				#     left = (p_cols - i_cols) // 2
				#     top = (p_rows - i_rows) // 2
				#     pos_embed = pos_embed[..., top:top+i_rows, left:left+i_cols]
				# else:
				max_dim = max(input_dims)
				pos_embed = F.crop(pos_embed.float(), top=0, left=0, height=max_dim, width=max_dim)
				# pos_embed = F.interpolate(pos_embed.float(), size=(max_dim, max_dim), align_corners=False, mode='bilinear').to(pos_embed.dtype)

				pos_embed = window_select(pos_embed)
		else:
			pos_embed = window_select(pos_embed)

		if pos_embed.shape[-2:] != input_dims:
			pos_embed = F.interpolate(pos_embed.float(), size=input_dims, align_corners=False, mode='bilinear').to(pos_embed.dtype)

		pos_embed = pos_embed.flatten(2).permute(0, 2, 1)

		return pos_embed

class RADIOImageEmbedder(AbstractEmbModel):

	def __init__(
		self,
		out_channels,
		out_indices="auto",
		model_version="radio_v2.5-b",
		img_downscale_factor=1/VAE_DOWNSCALE_FACTOR,
		resize=False,
		dtype="bfloat16",
		crop_pos=False,
		**kwargs
	):
		super().__init__(**kwargs)
		assert model_version in ["radio_v2.5-b", "radio_v2.5-l", "radio_v2.5-h"]
		
		self.out_channels = out_channels
		self.img_downscale_factor=img_downscale_factor
		self.model_version = model_version
		self.out_indices = out_indices
		self.resize = resize
		self.dtype = getattr(th, dtype)

		if out_indices == "auto":
			if model_version == "radio_v2.5-b":
				self.out_indices = [2, 5, 8, 11]
			elif model_version == "radio_v2.5-l":
				self.out_indices = [5, 11, 17, 23]
			elif model_version == "radio_v2.5-h":
				self.out_indices = [7, 15, 23, 31]

		self.backbone = th.hub.load('NVlabs/RADIO', 'radio_model', version=model_version, progress=False, skip_validation=True)
		if crop_pos:
			self.backbone.model.patch_generator._get_pos_embeddings = types.MethodType(_new_get_pos_embeddings, self.backbone.model.patch_generator)
		self.backbone.model.patch_generator._load_from_state_dict = lambda *args, **kwargs: None
		self.backbone.model.patch_generator.embedder._load_from_state_dict = lambda *args, **kwargs: None
		self.patch_size = self.backbone.patch_size

		assert max(self.out_indices) <= len(self.backbone.blocks) - 1
		in_channels = len(out_indices) * self.backbone.embed_dim

		self.out = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=True)
		self.freeze()

	def freeze(self):
		self.backbone.eval()
		self.backbone.train = disabled_train
		for name, param in self.backbone.named_parameters():
			param.requires_grad = False
	
	def forward(self, img):
		dev = next(iter(self.out.parameters())).device
		bs, _, h, w = img.shape
		lat_h, lat_w = (
			int(h * self.img_downscale_factor), 
			int(w * self.img_downscale_factor)
		)
		if self.resize:
			resize_h = lat_h * self.patch_size
			resize_w = lat_w * self.patch_size
			img = F.resize(img, (resize_h, resize_w))

		img = (img + 1.0) / 2.0

		with th.no_grad():
			with th.autocast(device_type=dev.type, dtype=self.dtype):
				(summary, final), features = self.backbone.forward_intermediates(img, indices=self.out_indices)

		fs = []
		for f in features:
			# This interplate has an effect only if self.resize is False
			f = interpolate(f, (lat_h, lat_w), mode='nearest')
			fs.append(f)
		x = th.cat(fs, dim=1)
		x = self.out(x)
		return x


class FrozenCLIPTextEmbedder(AbstractEmbModel):
	"""Uses the CLIP transformer encoder for text (from huggingface)"""

	LAYERS = ["last", "pooled", "hidden"]

	def __init__(
		self,
		out_channels,
		version="openai/clip-vit-large-patch14",
		max_length=77,
		layer="last",
		layer_idx=None,
		always_return_pooled=False,
		**kwargs
	):  # clip-vit-base-patch32
		super().__init__(**kwargs)
		assert layer in self.LAYERS

		self.tokenizer = CLIPTokenizer.from_pretrained(version)
		self.transformer = CLIPTextModel.from_pretrained(version)

		self.max_length = max_length
		self.layer = layer
		self.layer_idx = layer_idx
		self.return_pooled = always_return_pooled

		if layer == "hidden":
			assert layer_idx is not None
			assert 0 <= abs(layer_idx) <= 12

		self.out = nn.Linear(self.transformer.config.hidden_size, out_channels)
		self.freeze()

	def freeze(self):
		self.transformer = self.transformer.eval()
		self.train = disabled_train
		for param in self.parameters():
			param.requires_grad = False

	def forward(self, text):
		dev = self.transformer.device

		with th.no_grad():
			batch_encoding = self.tokenizer(
				text,
				truncation=True,
				max_length=self.max_length,
				return_length=True,
				return_overflowing_tokens=False,
				padding="max_length",
				return_tensors="pt",
			)
			tokens = batch_encoding["input_ids"].to(dev)
			outputs = self.transformer(
				input_ids=tokens, output_hidden_states=self.layer == "hidden"
			)

		if self.layer == "last":
			z = outputs.last_hidden_state
		elif self.layer == "pooled":
			z = outputs.pooler_output[:, None, :]
		else:
			z = outputs.hidden_states[self.layer_idx]
		z = self.out(z)

		if self.return_pooled:
			return z, outputs.pooler_output
		return z

	def encode(self, text):
		return self(text)
