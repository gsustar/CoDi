import os
import yaml
import random
import numpy as np
import torch as th

from types import SimpleNamespace
from diffusers import AutoencoderKL
from torch.utils.data import DistributedSampler

from . import denoise_diffusion as dd
from . import conditioning as cond

from .datasets import FSC147, MNIST, MCAC, load_data, FSCD_LVIS, FullCA44
from .respace import SpacedDiffusion, space_timesteps
from .unet import UNetModel
from .nn import disabled_train


def assert_config(config):
	for att in ["model", "data", "log"]:
		assert hasattr(config, att), f"config missing attribute: {att}"
	assert hasattr(config, "diffusion") or hasattr(config, "rf")
	if hasattr(config, "diffusion"):
		assert config.model.params.learn_count == (config.diffusion.params.lmbd_cb_count > 0.0), (
			"Ayo, don't set learn_count to False if lmbd_cb_count > 0.0"
		)
		assert config.model.params.learn_sigma == config.diffusion.params.learn_sigma

		if config.diffusion.type == "Deblur":
			assert not hasattr(config.diffusion.params, "learn_sigma"), (
				"learn_sigma is not supported for Deblur diffusion."
			)
			
	for embconf in config.conditioner.embedders:
		assert hasattr(embconf, "input_keys"), (
			"input_keys must be specified for each conditioner."
		)



def create_model(model_config):
	if model_config.type == "UNet":
		model = create_unet_model(**vars(model_config.params))
	else:
		raise ValueError(f"Unsupported model type: {model_config.type}")
	return model


def create_diffusion(diffusion_config):
	if diffusion_config.type == "Denoise":
		diffusion = create_denoise_diffusion(**vars(diffusion_config.params),)
	else:
		raise ValueError(f"Unsupported diffusion type: {diffusion_config.type}")
	return diffusion


def create_data(data_config, train=True, distributed=False):
	splits = ['train', 'val', None] if train else [None, 'val', 'test']
	if data_config.dataset.name == "FSC147":
		train_dataset, val_dataset, test_dataset = (
			FSC147(
				**vars(data_config.dataset.params),
				split=split,
			) if split else None for split in splits
		)
	elif data_config.dataset.name == "MNIST":
		train_dataset, val_dataset, test_dataset = (
			MNIST(
				**vars(data_config.dataset.params),
				split=split,
			) if split else None for split in splits
		)
	elif data_config.dataset.name == "MCAC":
		train_dataset, val_dataset, test_dataset = (
			MCAC(
				**vars(data_config.dataset.params),
				split=split,
			) if split else None for split in splits
		)
	elif data_config.dataset.name == "FSCD_LVIS":
		unseen_splits = ["train", "test", None] if train else [None, None, "test"]
		splits = unseen_splits if data_config.dataset.params.unseen else splits
		train_dataset, val_dataset, test_dataset = (
			FSCD_LVIS(
				**vars(data_config.dataset.params),
				split=split
			) if split else None for split in splits
		)
	elif data_config.dataset.name == "FullCA44":
		train_dataset, val_dataset, test_dataset = (
			FullCA44(
				**vars(data_config.dataset.params),
				split=split
			) if split else None for split in splits
		)
	elif data_config.dataset.name == "All":
		datadir = data_config.dataset.params.datadir
		delattr(data_config.dataset.params, "datadir")
		fsc147_train_dataset, fsc147_val_dataset, fsc147_test_dataset = (
			FSC147(
				datadir=os.path.join(datadir, "FSC147_384_V2"),
				**vars(data_config.dataset.params),
				split=split,
			) if split else None for split in splits
		)
		mcac_train_dataset, mcac_val_dataset, mcac_test_dataset = (
			MCAC(
				datadir=os.path.join(datadir, "MCAC"),
				**vars(data_config.dataset.params),
				split=split,
			) if split else None for split in splits
		)
		lvis_train_dataset, lvis_val_dataset, lvis_test_dataset = (
			FSCD_LVIS(
				datadir=os.path.join(datadir, "FSCD_LVIS"),
				**vars(data_config.dataset.params),
				split=split
			) if split else None for split in splits
		)
		ca44_train_dataset, ca44_val_dataset, ca44_test_dataset = (
			FullCA44(
				datadir=os.path.join("/d/hpc/projects/FRI/pelhanj/", "CountAnythingV1_clean"),
				**vars(data_config.dataset.params),
				split=split
			) if split else None for split in splits
		)
		train_dataset = th.utils.data.ConcatDataset(
			[d for d in [fsc147_train_dataset, mcac_train_dataset, lvis_train_dataset, ca44_train_dataset] if d is not None]
		)
		val_dataset = th.utils.data.ConcatDataset(
			[d for d in [fsc147_val_dataset, mcac_val_dataset, lvis_val_dataset, ca44_val_dataset] if d is not None]
		)
		test_dataset = None
	else:
		raise ValueError(f"Unknown dataset: {data_config.dataset}")

	if train:
		train_data = load_data(
			dataset=train_dataset,
			sampler=DistributedSampler(train_dataset) if distributed else None,
			batch_size=data_config.dataloader.params.batch_size,
			num_workers=data_config.dataloader.params.num_workers,
			shuffle=False if distributed else True
		)
		test_data = None
	else:
		test_data = load_data(
			dataset=test_dataset,
			sampler=DistributedSampler(test_dataset, shuffle=False) if distributed else None,
			batch_size=data_config.dataloader.params.batch_size,
			num_workers=data_config.dataloader.params.num_workers,
			shuffle=False
		)
		train_data = None
	
	val_data = load_data(
		dataset=val_dataset,
		sampler=DistributedSampler(val_dataset, shuffle=False) if distributed else None,
		batch_size=data_config.dataloader.params.batch_size,
		num_workers=data_config.dataloader.params.num_workers,
		shuffle=False
	)

	return train_data, val_data, test_data


def create_conditioner(conditioner_config, train=True):
	embedders = []
	for embconf in conditioner_config.embedders:
		params = vars(embconf.params) if hasattr(embconf, "params") else {}
		emb = getattr(cond, embconf.type)(
			**params, 
			input_keys=embconf.input_keys,
			reference_key=getattr(embconf, "reference_key", None),
			custom_outkey=getattr(embconf, "custom_outkey", None),
			custom_catdim=getattr(embconf, "custom_catdim", None),
			is_trainable=getattr(embconf, "is_trainable", False) if train else False, 
			ucg_rate=getattr(embconf, "ucg_rate", 0.0) if train else 0.0, 
		)
		embedders.append(emb)
	return cond.Conditioner(embedders)


def create_vae(vae_config, device, freeze=True):
	vae = None
	if vae_config is not None and vae_config.enabled:
		try:
			vae = AutoencoderKL.from_pretrained(vae_config.path)
		except:
			vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
			vae.load_state_dict(th.load(vae_config.path)["model"])
		if freeze:
			vae.train = disabled_train
			for param in vae.parameters():
				param.requires_grad = False
			vae.eval()
		vae.to(device)
	return vae


def create_unet_model(
	in_channels,
	model_channels,
	out_channels,
	num_res_blocks,
	attention_resolutions,
	dropout,
	channel_mult,
	conv_resample,
	dims,
	context_dim,
	y_dim,
	use_checkpoint,
	num_heads,
	num_head_channels,
	num_heads_upsample,
	use_scale_shift_norm,
	resblock_updown,
	learn_sigma,
	learn_count,
	transformer_depth,
	initial_ds,
	st_roi_encode_size,
	st_roi_output_size,
	st_unfold_roi,
	st_skip_linear,
	disable_middle_transformer,
	disable_self_attentions,
	enhanced_spatial_transformer,
	nx_enhanced,
	deformable_self_attn,
	topk,
	num_embeddings=0,
	concat_at_ds=None,
	concat_at_ds_ch=None,
):
	channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult)

	return UNetModel(
		in_channels=in_channels,
		model_channels=model_channels,
		out_channels=(out_channels if not learn_sigma else 2 * out_channels),
		num_res_blocks=num_res_blocks,
		attention_resolutions=tuple(attention_resolutions),
		dropout=dropout,
		channel_mult=channel_mult,
		conv_resample=conv_resample,
		dims=dims,
		context_dim=context_dim,
		y_dim=y_dim,
		use_checkpoint=use_checkpoint,
		num_heads=num_heads,
		num_head_channels=num_head_channels,
		num_heads_upsample=num_heads_upsample,
		use_scale_shift_norm=use_scale_shift_norm,
		resblock_updown=resblock_updown,
		learn_count=learn_count,
		learn_sigma=learn_sigma,
		transformer_depth=transformer_depth,
		initial_ds=initial_ds,
		st_roi_encode_size=st_roi_encode_size,
		st_roi_output_size=st_roi_output_size,
		st_unfold_roi=st_unfold_roi,
		st_skip_linear=st_skip_linear,
		disable_middle_transformer=disable_middle_transformer,
		disable_self_attentions=disable_self_attentions,
		enhanced_spatial_transformer=enhanced_spatial_transformer,
		nx_enhanced=nx_enhanced,
		deformable_self_attn=deformable_self_attn,
		topk=topk,
		num_embeddings=num_embeddings,
		concat_at_ds=concat_at_ds,
		concat_at_ds_ch=concat_at_ds_ch,
	)


def create_denoise_diffusion(
	diffusion_steps,
	learn_sigma,
	sigma_small,
	noise_schedule,
	use_kl,
	parametrization,
	rescale_timesteps,
	rescale_learned_sigmas,
	timestep_respacing,
	lmbd_vlb,
	lmbd_cb_count,
	t_mse_weighting_scheme,
	t_cb_count_weighting_scheme,
	enforce_zero_terminal_snr,
	do_giou_loss=False,
	weighted_tlrb_loss=False,
	apply_multi_res_noise=False,
):
	_model_mean_type = dd.ModelMeanType.EPSILON
	if parametrization == "xstart":
		_model_mean_type = dd.ModelMeanType.START_X
	elif parametrization == "eps":
		_model_mean_type = dd.ModelMeanType.EPSILON
	elif parametrization == "xprev":
		_model_mean_type = dd.ModelMeanType.PREVIOUS_X
	elif parametrization == "v":
		_model_mean_type = dd.ModelMeanType.V
	else:
		raise ValueError(f"Unsupported model mean parametrization: {parametrization}")
	
	betas = dd.get_named_beta_schedule(noise_schedule, diffusion_steps)
	if enforce_zero_terminal_snr:
		assert parametrization == "v", ("Zero terminal SNR is only viable when using V parametrization.")
		if noise_schedule == "linear" or noise_schedule == "scaled_linear":
			betas = dd.enforce_zero_terminal_snr(betas)
		elif noise_schedule == "cosine":
			betas = dd.get_named_beta_schedule(noise_schedule, diffusion_steps, max_beta=1.0)
		else:
			raise ValueError(f"Zero terminal SNR is not supported for {noise_schedule} noise schedule.")

	if use_kl:
		loss_type = dd.LossType.RESCALED_KL
	elif rescale_learned_sigmas:
		loss_type = dd.LossType.RESCALED_MSE
	else:
		loss_type = dd.LossType.MSE
	if not timestep_respacing:
		timestep_respacing = [diffusion_steps]
	return SpacedDiffusion(
		use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
		betas=betas,
		model_mean_type=_model_mean_type,
		model_var_type=(
			(
				dd.ModelVarType.FIXED_LARGE
				if not sigma_small
				else dd.ModelVarType.FIXED_SMALL
			)
			if not learn_sigma
			else dd.ModelVarType.LEARNED_RANGE
		),
		loss_type=loss_type,
		rescale_timesteps=rescale_timesteps,
		lmbd_vlb=lmbd_vlb,
		lmbd_cb_count=lmbd_cb_count,
		t_mse_weighting_scheme=t_mse_weighting_scheme,
		t_cb_count_weighting_scheme=t_cb_count_weighting_scheme,
		do_giou_loss=do_giou_loss,
		weighted_tlrb_loss=weighted_tlrb_loss,
		apply_multi_res_noise=apply_multi_res_noise,
	)


def seed_everything(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	th.manual_seed(seed)
	th.cuda.manual_seed(seed)


def dict_to_namespace(d):
	x = SimpleNamespace()
	_ = [setattr(x, k,
				 dict_to_namespace(v) if isinstance(v, dict)
				 else [dict_to_namespace(e) if isinstance(e, dict) else e for e in v] if isinstance(v, list)
				 else v) for k, v in d.items()]
	return x


def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, SimpleNamespace) 
		else [namespace_to_dict(e) if isinstance(e, SimpleNamespace) else e for e in v] if isinstance(v, list)
		else v
        for k, v in vars(namespace).items()
    }


def parse_config(configpath):
	with open(configpath, "r") as stream:
		try:
			return dict_to_namespace(
				yaml.safe_load(stream)
			)
		except yaml.YAMLError as e:
			print(e)
