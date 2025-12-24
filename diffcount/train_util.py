import torch as th
import numpy as np
import pprint

import os.path as osp
import torch.distributed as dist

from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP

from . import logger
from .resample import UniformSampler
from .plot_util import draw_bboxes, draw_text, draw_result
from .ema import ExponentialMovingAverage
from .nn import possibly_vae_decode, torch_to, possibly_vae_encode, VAE_DOWNSCALE_FACTOR
from .infer_util import(
	counting, 
	collate_channels, 
)


class TrainLoop:
	def __init__(
		self,
		*,
		model,
		diffusion,
		data,
		val_data,
		conditioner,
		vae,
		batch_size,
		lr,
		log_interval,
		save_interval,
		validation_interval,
		resume_checkpoint,
		device,
		ema_rate,
		weight_decay=0.0,
		num_epochs=0,
		grad_clip=0.0,
		lr_scheduler=None,
		overfit=False,
		mixed_precision=False,
		single_ch_vae=False,
		zeroshot_adapt=None,
	):
		self.data = data
		self.val_data = val_data
		self.batch_size = batch_size
		self.lr = lr
		self.device = device
		self.log_interval = log_interval
		self.save_interval = save_interval
		self.validation_interval = validation_interval
		self.resume_checkpoint = resume_checkpoint
		self.ema_rate = ema_rate
		self.schedule_sampler = UniformSampler(diffusion)
		self.weight_decay = weight_decay
		self.num_epochs = num_epochs
		self.grad_clip = grad_clip
		self.lr_scheduler = lr_scheduler
		self.overfit = overfit
		self.mixed_precision = mixed_precision
		self.zeroshot_adapt = zeroshot_adapt

		self.steps_per_epoch = len(self.data)
		self.log_training_imgs = False
		self.step = 0
		self.epoch = 0

		if self.overfit:
			self.data = [next(iter(self.data))]
			self.val_data = self.data

		self.diffusion = diffusion
		self.conditioner = conditioner
		self.model = model
		self.vae = vae
		self.single_ch_vae = single_ch_vae

		try:
			self.scaler = th.amp.GradScaler("cuda", enabled=self.mixed_precision)
		except Exception as e:
			logger.log(f"GradScaler not available: {e}")
			self.scaler = None
		self.opt = self.configure_optimizer()
		self.sch = self.configure_scheduler(self.opt)
		self.ema = ExponentialMovingAverage(
			self.model.parameters(),
			decay=ema_rate,
		)

		if self.resume_checkpoint:
			self.load()

		if zeroshot_adapt:
			assert self.resume_checkpoint
			for param in self.model.parameters():
				param.requires_grad = False
			for param in self.conditioner.parameters():
				param.requires_grad = False

			# unfreeze learnable embeddings and ctx_x_im attn
			for name, param in self.model.named_parameters():
				matches = ["embeddings", "ctx_x_im_attn", "ctx_x_im_norm"]
				if any(m in name for m in matches):
					param.requires_grad = True
			for name, param in self.conditioner.named_parameters():
				matches = ["embedders.1.out"] # embdder.1 is text_embedder]
				if any(m in name for m in matches):
					param.requires_grad = True

		self.ddp_model = DDP(
			self.model,
			device_ids=[self.device],
			output_device=self.device,
			find_unused_parameters=True,
		)

		if any([emb.is_trainable for emb in self.conditioner.embedders]):
			self.ddp_conditioner = DDP(
				self.conditioner,
				device_ids=[self.device],
				output_device=self.device,
				find_unused_parameters=True,
			)
		else:
			self.ddp_conditioner = self.conditioner



	def load(self):
		logger.log(f"loading model from checkpoint: {self.resume_checkpoint}...")
		checkpoint = th.load(self.resume_checkpoint, map_location=self.device)

		model_state_dict = checkpoint.get("model", checkpoint)
		# optimizer_state_dict = checkpoint.get("optimizer", None)
		conditioner_state_dict = checkpoint.get("conditioner", None)
		# scheduler_state_dict = checkpoint.get("scheduler", None)
		ema_state_dict = checkpoint.get("ema", None)

		msd = dict()
		self_msd = self.model.state_dict()
		for k, v in model_state_dict.items():
			if k in self_msd and v.shape != self_msd[k].shape:
				logger.log(
					f"size mismatch for {k}: copying a param with shape {v.shape} from checkpoint, the shape in current model is {self_msd[k].shape}"
				)
			else:
				msd[k] = v
		m, u = self.model.load_state_dict(msd, strict=False)
		if m: logger.log(f"missing keys: {pprint.pformat(m)}\n")
		if u: logger.log(f"unexpected keys: {pprint.pformat(u)}\n")

		# if optimizer_state_dict:
		# 	self.opt.load_state_dict(optimizer_state_dict)
		if conditioner_state_dict:
			self.conditioner.load_state_dict(conditioner_state_dict, strict=False)
		# if scheduler_state_dict:
		# 	self.sch.load_state_dict(scheduler_state_dict)
		# if ema_state_dict:
		# 	self.ema.load_state_dict(ema_state_dict)


	def run_loop(self):
		while (
			(not self.num_epochs or self.epoch < self.num_epochs)
		):
			self.run_epoch()
		# Save the last checkpoint if it wasn't already saved.
		if self.epoch % self.save_interval != 0:
			self.save()


	def run_epoch(self):
		self.ddp_model.train()
		self.ddp_conditioner.train()

		if self.epoch % self.validation_interval == 0 and self.epoch > 0:
			self.log_training_imgs = True

		if not self.overfit:
			self.data.sampler.set_epoch(self.epoch)

		for tgt, cond in self.data:
			self.run_step(tgt, cond)
		self.epoch += 1

		if self.epoch % self.validation_interval == 0:
			self.validate()

		if self.epoch % self.save_interval == 0:
			self.save()


	def run_step(self, tgt, cond):
		self.opt.zero_grad()
		tgt = torch_to(tgt, self.device, non_blocking=True)
		cond = torch_to(cond, self.device, non_blocking=True)

		_tgt = possibly_vae_encode(tgt, self.vae, single_ch=self.single_ch_vae)

		count = cond.pop("count")
		_cond, clog = self.ddp_conditioner(
			cond, self.vae,
		)

		t, weights = self.schedule_sampler.sample(_tgt.shape[0], self.device)

		with th.autocast(device_type=self.device.type, dtype=th.float16, enabled=self.mixed_precision):
			losses = self.diffusion.training_losses(
				model=self.ddp_model,
				x_start=_tgt,
				t=t,
				model_kwargs=dict(
					cond=_cond,
					count=count,
					tlrb_mask=cond["tlrb_mask"],
					tlrb_wm=cond["tlrb_wm"],
					gt_boxes=cond["gt_bboxes"],
				),
				noise=None,
			)

		loss = (losses["loss"] * weights).mean()

		if self.scaler is not None:
			self.scaler.scale(loss).backward()
			self.scaler.unscale_(self.opt)

			grad_norm, param_norm = self.compute_norms()
			if self.grad_clip > 0:
				th.nn.utils.clip_grad_norm_(self.ddp_model.parameters(), max_norm=self.grad_clip)

			# self.opt.step()
			self.scaler.step(self.opt)
			self.scaler.update()
			if self.sch is not None:
				self.sch.step()
		else:
			loss.backward()
			grad_norm, param_norm = self.compute_norms()
			if self.grad_clip > 0:
				th.nn.utils.clip_grad_norm_(self.ddp_model.parameters(), max_norm=self.grad_clip)
			self.opt.step()
			if self.sch is not None:
				self.sch.step()

		log_loss_dict(
			self.diffusion, t, {k: v * weights for k, v in losses.items()}
		)

		if self.log_training_imgs:
			log_tgt(tgt, prefix="train", postfix="targets", step=self.step)
			log_cond(cond | clog, prefix="train", step=self.step)

		logger.logkv("step", self.step)
		logger.logkv("epoch", self.epoch)
		logger.logkv("lr", self.opt.param_groups[0]["lr"])
		logger.logkv_mean("grad_norm", grad_norm)
		logger.logkv_mean("param_norm", param_norm)

		if self.step % self.log_interval == 0:
			logger.dumpkvs()
		self.step += 1
		self.log_training_imgs = False


	@th.no_grad
	def validate(self):
		logger.log("creating samples...")
		self.ddp_model.eval()
		self.ddp_conditioner.eval()
		tgt, cond = next(iter(self.val_data))
		tgt = torch_to(tgt, self.device, non_blocking=True)
		cond = torch_to(cond, self.device, non_blocking=True)
		_cond, clog = self.ddp_conditioner(
			cond, self.vae,
		)

		ch_mult = self.vae.config.latent_channels if self.vae else 1
		if self.with_tlrb:
			ich = 5 * ch_mult
			if self.only_tlrb:
				ich = 4 * ch_mult
		else:
			ich = ch_mult
		h, w = cond["img"].shape[-2:]
		h = h // VAE_DOWNSCALE_FACTOR if self.vae else h
		w = w // VAE_DOWNSCALE_FACTOR if self.vae else w

		with self.ema.average_parameters(self.ddp_model.parameters()):
			samples = self.diffusion.p_sample_loop_progressive(
				model=self.ddp_model,
				shape=(self.batch_size, ich, h , w),
				noise=None,
				clip_denoised=False,
				denoised_fn=None,
				model_kwargs=dict(
					cond=_cond
				),
				device=self.device,
				progress=False
			)

			final = log_denoising_process(
				samples, self.diffusion, vae=self.vae, t_step=None, step=self.step
			)
		log_results(
			final, cond, step=self.step
		)
		log_tgt(
			tgt, prefix="val", postfix="targets", step=self.step
		)
		log_cond(
			cond | clog, prefix="val", step=self.step
		)
		dist.barrier()


	def save(self):
		if dist.get_rank() == 0:
			logger.log(f"saving model...")
			filename = f"model{(self.epoch):06d}.pt"

			_conditioner = (
				self.ddp_conditioner.module 
				if isinstance(self.ddp_conditioner, DDP) 
				else self.ddp_conditioner
			)

			csd = _conditioner.state_dict()

			with open(osp.join(logger.get_dir(), filename), "wb") as f:
				checkpoint = {
					"epoch": self.epoch,
					"step": self.step,
					"model": self.ddp_model.module.state_dict(),
					# "optimizer": self.opt.state_dict(),
					# "scheduler": self.sch.state_dict() if self.sch is not None else None,
					# "scaler": self.scaler.state_dict(),
					"ema": self.ema.state_dict(),
					"conditioner": csd
				}
				th.save(checkpoint, f)
		dist.barrier()
	

	def configure_optimizer(self):
		params = list(self.model.parameters())
		for embedder in self.conditioner.embedders:
			if embedder.is_trainable:
				params = params + list(embedder.parameters())
		opt = AdamW(
			params, lr=self.lr, weight_decay=self.weight_decay
		)
		if not self.zeroshot_adapt:
			opt.register_step_post_hook(
				lambda optimizer, args, kwargs: self.ema.update(self.ddp_model.parameters()) 
			)
		return opt


	def configure_scheduler(self, opt):
		if self.lr_scheduler == "warmup":
			warmup_steps = 5 * self.steps_per_epoch
			sch = th.optim.lr_scheduler.LinearLR(
				optimizer=opt, 
				start_factor=0.01,
				total_iters=warmup_steps
			)
		elif self.lr_scheduler == "warmup_cosine_anneal":
			assert self.num_epochs > 0
			warmup_steps = 5 * self.steps_per_epoch
			sch = th.optim.lr_scheduler.SequentialLR(
				optimizer=opt,
				schedulers=[
					th.optim.lr_scheduler.LinearLR(
						optimizer=opt, 
						start_factor=0.01,
						total_iters=warmup_steps
					),
					th.optim.lr_scheduler.CosineAnnealingLR(
						optimizer=opt,
						T_max=self.num_epochs * self.steps_per_epoch - warmup_steps,
						eta_min=0
					)
				],
				milestones=[warmup_steps]
			)
		elif self.lr_scheduler == "warmup_steplr":
			assert self.num_epochs > 0
			warmup_steps = 5 * self.steps_per_epoch
			sch = th.optim.lr_scheduler.SequentialLR(
				optimizer=opt,
				schedulers=[
					th.optim.lr_scheduler.LinearLR(
						optimizer=opt, 
						start_factor=0.01,
						total_iters=warmup_steps
					),
					th.optim.lr_scheduler.StepLR(
						optimizer=opt,
						step_size=60 * self.steps_per_epoch,
						gamma=0.1
					)
				],
				milestones=[warmup_steps]
			)
		elif self.lr_scheduler == "steplr":
			sch = th.optim.lr_scheduler.StepLR(
				optimizer=opt,
				step_size=150 * self.steps_per_epoch,
				gamma=0.1
			)
		elif self.lr_scheduler is None:
			sch = None
		else:
			raise ValueError(f"Unsupported lr_scheduler: {self.lr_scheduler}")
		return sch


	def compute_norms(self):
		grad_norm = 0.0
		param_norm = 0.0
		for p in self.ddp_model.parameters():
			with th.no_grad():
				param_norm += th.norm(p, p=2, dtype=th.float32).item() ** 2
				if p.grad is not None:
					grad_norm += th.norm(p.grad, p=2, dtype=th.float32).item() ** 2
		return np.sqrt(grad_norm), np.sqrt(param_norm)


def log_loss_dict(diffusion, ts, losses):
	for key, values in losses.items():
		logger.logkv_mean(key, values.mean().item())
		# Log the quantiles (four quartiles, in particular).
		for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
			quartile = int(4 * sub_t / diffusion.num_timesteps)
			logger.logkv_mean(f"{key}_q{quartile}", sub_loss)


def log_tgt(tgt, prefix="train", postfix="", step=None): #with_tlrb=True, only_tlrb=False):
	logger.logimg(tgt, f"{prefix}_dm_{postfix}", step)


def log_cond(cond, prefix="train", step=None):
	if "img" in cond:
		img = cond["img"]
		if "bboxes" in cond:
			img = draw_bboxes(img, cond["bboxes"])
		logger.logimg(img, f"{prefix}_img_cond", step=step)
	if "text" in cond:
		img = draw_text(cond["text"])
		logger.logimg(img, f"{prefix}_text_cond", step=step)
	if "sam_masks" in cond:
		logger.logimg(cond["sam_masks"], name=f"{prefix}_sam_masks", step=step)


def log_denoising_process(samples, diffusion, vae, t_step=None, step=None):
	t_step = diffusion.num_timesteps if t_step is None else t_step
	assert diffusion.num_timesteps % t_step == 0
	dev = "cpu"
	if vae is not None:
		dev = vae.device

	for i, s in enumerate(samples):
		_dm = s["sample"].to(dev)

		if i % t_step == 0 or i == diffusion.num_timesteps - 1:
			_dm = possibly_vae_decode(_dm, vae, clip_decoded=True)
			_dm = collate_channels(_dm, mode="mean")

	final = _dm.cpu()

	log_tgt(final, prefix="final", postfix="", step=step)
	logger.savetensor(final, "final", step)
	return final


def log_results(final, cond, step=None):
	results = []
	target_count = cond["count"].float()
	for j, f in enumerate(final):
		_dm = f

		pred_count, pred_coords = counting(_dm)
		res = draw_result(cond["img"][j], _dm, float(pred_count), target_count[j], pred_coords)
		results.append(res)

	logger.logimg(results, "results", step=step)
