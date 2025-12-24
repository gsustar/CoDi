"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math

import numpy as np
import torch as th

from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood #generalized_iou_loss
from .base_diffusion import BaseDiffusion


def shifted_sigmoid_cumprod(timesteps, shift=0.7, scale=10.0):
	steps = np.linspace(0, 1, timesteps)
	sigmoid = 1 / (1 + np.exp(-scale * (steps - shift)))
	one_minus_alpha_cumprod = (sigmoid - sigmoid.min()) / (sigmoid.max() - sigmoid.min())
	alpha_cumprod = 1.0 - one_minus_alpha_cumprod
	alpha_cumprod = np.clip(alpha_cumprod, 1e-8, 1.0)
	alpha_cumprod = np.maximum.accumulate(alpha_cumprod[::-1])[::-1]
	alpha_prev = np.append(1.0, alpha_cumprod[:-1])
	alphas = alpha_cumprod / alpha_prev
	betas = 1.0 - alphas
	betas[0] = 0.0001
	betas[-1] = 1.000
	return betas, alphas, alpha_cumprod


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, max_beta=0.999):
	"""
	Get a pre-defined beta schedule for the given name.

	The beta schedule library consists of beta schedules which remain similar
	in the limit of num_diffusion_timesteps.
	Beta schedules may be added, but should not be removed or changed once
	they are committed to maintain backwards compatibility.
	"""
	if schedule_name == "linear":
		# Linear schedule from Ho et al, extended to work for any number of
		# diffusion steps.
		scale = 1000 / num_diffusion_timesteps
		beta_start = scale * 0.0001
		beta_end = scale * 0.02
		return np.linspace(
			beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
		)
	elif schedule_name == "scaled_linear":
		scale = 1000 / num_diffusion_timesteps
		beta_start = scale * 0.0001
		beta_end = scale * 0.02
		return np.linspace(
			beta_start**0.5, beta_end**0.5, num_diffusion_timesteps, dtype=np.float64
		) ** 2
	elif schedule_name == "cosine":
		return betas_for_alpha_bar(
			num_diffusion_timesteps,
			lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
			max_beta=max_beta
		)
	elif schedule_name == "shifted_sigmoid":
		return shifted_sigmoid_cumprod(
			timesteps=num_diffusion_timesteps, shift=0.7, scale=15.0
		)[0]
	else:
		raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
	"""
	Create a beta schedule that discretizes the given alpha_t_bar function,
	which defines the cumulative product of (1-beta) over time from t = [0,1].

	:param num_diffusion_timesteps: the number of betas to produce.
	:param alpha_bar: a lambda that takes an argument t from 0 to 1 and
					  produces the cumulative product of (1-beta) up to that
					  part of the diffusion process.
	:param max_beta: the maximum beta to use; use values lower than 1 to
					 prevent singularities.
	"""
	betas = []
	for i in range(num_diffusion_timesteps):
		t1 = i / num_diffusion_timesteps
		t2 = (i + 1) / num_diffusion_timesteps
		betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
	return np.array(betas)


def get_t_weighting_scheme(name, num_difusion_timesteps, snr):
	if name == "p2":
		k, gamma = 1.0, 0.5
		t_weights = 1.0 / (k + snr) ** gamma
	elif name == "exp":
		k = 40.0
		t_weights = (
			1 - (np.exp(-k * (np.arange(num_difusion_timesteps) / num_difusion_timesteps)) - 1) / (np.exp(-k) - 1)
		)
	elif name == "uniform":
		t_weights = np.ones(num_difusion_timesteps)
	else:
		raise ValueError(f"Unknown t_count_weighting_scheme: {name}")
	return t_weights


def enforce_zero_terminal_snr(betas):
	# Convert betas to sqrt_alphas_cumprod
	alphas = 1.0 - betas
	alphas_cumprod = np.cumprod(alphas, axis=0)
	sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)

	# Store old values.
	sqrt_alphas_cumprod_0 = sqrt_alphas_cumprod[0].copy()
	sqrt_alphas_cumprod_T = sqrt_alphas_cumprod[-1].copy()

	# Shift so last timestep is zero.
	sqrt_alphas_cumprod -= sqrt_alphas_cumprod_T
	# Scale so first timestep is back to old value.
	sqrt_alphas_cumprod *= sqrt_alphas_cumprod_0 / (
		sqrt_alphas_cumprod_0 - sqrt_alphas_cumprod_T
	)

	# Convert alphas_bar_sqrt to betas
	alphas_cumprod = sqrt_alphas_cumprod ** 2
	alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
	alphas = np.concatenate((alphas_cumprod[0:1], alphas))
	# alphas = th.cat([alphas_cumprod[0:1], alphas])
	betas = 1 - alphas
	return betas


class ModelMeanType(enum.Enum):
	"""
	Which type of output the model predicts.
	"""

	PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
	START_X = enum.auto()  # the model predicts x_0
	EPSILON = enum.auto()  # the model predicts epsilon
	V = enum.auto()


class ModelVarType(enum.Enum):
	"""
	What is used as the model's output variance.

	The LEARNED_RANGE option has been added to allow the model to predict
	values between FIXED_SMALL and FIXED_LARGE, making its job easier.
	"""

	LEARNED = enum.auto()
	FIXED_SMALL = enum.auto()
	FIXED_LARGE = enum.auto()
	LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
	MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
	RESCALED_MSE = enum.auto()  # use raw MSE loss (with RESCALED_KL when learning variances)
	KL = enum.auto()  # use the variational lower-bound
	RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

	def is_vb(self):
		return self == LossType.KL or self == LossType.RESCALED_KL


class DenoiseDiffusion(BaseDiffusion):
	"""
	Utilities for training and sampling diffusion models.

	Ported directly from here, and then adapted over time to further experimentation.
	https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

	:param betas: a 1-D numpy array of betas for each diffusion timestep,
				  starting at T and going to 1.
	:param model_mean_type: a ModelMeanType determining what the model outputs.
	:param model_var_type: a ModelVarType determining how variance is output.
	:param loss_type: a LossType determining the loss function to use.
	:param rescale_timesteps: if True, pass floating point timesteps into the
							  model so that they are always scaled like in the
							  original paper (0 to 1000).
	"""

	def __init__(
		self,
		*,
		betas,
		model_mean_type,
		model_var_type,
		loss_type,
		rescale_timesteps=False,
		lmbd_vlb=0.001,
		lmbd_cb_count=0.0,
		t_mse_weighting_scheme="uniform",
		t_cb_count_weighting_scheme="uniform",
		do_giou_loss=False,
		weighted_tlrb_loss=False,
		apply_multi_res_noise=False,
	):
		self.model_mean_type = model_mean_type
		self.model_var_type = model_var_type
		self.loss_type = loss_type
		self.rescale_timesteps = rescale_timesteps

		self.lmbd_vb = lmbd_vlb
		self.lmbd_cb_count = lmbd_cb_count
		self.t_mse_weighting_scheme = t_mse_weighting_scheme
		self.t_cb_count_weighting_scheme = t_cb_count_weighting_scheme

		self.do_giou_loss = do_giou_loss
		self.weighted_tlrb_loss = weighted_tlrb_loss
		self.apply_multi_res_noise = apply_multi_res_noise

		# Use float64 for accuracy.

		betas = np.array(betas, dtype=np.float64)
		self.betas = betas
		assert len(betas.shape) == 1, "betas must be 1-D"
		assert (betas > 0).all() and (betas <= 1).all()

		self.num_timesteps = int(betas.shape[0])
		alphas = 1.0 - betas
		self.alphas_cumprod = np.cumprod(alphas, axis=0)
		self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
		self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
		assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

		# calculations for diffusion q(x_t | x_{t-1}) and others
		self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
		self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
		self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
		self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
		self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

		# calculations for posterior q(x_{t-1} | x_t, x_0)
		self.posterior_variance = (
			betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
		)
		# log calculation clipped because the posterior variance is 0 at the
		# beginning of the diffusion chain.
		self.posterior_log_variance_clipped = np.log(
			np.append(self.posterior_variance[1], self.posterior_variance[1:])
		)
		self.posterior_mean_coef1 = (
			betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
		)
		self.posterior_mean_coef2 = (
			(1.0 - self.alphas_cumprod_prev)
			* np.sqrt(alphas)
			/ (1.0 - self.alphas_cumprod)
		)

		self.snr = 1.0 / (1 - self.alphas_cumprod) - 1
		self.t_mse_weights = get_t_weighting_scheme(
			self.t_mse_weighting_scheme, 
			self.num_timesteps, 
			self.snr
		)
		self.t_cb_count_weights = get_t_weighting_scheme(
			self.t_cb_count_weighting_scheme,
			self.num_timesteps,
			self.snr
		)


	def q_mean_variance(self, x_start, t):
		"""
		Get the distribution q(x_t | x_0).

		:param x_start: the [N x C x ...] tensor of noiseless inputs.
		:param t: the number of diffusion steps (minus 1). Here, 0 means one step.
		:return: A tuple (mean, variance, log_variance), all of x_start's shape.
		"""
		mean = (
			_extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
		)
		variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
		log_variance = _extract_into_tensor(
			self.log_one_minus_alphas_cumprod, t, x_start.shape
		)
		return mean, variance, log_variance

	def q_sample(self, x_start, t, noise=None):
		"""
		Diffuse the data for a given number of diffusion steps.

		In other words, sample from q(x_t | x_0).

		:param x_start: the initial data batch.
		:param t: the number of diffusion steps (minus 1). Here, 0 means one step.
		:param noise: if specified, the split-out normal noise.
		:return: A noisy version of x_start.
		"""
		if noise is None:
			noise = th.randn_like(x_start)
		assert noise.shape == x_start.shape
		return (
			_extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
			+ _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
			* noise
		)

	def q_posterior_mean_variance(self, x_start, x_t, t):
		"""
		Compute the mean and variance of the diffusion posterior:

			q(x_{t-1} | x_t, x_0)

		"""
		assert x_start.shape == x_t.shape
		posterior_mean = (
			_extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
			+ _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
		)
		posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
		posterior_log_variance_clipped = _extract_into_tensor(
			self.posterior_log_variance_clipped, t, x_t.shape
		)
		assert (
			posterior_mean.shape[0]
			== posterior_variance.shape[0]
			== posterior_log_variance_clipped.shape[0]
			== x_start.shape[0]
		)
		return posterior_mean, posterior_variance, posterior_log_variance_clipped

	def p_mean_variance(
		self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
	):
		"""
		Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
		the initial x, x_0.

		:param model: the model, which takes a signal and a batch of timesteps
					  as input.
		:param x: the [N x C x ...] tensor at time t.
		:param t: a 1-D Tensor of timesteps.
		:param clip_denoised: if True, clip the denoised signal into [-1, 1].
		:param denoised_fn: if not None, a function which applies to the
			x_start prediction before it is used to sample. Applies before
			clip_denoised.
		:param model_kwargs: if not None, a dict of extra keyword arguments to
			pass to the model. This can be used for conditioning.
		:return: a dict with the following keys:
				 - 'mean': the model mean output.
				 - 'variance': the model variance output.
				 - 'log_variance': the log of 'variance'.
				 - 'pred_xstart': the prediction for x_0.
		"""
		if model_kwargs is None:
			model_kwargs = {}

		B, C = x.shape[:2]
		assert t.shape == (B,)
		model_output = model(x, self._scale_timesteps(t), **model_kwargs)

		model_cb_count = None
		model_dethead_f = None
		if isinstance(model_output, dict):
			model_cb_count = model_output.get("count", None)
			model_dethead_f = model_output.get("dethead_f", None)
			model_output = model_output["out"]

		if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
			assert model_output.shape == (B, C * 2, *x.shape[2:])
			model_output, model_var_values = th.split(model_output, C, dim=1)
			if self.model_var_type == ModelVarType.LEARNED:
				model_log_variance = model_var_values
				model_variance = th.exp(model_log_variance)
			else:
				min_log = _extract_into_tensor(
					self.posterior_log_variance_clipped, t, x.shape
				)
				max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
				# The model_var_values is [-1, 1] for [min_var, max_var].
				frac = (model_var_values + 1) / 2
				model_log_variance = frac * max_log + (1 - frac) * min_log
				model_variance = th.exp(model_log_variance)
		else:
			model_variance, model_log_variance = {
				# for fixedlarge, we set the initial (log-)variance like so
				# to get a better decoder log likelihood.
				ModelVarType.FIXED_LARGE: (
					np.append(self.posterior_variance[1], self.betas[1:]),
					np.log(np.append(self.posterior_variance[1], self.betas[1:])),
				),
				ModelVarType.FIXED_SMALL: (
					self.posterior_variance,
					self.posterior_log_variance_clipped,
				),
			}[self.model_var_type]
			model_variance = _extract_into_tensor(model_variance, t, x.shape)
			model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

		def process_xstart(x):
			if denoised_fn is not None:
				x = denoised_fn(x)
			if clip_denoised:
				return x.clamp(-1, 1)
			return x

		if self.model_mean_type == ModelMeanType.PREVIOUS_X:
			pred_xstart = process_xstart(
				self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
			)
			model_mean = model_output
		elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON, ModelMeanType.V]:
			if self.model_mean_type == ModelMeanType.START_X:
				pred_xstart = process_xstart(model_output)
			elif self.model_mean_type == ModelMeanType.V:
				pred_xstart = process_xstart(
					self._predict_start_from_v(x_t=x, t=t, v=model_output)
				)
			else:
				pred_xstart = process_xstart(
					self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
				)
			model_mean, _, _ = self.q_posterior_mean_variance(
				x_start=pred_xstart, x_t=x, t=t
			)
		else:
			raise NotImplementedError(self.model_mean_type)

		assert (
			model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
		)
		return {
			"mean": model_mean,
			"variance": model_variance,
			"log_variance": model_log_variance,
			"pred_xstart": pred_xstart,
			"dethead_f": model_dethead_f,
		}

	def _predict_v(self, x_start, t, noise):
		return (
			_extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
			_extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
		)

	def _predict_start_from_v(self, x_t, t, v):
		return (
			_extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
			_extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
		)

	def _predict_xstart_from_eps(self, x_t, t, eps):
		assert x_t.shape == eps.shape
		return (
			_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
			- _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
		)

	def _predict_xstart_from_xprev(self, x_t, t, xprev):
		assert x_t.shape == xprev.shape
		return (  # (xprev - coef2*x_t) / coef1
			_extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
			- _extract_into_tensor(
				self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
			)
			* x_t
		)

	def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
		return (
			_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
			- pred_xstart
		) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

	def _scale_timesteps(self, t):
		if self.rescale_timesteps:
			return t.float() * (1000.0 / self.num_timesteps)
		return t
	
	def get_init_sample(self, shape, device, noise=None):
		if noise is not None:
			return noise
		return th.randn(*shape, device=device)

	def p_sample(
		self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
	):
		"""
		Sample x_{t-1} from the model at the given timestep.

		:param model: the model to sample from.
		:param x: the current tensor at x_{t-1}.
		:param t: the value of t, starting at 0 for the first diffusion step.
		:param clip_denoised: if True, clip the x_start prediction to [-1, 1].
		:param denoised_fn: if not None, a function which applies to the
			x_start prediction before it is used to sample.
		:param model_kwargs: if not None, a dict of extra keyword arguments to
			pass to the model. This can be used for conditioning.
		:return: a dict containing the following keys:
				 - 'sample': a random sample from the model.
				 - 'pred_xstart': a prediction of x_0.
		"""
		out = self.p_mean_variance(
			model,
			x,
			t,
			clip_denoised=clip_denoised,
			denoised_fn=denoised_fn,
			model_kwargs=model_kwargs,
		)
		noise = th.randn_like(x)
		nonzero_mask = (
			(t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
		)  # no noise when t == 0
		sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
		return {"sample": sample, "pred_xstart": out["pred_xstart"], "dethead_f": out["dethead_f"]}


	def ddim_sample(
		self,
		model,
		x,
		t,
		clip_denoised=True,
		denoised_fn=None,
		model_kwargs=None,
		eta=0.0,
	):
		"""
		Sample x_{t-1} from the model using DDIM.

		Same usage as p_sample().
		"""
		out = self.p_mean_variance(
			model,
			x,
			t,
			clip_denoised=clip_denoised,
			denoised_fn=denoised_fn,
			model_kwargs=model_kwargs,
		)
		# Usually our model outputs epsilon, but we re-derive it
		# in case we used x_start or x_prev prediction.
		eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
		alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
		alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
		sigma = (
			eta
			* th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
			* th.sqrt(1 - alpha_bar / alpha_bar_prev)
		)
		# Equation 12.
		noise = th.randn_like(x)
		mean_pred = (
			out["pred_xstart"] * th.sqrt(alpha_bar_prev)
			+ th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
		)
		nonzero_mask = (
			(t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
		)  # no noise when t == 0
		sample = mean_pred + nonzero_mask * sigma * noise
		return {"sample": sample, "pred_xstart": out["pred_xstart"]}

	def ddim_reverse_sample(
		self,
		model,
		x,
		t,
		clip_denoised=True,
		denoised_fn=None,
		model_kwargs=None,
		eta=0.0,
	):
		"""
		Sample x_{t+1} from the model using DDIM reverse ODE.
		"""
		assert eta == 0.0, "Reverse ODE only for deterministic path"
		out = self.p_mean_variance(
			model,
			x,
			t,
			clip_denoised=clip_denoised,
			denoised_fn=denoised_fn,
			model_kwargs=model_kwargs,
		)
		# Usually our model outputs epsilon, but we re-derive it
		# in case we used x_start or x_prev prediction.
		eps = (
			_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
			- out["pred_xstart"]
		) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
		alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

		# Equation 12. reversed
		mean_pred = (
			out["pred_xstart"] * th.sqrt(alpha_bar_next)
			+ th.sqrt(1 - alpha_bar_next) * eps
		)

		return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

	def _vb_terms_bpd(
		self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
	):
		"""
		Get a term for the variational lower-bound.

		The resulting units are bits (rather than nats, as one might expect).
		This allows for comparison to other papers.

		:return: a dict with the following keys:
				 - 'output': a shape [N] tensor of NLLs or KLs.
				 - 'pred_xstart': the x_0 predictions.
		"""
		true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
			x_start=x_start, x_t=x_t, t=t
		)
		out = self.p_mean_variance(
			model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
		)
		kl = normal_kl(
			true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
		)
		kl = mean_flat(kl) / np.log(2.0)

		decoder_nll = -discretized_gaussian_log_likelihood(
			x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
		)
		assert decoder_nll.shape == x_start.shape
		decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

		# At the first timestep return the decoder NLL,
		# otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
		output = th.where((t == 0), decoder_nll, kl)
		return {"output": output, "pred_xstart": out["pred_xstart"]}

	def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
		"""
		Compute training losses for a single timestep.

		:param model: the model to evaluate loss on.
		:param x_start: the [N x C x ...] tensor of inputs.
		:param t: a batch of timestep indices.
		:param model_kwargs: if not None, a dict of extra keyword arguments to
			pass to the model. This can be used for conditioning.
		:param noise: if specified, the specific Gaussian noise to try to remove.
		:return: a dict with the key "loss" containing a tensor of shape [N].
				 Some mean or variance settings may also have other keys.
		"""
		if model_kwargs is None:
			model_kwargs = {}
		if noise is None:
			noise = th.randn_like(x_start)
		x_t = self.q_sample(x_start, t, noise=noise)

		terms = {}
		
		target_count = model_kwargs.pop("count").unsqueeze(1)

		if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
			terms["loss"] = self._vb_terms_bpd(
				model=model,
				x_start=x_start,
				x_t=x_t,
				t=t,
				clip_denoised=False,
				model_kwargs=model_kwargs,
			)["output"]
			if self.loss_type == LossType.RESCALED_KL:
				terms["loss"] *= self.num_timesteps
		elif (self.loss_type == LossType.MSE or 
			  self.loss_type == LossType.RESCALED_MSE):
			
			model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

			model_cb_count = None
			if isinstance(model_output, dict):
				model_cb_count = model_output.get("count", None)
				model_output = model_output["out"]

			if self.model_var_type in [
				ModelVarType.LEARNED,
				ModelVarType.LEARNED_RANGE,
			]:
				B, C = x_t.shape[:2]
				assert model_output.shape == (B, C * 2, *x_t.shape[2:])
				model_output, model_var_values = th.split(model_output, C, dim=1)
				# Learn the variance using the variational bound, but don't let
				# it affect our mean prediction.
				frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
				_vb = self._vb_terms_bpd(
					model=lambda *args, r=frozen_out: dict(out=r),
					x_start=x_start,
					x_t=x_t,
					t=t,
					clip_denoised=False,
				)

				if self.loss_type == LossType.RESCALED_MSE:
					# Divide by 1000 for equivalence with initial implementation.
					# Without a factor of 1/1000, the VB term hurts the MSE term.
					terms["vb"] *= self.num_timesteps / 1000.0

				terms["vb"] = _vb["output"] * self.lmbd_vb

			target = {
				ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
					x_start=x_start, x_t=x_t, t=t
				)[0],
				ModelMeanType.V: self._predict_v(
					x_start=x_start, t=t, noise=noise
				),
				ModelMeanType.START_X: x_start,
				ModelMeanType.EPSILON: noise,
			}[self.model_mean_type]
			assert model_output.shape == target.shape == x_start.shape

			lmbd_t_mse = _extract_into_tensor(self.t_mse_weights, t, target.shape)
			terms["mse"] = mean_flat(lmbd_t_mse * (model_output - target)**2)


			if model_cb_count is not None and self.lmbd_cb_count > 0.0:
				lmbd_t_cb_count = _extract_into_tensor(self.t_cb_count_weights, t, target_count.shape)
				terms["cb_count"] = self.lmbd_cb_count * mean_flat(lmbd_t_cb_count * abs(target_count - model_cb_count))

			terms["loss"] = terms["mse"]

			if "cb_count" in terms:
				terms["loss"] = terms["loss"] + terms["cb_count"]

			if "vb" in terms:
				terms["loss"] = terms["loss"] + terms["vb"]
		else:
			raise NotImplementedError(self.loss_type)

		return terms

	def _prior_bpd(self, x_start):
		"""
		Get the prior KL term for the variational lower-bound, measured in
		bits-per-dim.

		This term can't be optimized, as it only depends on the encoder.

		:param x_start: the [N x C x ...] tensor of inputs.
		:return: a batch of [N] KL values (in bits), one per batch element.
		"""
		batch_size = x_start.shape[0]
		t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
		qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
		kl_prior = normal_kl(
			mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
		)
		return mean_flat(kl_prior) / np.log(2.0)

	def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
		"""
		Compute the entire variational lower-bound, measured in bits-per-dim,
		as well as other related quantities.

		:param model: the model to evaluate loss on.
		:param x_start: the [N x C x ...] tensor of inputs.
		:param clip_denoised: if True, clip denoised samples.
		:param model_kwargs: if not None, a dict of extra keyword arguments to
			pass to the model. This can be used for conditioning.

		:return: a dict containing the following keys:
				 - total_bpd: the total variational lower-bound, per batch element.
				 - prior_bpd: the prior term in the lower-bound.
				 - vb: an [N x T] tensor of terms in the lower-bound.
				 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
				 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
		"""
		device = x_start.device
		batch_size = x_start.shape[0]

		vb = []
		xstart_mse = []
		mse = []
		for t in list(range(self.num_timesteps))[::-1]:
			t_batch = th.tensor([t] * batch_size, device=device)
			noise = th.randn_like(x_start)
			x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
			# Calculate VLB term at the current timestep
			with th.no_grad():
				out = self._vb_terms_bpd(
					model,
					x_start=x_start,
					x_t=x_t,
					t=t_batch,
					clip_denoised=clip_denoised,
					model_kwargs=model_kwargs,
				)
			vb.append(out["output"])
			xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
			eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
			mse.append(mean_flat((eps - noise) ** 2))

		vb = th.stack(vb, dim=1)
		xstart_mse = th.stack(xstart_mse, dim=1)
		mse = th.stack(mse, dim=1)

		prior_bpd = self._prior_bpd(x_start)
		total_bpd = vb.sum(dim=1) + prior_bpd
		return {
			"total_bpd": total_bpd,
			"prior_bpd": prior_bpd,
			"vb": vb,
			"xstart_mse": xstart_mse,
			"mse": mse,
		}


def _extract_into_tensor(arr, timesteps, broadcast_shape):
	"""
	Extract values from a 1-D numpy array for a batch of indices.

	:param arr: the 1-D numpy array.
	:param timesteps: a tensor of indices into the array to extract.
	:param broadcast_shape: a larger shape of K dimensions with the batch
							dimension equal to the length of timesteps.
	:return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
	"""
	res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
	while len(res.shape) < len(broadcast_shape):
		res = res[..., None]
	return res.expand(broadcast_shape)


def hardsigmoid(x):
	return th.min(th.ones_like(x), th.max(th.zeros_like(x), (x + 1) / 2))


def fix_boxes(boxes):
    # Ensure x_min <= x_max and y_min <= y_max
    x1 = th.min(boxes[..., 0], boxes[..., 2])
    y1 = th.min(boxes[..., 1], boxes[..., 3])
    x2 = th.max(boxes[..., 0], boxes[..., 2])
    y2 = th.max(boxes[..., 1], boxes[..., 3])
    return th.stack([x1, y1, x2, y2], dim=2)