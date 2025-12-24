import torch as th


class BaseDiffusion:

	def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
		raise NotImplementedError
	
	def get_init_sample(self, shape, device, noise=None):
		raise NotImplementedError
	
	def q_sample(self, x_start, t, noise=None):
		raise NotImplementedError
	
	def p_sample(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
		raise NotImplementedError
	
	def ddim_sample(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, eta=0.0,):
		raise NotImplementedError

	def p_sample_loop_progressive(
		self,
		model,
		shape,
		noise=None,
		clip_denoised=True,
		denoised_fn=None,
		model_kwargs=None,
		device=None,
		progress=False,
	):
		"""
		Generate samples from the model and yield intermediate samples from
		each timestep of diffusion.

		Arguments are the same as p_sample_loop().
		Returns a generator over dicts, where each dict is the return value of
		p_sample().
		"""
		if device is None:
			device = next(model.parameters()).device
		assert isinstance(shape, (tuple, list))

		img = self.get_init_sample(shape, device, noise=noise)
		indices = list(range(self.num_timesteps))[::-1]

		if progress:
			# Lazy import so that we don't depend on tqdm.
			from tqdm.auto import tqdm

			indices = tqdm(indices)

		for i in indices:
			t = th.tensor([i] * shape[0], device=device)
			# with th.no_grad():
			out = self.p_sample(
				model,
				img,
				t,
				clip_denoised=clip_denoised,
				denoised_fn=denoised_fn,
				model_kwargs=model_kwargs,
			)
			yield out
			img = out["sample"]
		
	def p_sample_loop(
		self,
		model,
		shape,
		noise=None,
		clip_denoised=True,
		denoised_fn=None,
		model_kwargs=None,
		device=None,
		progress=False,
	):
		"""
		Generate samples from the model.

		:param model: the model module.
		:param shape: the shape of the samples, (N, C, H, W).
		:param noise: if specified, the noise from the encoder to sample.
					Should be of the same shape as `shape`.
		:param clip_denoised: if True, clip x_start predictions to [-1, 1].
		:param denoised_fn: if not None, a function which applies to the
			x_start prediction before it is used to sample.
		:param model_kwargs: if not None, a dict of extra keyword arguments to
			pass to the model. This can be used for conditioning.
		:param device: if specified, the device to create the samples on.
					If not specified, use a model parameter's device.
		:param progress: if True, show a tqdm progress bar.
		:return: a non-differentiable batch of samples.
		"""
		final = None
		for sample in self.p_sample_loop_progressive(
			model,
			shape,
			noise=noise,
			clip_denoised=clip_denoised,
			denoised_fn=denoised_fn,
			model_kwargs=model_kwargs,
			device=device,
			progress=progress,
		):
			final = sample
		return final["sample"]
	
	def ddim_sample_loop_progressive(
		self,
		model,
		shape,
		noise=None,
		clip_denoised=True,
		denoised_fn=None,
		model_kwargs=None,
		device=None,
		progress=False,
		eta=0.0,
	):
		"""
		Use DDIM to sample from the model and yield intermediate samples from
		each timestep of DDIM.

		Same usage as p_sample_loop_progressive().
		"""
		if device is None:
			device = next(model.parameters()).device
		assert isinstance(shape, (tuple, list))
		img = self.get_init_sample(shape, device, noise=noise)
		indices = list(range(self.num_timesteps))[::-1]

		if progress:
			# Lazy import so that we don't depend on tqdm.
			from tqdm.auto import tqdm

			indices = tqdm(indices)

		for i in indices:
			t = th.tensor([i] * shape[0], device=device)
			with th.no_grad():
				out = self.ddim_sample(
					model,
					img,
					t,
					clip_denoised=clip_denoised,
					denoised_fn=denoised_fn,
					model_kwargs=model_kwargs,
					eta=eta,
				)
				yield out
				img = out["sample"]
	
	def ddim_sample_loop(
		self,
		model,
		shape,
		noise=None,
		clip_denoised=True,
		denoised_fn=None,
		model_kwargs=None,
		device=None,
		progress=False,
		eta=0.0,
	):
		"""
		Generate samples from the model using DDIM.

		Same usage as p_sample_loop().
		"""
		final = None
		for sample in self.ddim_sample_loop_progressive(
			model,
			shape,
			noise=noise,
			clip_denoised=clip_denoised,
			denoised_fn=denoised_fn,
			model_kwargs=model_kwargs,
			device=device,
			progress=progress,
			eta=eta,
		):
			final = sample
		return final["sample"]