import argparse
import torch as th
import os.path as osp

from torch.utils.data import default_collate

from diffcount import logger
from diffcount.ema import ExponentialMovingAverage
from diffcount.plot_util import draw_result
from diffcount.infer_util import (
	counting, 
	collate_channels, 
	eval_preprocess, 
	ttn,
)
from diffcount.nn import (
	torch_to, 
	possibly_vae_decode,
	freeze,
	count_params,
)
from diffcount.datasets import FSC147, MCAC, FSCD_LVIS
from diffcount.script_util import (
	create_model,
	create_diffusion,
	create_conditioner,
	create_vae,
	parse_config,
	seed_everything
)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def main():
	args = parse_args()
	expdir = args.expdir
	config = parse_config(osp.join(expdir, "config.yaml"))
	dev = "cuda" if th.cuda.is_available() else "cpu"
	ckpt = th.load(osp.join(expdir, args.checkpoint), map_location=dev)

	if args.seed is not None:
		seed_everything(args.seed)

	config.model.params.topk = args.topk

	logger.configure(
		dir=expdir, 
		format_strs=['stdout', 'log'],
		log_suffix=f'_sample'
	)
	logger.set_mediadir("samples")

	logger.log("creating model...")
	model = create_model(config.model)
	model.to(dev)
	model.load_state_dict(ckpt["model"], strict=False)
	model.eval()
	model = freeze(model)
	logger.log(f"Model has: {count_params(model, verbose=False)} number of parameters")

	logger.log("creating diffusion...")
	config.diffusion.params.timestep_respacing = args.timestep_respacing
	diffusion = create_diffusion(config.diffusion)

	logger.log("creating VAE...")
	vae = create_vae(
		getattr(config, "vae", None), device=dev
	)

	logger.log("creating data...")
	if args.n_exemplars is not None:
		config.data.dataset.params.n_exemplars = args.n_exemplars
	if args.mcac:
		sampling_data = MCAC(
			**vars(config.data.dataset.params),
			split="sample"
		)
	elif args.lvis:
		sampling_data = FSCD_LVIS(
			**vars(config.data.dataset.params),
			split="sample"
		)
	else:
		sampling_data = FSC147(
			**vars(config.data.dataset.params),
			split="sample"
		)

	logger.log("creating conditioner...")
	conditioner = create_conditioner(
		getattr(config, "conditioner", []),
		train=False
	)
	conditioner.to(dev)
	conditioner.load_state_dict(ckpt["conditioner"], strict=False)
	conditioner.eval()
	conditioner = freeze(conditioner)

	logger.log("creating EMA...")
	ema = ExponentialMovingAverage(
		model.parameters(),
		decay=config.train.ema_rate
	)
	ema.load_state_dict(ckpt["ema"])

	imgs = [_id if args.mcac else f"{_id}.jpg" for _id in args.ids]
	assert len(imgs) > 0, ("Please provide a list of images to sample.")

	ch_mult = vae.config.latent_channels if vae else 1
	with_tlrb = config.data.dataset.params.with_tlrb
	ich = 5 * ch_mult if with_tlrb else ch_mult
	if with_tlrb and config.train.only_tlrb:
		ich = ich - 1

	use_cfg = args.cfg_scale > 0.0
	use_zero_shot = args.zero_shot or args.n_exemplars == 0
	assert not (use_cfg and use_zero_shot)
	sample_fn = (
		diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
	)
	model_fn = (
		model if not use_cfg else model.forward_with_cfg
	)
	fsizes = [None] if args.force_resize is None else args.force_resize

	logger.log("sampling...")
	for k in imgs:
		for nt in range(args.ntimes):
			for siz in fsizes:
				try:
					tgt, cond = sampling_data.get_by_name(k)
				except ValueError:
					logger.log(f"'{k}' was not found in val/test set. Skipping...")
					continue

				tgt = default_collate([tgt])
				cond = default_collate([cond])

				tgt = torch_to(tgt, dev)
				cond = torch_to(cond, dev)

				target_count = cond["count"].float().cpu().item()
				_id = cond["id"][0]

				tcond = eval_preprocess(
					cond.copy(), 
					allow_resizing=args.allow_resizing,
					force_resize=siz,
				)

				bs, _, imgh, _ = tcond["img"].shape

				with th.no_grad():
					c, uc = conditioner.get_unconditional_conditioning(
						tcond,
						ucond=None,
						force_uc_zero_embeddings=["bboxes"],
						force_cond_zero_embeddings=None,
						vae=vae
					)
					h, w = c["concat"].shape[-2:]
					ch = ich
					z = diffusion.get_init_sample((bs, ch, h, w), device=dev)

					if use_cfg:
						z = th.cat((z, z), dim=0)
						assert c.keys() == uc.keys(), (
							"conditional and unctonditional conditioning dictionaries must have same keys"
						)
						_cond = {k: th.cat((c[k], uc[k]), dim=0) for k in c}
								
					elif use_zero_shot:
						_cond = uc
					else:
						_cond = c

					all_samples = []
					with th.autocast(device_type=dev, dtype=th.float16, enabled=args.use_fp16):
						with ema.average_parameters(model.parameters()):
							for sample in diffusion.p_sample_loop_progressive(
								model=model_fn,
								shape=z.shape,
								noise=z,
								clip_denoised=False,
								denoised_fn=None,
								model_kwargs=dict(
									cond=_cond,
									cfg_scale=args.cfg_scale
								),
								device=dev,
								progress=False
							):
								all_samples.append(sample["sample"])
					all_samples = th.cat(all_samples, dim=0)
					logger.savetensor(
						all_samples,
						name=f"{_id}",
						step="samples"
					)
					samples = all_samples[-1, ...].unsqueeze(0)
					if use_cfg:
						samples, _ = samples.chunk(2, dim=0)

					if args.save_latents:
						logger.savetensor(
							samples,
							name=f"{_id}", 
							step="latents"
						)

					dm = possibly_vae_decode(samples, vae, clip_decoded=True)
					savable_dm = dm.clone().detach().cpu()
					dm = collate_channels(dm, mode="mean")
					dm = dm.squeeze(0)

					count, coords = counting(dm)
					cond = tcond

					ttn_fact = 1
					if args.allow_ttn:
						ttn_fact = ttn(coords, cond["bboxes"].cpu())
						count = count / ttn_fact
					isttn = "ttn" if ttn_fact > 1 else ""


					count_error = int(abs(target_count - count))
					res = draw_result(cond["img"][0], dm, float(count), target_count, coords)

					logger.logimg(
						res, 
						name=f"{_id}_{nt}_{imgh}_{count_error}_{isttn}",
						step="imgs"
					)

					logger.log(f"{_id}_{nt}_{imgh}_{count_error}_{isttn}")
					if args.save_densities:
						logger.savetensor(
							savable_dm,
							name=f"{_id}_{nt}_{imgh}_{count_error}_{isttn}", 
							step="dms"
						)

					logger.log(str(k) + "  \u2713")


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--expdir", type=str)
	parser.add_argument("--checkpoint", type=str)
	parser.add_argument("--timestep_respacing", type=str, default="")
	parser.add_argument("--use_ddim", action="store_true")
	parser.add_argument("--use_fp16", action="store_true")
	parser.add_argument("--cfg_scale", type=float, default=0.0)
	parser.add_argument("--zero_shot", action="store_true")
	parser.add_argument("--seed", type=int, default=None)
	parser.add_argument("--ids", nargs="+", type=str, default=[])
	parser.add_argument("--save_densities", action="store_true")
	parser.add_argument("--save_latents", action="store_true")
	parser.add_argument("--allow_tiling", action="store_true")
	parser.add_argument("--allow_resizing", action="store_true")
	parser.add_argument("--allow_ttn", action="store_true")
	parser.add_argument("--force_resize", nargs="+", type=int, default=None)
	parser.add_argument("--mcac", action="store_true")
	parser.add_argument("--lvis", action="store_true")
	parser.add_argument("--topk", type=int, default=-1)
	parser.add_argument("--ntimes", type=int, default=1, help="Number of times to sample the same image")
	parser.add_argument("--n_exemplars", type=int, default=None)
	return parser.parse_args()


if __name__ == "__main__":
	main()