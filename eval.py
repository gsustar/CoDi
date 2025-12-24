import argparse
import torch as th
import os.path as osp
import math
import json
import os
import time

from diffcount import logger
from diffcount.ema import ExponentialMovingAverage
from diffcount.plot_util import draw_result, draw_bboxes
from diffcount.infer_util import (
    counting, 
	collate_channels, 
	eval_preprocess, 
	ttn, 
)
from diffcount.nn import (
	torch_to, 
	possibly_vae_decode,
	count_params,
)
from diffcount.script_util import (
	create_model,
	create_diffusion,
	create_data,
	create_conditioner,
	create_vae,
	parse_config,
	seed_everything,
)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

ADD_NOISE_TO_EXEMPLARS = int(os.environ.get("ADD_NOISE_TO_EXEMPLARS", 0))

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
		log_suffix=f'_eval'
	)
	if args.trainset:
		logger.set_mediadir("trainset_predictions")
	else:
		logger.set_mediadir("results")

	logger.log("creating model...")
	model = create_model(config.model)
	model.to(dev)
	model.load_state_dict(ckpt["model"], strict=False)
	model.eval()
	logger.log(f"Model has: {count_params(model, verbose=False)} number of parameters")

	logger.log("creating diffusion...")
	config.diffusion.params.timestep_respacing = args.timestep_respacing
	diffusion = create_diffusion(config.diffusion)

	logger.log("creating VAE...")
	vae = create_vae(
		getattr(config, "vae", None), device=dev
	)

	logger.log("creating data...")
	config.data.dataloader.params.batch_size = 1
	if args.n_exemplars is not None:
		config.data.dataset.params.n_exemplars = args.n_exemplars
	_, val_data, test_data = create_data(config.data, train=False)
	if args.trainset:
		config.data.dataset.params.hflip_p = 0.0
		config.data.dataset.params.vflip_p = 0.0
		config.data.dataset.params.mosaic_p = 0.0
		train_data, _, _ = create_data(config.data, train=True)

	logger.log("creating conditioner...")
	conditioner = create_conditioner(
		getattr(config, "conditioner", []),
		train=False
	)
	conditioner.to(dev)
	conditioner.load_state_dict(ckpt["conditioner"], strict=False)
	conditioner.eval()

	logger.log("creating EMA...")
	ema = ExponentialMovingAverage(
		model.parameters(),
		decay=config.train.ema_rate
	)
	ema.load_state_dict(ckpt["ema"])

	ch_mult = vae.config.latent_channels if vae else 1
	with_tlrb = config.data.dataset.params.with_tlrb
	ich = 5 * ch_mult if with_tlrb else ch_mult

	use_cfg = args.cfg_scale > 0.0
	use_zero_shot = args.zero_shot or args.n_exemplars == 0
	assert not (use_cfg and use_zero_shot)
	sample_fn = (
		diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
	)
	model_fn = (
		model if not use_cfg else model.forward_with_cfg
	)
	splits = ["train"] if args.trainset else ["val", "test"]
	predpoints = {}
	dataset_folder = config.data.dataset.params.datadir

	times = []
	vram = []
	for split in splits:
		if config.data.dataset.name == "FSCD_LVIS" and config.data.dataset.params.unseen and split == "val":
			continue
		logger.log(f"Evaluating on {split.upper()} set...")
		predpoints[split] = {}

		eval_data = None
		if split == "val":
			eval_data = val_data
		elif split == "test":
			eval_data = test_data
		elif split == "train":
			eval_data = train_data
		else:
			raise ValueError("invalid split")

		N = len(eval_data.dataset)
		MAE = 0.0
		RMSE = 0.0
		i = 0
		for tgt, cond in eval_data:
			tgt = torch_to(tgt, dev)
			cond = torch_to(cond, dev)

			target_count = cond["count"].float().cpu().item()
			_id = cond["id"][0]
			og_h, og_w = cond["og_size"]
			th.cuda.reset_max_memory_allocated()
			start_time = time.time()
			tcond = eval_preprocess(
				cond.copy(), 
				# max_downscale_factor=2*len(model.channel_mult) * VAE_DOWNSCALE_FACTOR, 
				# allow_tiling=args.allow_tiling, 
				allow_resizing=args.allow_resizing,
			)
			# do_tile = tiler is not None

			# bs, _, h, w = cond["img"].shape
			bs, _, h, w = tcond["img"].shape
			# bs, ch, h, w = 1, ich, h // VAE_DOWNSCALE_FACTOR, w // VAE_DOWNSCALE_FACTOR
			# z = th.randn(bs, ch, h, w, device=dev)
			if ADD_NOISE_TO_EXEMPLARS > 0:
				pct = 0.05
				# tcond["bboxes"] = tcond["bboxes"] + th.randn_like(tcond["bboxes"]) * ADD_NOISE_TO_EXEMPLARS
				exemplars_w = tcond["bboxes"][:, :, 2] - tcond["bboxes"][:, :, 0]
				exemplars_h = tcond["bboxes"][:, :, 3] - tcond["bboxes"][:, :, 1]
				w_noise = th.randn_like(tcond["bboxes"][:, :, 0]) * exemplars_w * pct
				h_noise = th.randn_like(tcond["bboxes"][:, :, 0]) * exemplars_h * pct
				tcond["bboxes"][:, :, 0] = th.clamp(tcond["bboxes"][:, :, 0] + w_noise, min=0, max=w)
				tcond["bboxes"][:, :, 1] = th.clamp(tcond["bboxes"][:, :, 1] + h_noise, min=0, max=h)
				tcond["bboxes"][:, :, 2] = th.clamp(tcond["bboxes"][:, :, 2] + w_noise, min=0, max=w)
				tcond["bboxes"][:, :, 3] = th.clamp(tcond["bboxes"][:, :, 3] + h_noise, min=0, max=h)

			with th.no_grad():
				c, uc = conditioner.get_unconditional_conditioning(
					# tcond if do_tile else cond,
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
					uc.pop("bboxes", None)
					# uc.pop("crossattn", None)
					_cond = uc
				else:
					_cond = c

				with th.autocast(device_type=dev, dtype=th.float16, enabled=args.use_fp16):
					with ema.average_parameters(model.parameters()):
						samples = sample_fn(
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
						)
					
				if use_cfg:
					samples, _ = samples.chunk(2, dim=0)

				if args.save_latents:
					logger.savetensor(
						samples,
						name=f"{count_error}_{cond['id'][0]}", 
						step="latents"
					)

				dm = possibly_vae_decode(samples, vae, clip_decoded=True)
				savable_dm = dm.clone()
				dm = collate_channels(dm, mode="mean")
				dm = dm.squeeze(0)

				count, coords = counting(dm)
				cond = tcond

				ttn_fact = 1
				if args.allow_ttn:
					ttn_fact = ttn(coords, cond["bboxes"].cpu())
					count = count / ttn_fact
				isttn = "ttn" if ttn_fact > 1 else ""

				times.append(time.time() - start_time)
				vram.append(th.cuda.max_memory_allocated() / 1e6)
				logger.log(f"VRAM: {vram[-1]:.2f}MB")
				logger.log(f"Time: {times[-1]:.2f}s")

				# might not work with tiling
				closest_gt = th.cdist(coords.float().cpu(), tcond["points"].float().cpu(), p=2).argmin(dim=-1)
				predpoints[_id] = dict(
					points=coords.tolist(),
					closest_gt=closest_gt.tolist(),
					canvas_size=dm.shape[-1],
				)

				count_error = int(abs(target_count - count))
				res = draw_result(cond["img"][0], dm, float(count), target_count, coords)
				res = draw_bboxes(res, tcond["bboxes"][0])

				logger.logimg(
					res,
					name=f"{count_error}_{_id}_{isttn}", 
					step=f"{split}_imgs"
				)
				if args.save_densities:
					logger.savetensor(
						savable_dm,
						name=f"{count_error}_{_id}_{isttn}", 
						step=f"{split}_dms"
					)

				logger.log(f"{(i := i + bs)}/{N}")
				MAE += abs(target_count - count)
				RMSE += (target_count - count)**2

		MAE = MAE / N
		RMSE = math.sqrt(RMSE / N)
		# times.pop(0)
		max_time = max(times)
		max_vram = max(vram)
		min_time = min(times)
		min_vram = min(vram)
		avg_time = sum(times) / len(times)
		avg_vram = sum(vram) / len(vram)
		logger.log(f"max iteration time: {max_time:.2f}s")
		logger.log(f"min iteration time: {min_time:.2f}s")
		logger.log(f"avg iteration time: {avg_time:.2f}s")
		logger.log(f"max vram: {max_vram:.2f}MB")
		logger.log(f"min vram: {min_vram:.2f}MB")
		logger.log(f"avg vram: {avg_vram:.2f}MB")
		log_final_errors(MAE, RMSE, split)
	log_args(args)

	json_object = json.dumps(predpoints, indent=2)
	with open(osp.join(logger.get_mediadir(), "predpoints.json"), "w+") as outfile:
		outfile.write(json_object)


def log_final_errors(mae, rmse, split):
	with open(osp.join(logger.get_mediadir(), f"results.txt"), "a") as f:
		print(22 * "-", file=f)
		print("  VALIDATION RESULTS" if "val" in split else "     TEST RESULTS", file=f)
		print(22 * "-", file=f)
		print(f"MAE:\t{mae:.4f}", file=f)
		print(f"RMSE:\t{rmse:.4f}", file=f)
		print("\n", file=f, end="")


def log_args(args):
	with open(osp.join(logger.get_mediadir(), f"results.txt"), "a") as f:
		print(50 * "*", file=f)
		print("\n", file=f, end="")
		for k, v in vars(args).items():
			print(f" {k:25}{v}", file=f)

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--expdir", type=str)
	parser.add_argument("--checkpoint", type=str)
	parser.add_argument("--timestep_respacing", type=str, default="")
	parser.add_argument("--use_ddim", action="store_true")
	parser.add_argument("--use_fp16", action="store_true")
	parser.add_argument("--save_densities", action="store_true")
	parser.add_argument("--cfg_scale", type=float, default=0.0)
	parser.add_argument("--zero_shot", action="store_true")
	parser.add_argument("--seed", type=int, default=None)
	parser.add_argument("--save_latents", action="store_true")
	parser.add_argument("--allow_resizing", action="store_true")
	parser.add_argument("--allow_ttn", action="store_true")
	parser.add_argument("--topk", type=int, default=-1)
	parser.add_argument("--trainset", action="store_true")
	parser.add_argument("--n_exemplars", type=int, default=None)
	return parser.parse_args()


if __name__ == "__main__":
	main()