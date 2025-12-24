import argparse
import torch as th
import os.path as osp

from PIL import Image
from torchvision import transforms as T

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from torch.utils.data import default_collate

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
	freeze,
)
from diffcount.script_util import (
	create_model,
	create_diffusion,
	create_conditioner,
	create_vae,
	parse_config,
	seed_everything
)
plt.switch_backend("tkagg")

bounding_boxes = []
global clicked

# Global variables to track drawing state
rect = None
start_x, start_y = None, None

# Event handler for mouse press (start drawing)
def on_press(event):
	global start_x, start_y, rect
	if event.inaxes:
		start_x, start_y = event.xdata, event.ydata  # Store starting point
		# Create a rectangle (but do not draw yet)
		rect = patches.Rectangle((start_x, start_y), 0, 0, linewidth=2, edgecolor='r', facecolor='none')
		event.inaxes.add_patch(rect)
		plt.draw()  # Update plot to show rectangle (even if not yet drawn)

# Event handler for mouse motion (while drawing)
def on_motion(event):
	global start_x, start_y, rect
	if rect is not None and event.inaxes:
		# Update the width and height of the rectangle based on mouse position
		width = event.xdata - start_x
		height = event.ydata - start_y
		rect.set_width(width)
		rect.set_height(height)
		plt.draw()  # Redraw to update the rectangle while dragging

# Event handler for mouse release (end drawing)
def on_release(event):
	global rect
	# Once mouse is released, we finalize the bounding box
	if rect is not None:
		bounding_boxes.append([rect.get_x(), rect.get_y(), rect.get_x() + rect.get_width(), rect.get_y() + rect.get_height()])
		rect = None  # Reset rect after release


@th.no_grad()
def demo(args):
	img_path = args.image_path
	global fig, ax

	expdir = args.expdir
	config = parse_config(osp.join(expdir, "config.yaml"))
	dev = "cuda" if th.cuda.is_available() else "cpu"
	ckpt = th.load(osp.join(expdir, args.checkpoint), map_location=dev)

	if args.seed is not None:
		seed_everything(args.seed)

	logger.configure(
		dir=expdir, 
		format_strs=['stdout', 'log'],
		log_suffix=f"_{osp.basename(args.image_path)}"
	)
	logger.set_mediadir(f"{osp.basename(args.image_path)}")

	logger.log("creating model...")
	model = create_model(config.model)
	model.to(dev)
	model.load_state_dict(ckpt["model"], strict=False)
	model.eval()
	model = freeze(model)

	logger.log("creating diffusion...")
	config.diffusion.params.timestep_respacing = args.timestep_respacing
	diffusion = create_diffusion(config.diffusion)

	logger.log("creating VAE...")
	vae = create_vae(
		getattr(config, "vae", None), device=dev
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



	image = T.ToTensor()(Image.open(img_path).convert("RGB"))
	image = T.Resize(size=(512, 512))(image)
	# Create a figure and axis
	fig, ax = plt.subplots(1)
	ax.imshow(image.permute(1,2,0))
	plt.axis('off')
	# Connect the click event
	fig.canvas.mpl_connect('button_press_event', on_press)
	fig.canvas.mpl_connect('motion_notify_event', on_motion)
	fig.canvas.mpl_connect('button_release_event', on_release)
	plt.title("Click and drag to draw bboxes, then close window")
	# Show the image
	plt.show()
	image = (image * 2.0 - 1.0).clamp(-1.0, 1.0)
	bboxes = th.tensor(bounding_boxes, dtype=th.float32)

	

	ch_mult = vae.config.latent_channels if vae else 1
	with_tlrb = config.data.dataset.params.with_tlrb
	ich = 5 * ch_mult if with_tlrb else ch_mult

	use_cfg = args.cfg_scale > 0.0
	use_zero_shot = args.zero_shot
	assert not (use_cfg and use_zero_shot)
	sample_fn = (
		diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
	)
	model_fn = (
		model if not use_cfg else model.forward_with_cfg
	)


	logger.log("sampling...")
	cond = dict(img=image, bboxes=bboxes)
	cond = default_collate([cond])
	cond = torch_to(cond, dev)

	tcond = eval_preprocess(
		cond.copy(), 
		allow_resizing=args.allow_resizing,
		force_resize=args.force_resize,
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
		z = th.randn(bs, ch, h, w, device=dev)

		if use_cfg:
			z = th.cat((z, z), dim=0)
			assert c.keys() == uc.keys(), (
				"conditional and unctonditional conditioning dictionaries must have same keys"
			)
			_cond = {k: th.cat((c[k], uc[k]), dim=0) for k in c}
					
		elif use_zero_shot:
			uc.pop("bboxes", None)
			uc.pop("crossattn", None)
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
				name=f"{osp.basename(args.image_path)}", 
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

		plt.switch_backend("agg")
		zero_s = th.zeros_like(cond["img"][0])
		res = draw_result(
			cond["img"][0], 
			dm=zero_s,
			pred_count=None, 
			target_count=None, 
			pred_coords=coords,
			dm_alpha=0.0,
			marker="o",
			ms=20,
			mc="yellow"
		)
		res = draw_bboxes(res, cond["bboxes"][0])

		logger.logimg(
			res, 
			name=f"{osp.basename(args.image_path)}_{imgh}_{isttn}",
			step="imgs"
		)

		logger.log(f"{osp.basename(args.image_path)}_{imgh}")
		if args.save_densities:
			logger.savetensor(
				savable_dm,
				name=f"{osp.basename(args.image_path)}_{imgh}_{isttn}", 
				step="dms"
			)
		logger.log("\u2713")

	plt.switch_backend("tkagg")
	plt.clf()
	plt.imshow(res)
	plt.title("Object count:" + str(count))
	plt.axis('off')
	plt.show()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--image_path", type=str)
	parser.add_argument("--expdir", type=str)
	parser.add_argument("--checkpoint", type=str)
	parser.add_argument("--timestep_respacing", type=str, default="")
	parser.add_argument("--use_ddim", action="store_true")
	parser.add_argument("--use_fp16", action="store_true")
	parser.add_argument("--cfg_scale", type=float, default=0.0)
	parser.add_argument("--zero_shot", action="store_true")
	parser.add_argument("--seed", type=int, default=None)
	parser.add_argument("--save_densities", action="store_true")
	parser.add_argument("--save_latents", action="store_true")
	parser.add_argument("--allow_resizing", action="store_true")
	parser.add_argument("--allow_ttn", action="store_true")
	parser.add_argument("--force_resize", type=int, default=None)
	args = parser.parse_args()
	demo(args)