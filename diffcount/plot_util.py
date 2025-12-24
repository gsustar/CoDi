import torch as th
import numpy as np
import io

import matplotlib.pyplot as plt
plt.switch_backend('agg')

from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PIL import Image

import torchvision.transforms.functional as F

def _ready_for_plotting(t):
	return (
		(isinstance(t, th.Tensor)
		 and t.dtype == th.uint8
   		 and t.min() >= 0
   		 and t.max() <= 255)
		or isinstance(t, Image.Image)
		or isinstance(t, np.ndarray)
	)


def _maybe_to_plotting_range(x):
	if _ready_for_plotting(x):
		return x
	assert isinstance(x, th.Tensor)
	x = (x
		.float()
		.add(1)
		.div_(2)
		.mul_(255)
		.add_(0.5)
		.clamp_(0, 255)
		.to("cpu", th.uint8)
		.detach()
		.numpy())
	if x.ndim == 4:
		x = x.transpose(0, 2, 3, 1)
	elif x.ndim == 3:
		x = x.transpose(1, 2, 0)
	else:
		raise ValueError("Invalid number of dimensions for input")
	return x
	

def fig_to_pil(fig):
	buf = io.BytesIO()
	fig.set_frameon(False)
	fig.savefig(buf, bbox_inches="tight", pad_inches=0.0, dpi=fig.dpi)
	buf.seek(0)
	img = Image.open(buf)
	plt.close(fig)
	return img


def to_pil_image(_input):
	if isinstance(_input, (th.Tensor, list)):
		return fig_to_pil(grid(_input))
	assert isinstance(_input, Image.Image)
	return _input
	

def pil_to_tensor(pil):
	return F.pil_to_tensor(pil)


def grid(tensor_or_pils, nrow=None):
	if isinstance(tensor_or_pils, list):
		assert all([isinstance(x, Image.Image) for x in tensor_or_pils])
		ims = tensor_or_pils
		bs = len(tensor_or_pils)
		w, h = tensor_or_pils[0].size
	elif isinstance(tensor_or_pils, Image.Image):
		ims = [tensor_or_pils]
		bs = 1
		w, h = tensor_or_pils.size
	else:
		assert isinstance(tensor_or_pils, th.Tensor) and tensor_or_pils.ndim == 4
		bs, _, h, w = tensor_or_pils.shape
		ims = _maybe_to_plotting_range(tensor_or_pils)

	per_row = int(np.ceil(np.sqrt(bs)))
	nrow = int(np.ceil(bs / per_row)) if nrow is None else nrow
	ncol = int(np.ceil(bs / nrow))
	dpi = 100
	wspace = 0.01
	hspace = 0.01
	fig = plt.figure(
		figsize=(
			ncol * w / dpi + (ncol-1) * wspace * (w / dpi), 
			nrow * h / dpi + (nrow-1) * hspace * (h / dpi)
		), 
		frameon=False, 
		dpi=dpi
	)
	gs = fig.add_gridspec(
		nrows=nrow, 
		ncols=ncol,
		left=0.0, 
		right=1.0, 
		top=1.0, 
		bottom=0.0, 
		wspace=wspace, 
		hspace=hspace
	)
	_ = gs.subplots()

	for i, ax in enumerate(fig.get_axes()):
		ax.set_axis_off()
		if i > bs-1:
			continue
		ax.imshow(ims[i], aspect="equal")

	return fig


def draw_sequence(tensor):
	if isinstance(tensor, list):
		tensor = th.stack(tensor)
	assert tensor.dim() == 4
	fig = grid(tensor, nrow=1)
	return fig_to_pil(fig)


def draw_result(img, dm, pred_count=None, target_count=None, pred_coords=None, marker="P", mc="red", ms=12, dm_alpha=0.3, linewidths=0.5):
	if img.dim() == 4:
		assert img.shape[0] == 1
		img.squeeze(0)
	assert img.dim() == 3
	
	_, h, w = img.shape
	dpi = 100
	fig = plt.figure(figsize=(w/dpi, h/dpi), frameon=False, dpi=dpi)
	ax = plt.Axes(fig, [0., 0., 1., 1.], frameon=False)
	img = _maybe_to_plotting_range(img)
	dm = _maybe_to_plotting_range(dm)
	ax.set_axis_off()
	ax.imshow(img, aspect="equal")
	ax.imshow(dm, aspect="equal", alpha=dm_alpha)
	if pred_coords is not None:
		pts = ax.scatter(
			pred_coords.T[0].cpu(), 
			pred_coords.T[1].cpu(), 
			marker=marker, s=ms, c=mc, 
			edgecolor="black", linewidths=linewidths
		)
	if pred_count:
		ax.text(4.0, 13.0, f"PR: {pred_count:>.1f}", color="white", fontsize=9)
	if target_count:
		ax.text(4.0, 25.0, f"GT: {target_count:>.1f}", color="chartreuse", fontsize=9)
	fig.add_axes(ax)

	return fig_to_pil(fig)


def draw_bboxes(imgs, bboxes, edgecolor="red", linewidth=2):

	if isinstance(imgs, Figure):
		fig = imgs
	else:
		fig = grid(imgs)

	if bboxes is not None:
		
		if bboxes.dim() == 2:
			bboxes = bboxes.unsqueeze(0)
		bboxes = bboxes.cpu().numpy()

		for ax, boxes in zip(fig.get_axes(), bboxes):
			for b in boxes:
				rect = Rectangle(
					(b[0], b[1]), 
					width=b[2]-b[0], 
					height=b[3]-b[1],
					linewidth=linewidth,
					edgecolor=edgecolor,
					facecolor="none"
				)
				ax.add_patch(rect)

	return fig_to_pil(fig)


def draw_text(cs):
	if isinstance(cs, str):
		cs = [cs]
	assert isinstance(cs, list)
	background = th.ones((len(cs), 3, 60, 180), dtype=th.float32)
	fig = grid(background)
	for c, ax in zip(cs, fig.get_axes()):
		ax.text(
			90, 30, 
			str(c), 
			color="black", 
			fontsize=12,
			horizontalalignment="center",
        	verticalalignment="center" 
		)
	return fig_to_pil(fig)