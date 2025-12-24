import torch as th
import torch.nn as nn

import torchvision.transforms.v2 as v2
from torchvision import ops
from torchvision.transforms.v2 import functional as F

from torchvision.tv_tensors import BoundingBoxes
from .points import Points


class RandomMosaic(nn.Module):
	
	def __init__(
		self, 
		p, 
		avgbbox_h_lim=50.0, 
		avgbbox_w_lim=50.0, 
		avgbbox_area_lim=720.0, 
		hflip_p=0.0
	):
		super().__init__()

		self.p = p
		self.avgbbox_h_lim = avgbbox_h_lim
		self.avgbbox_w_lim = avgbbox_w_lim
		self.avgbbox_area_lim = avgbbox_area_lim
		self.hflip = v2.RandomHorizontalFlip(p=hflip_p)


	def make_tile(self, im, bbxs, pts, num_tiles, num_exemplars=3):
		_, h, w = im.shape
		assert (h, w) == pts.canvas_size == bbxs.canvas_size
		result_im = []
		result_pts = []
		result_bbxs = []
		for j in range(num_tiles):
			row_im = []
			for k in range(num_tiles):
				_im = im.clone()
				_pts = pts.clone()
				if k + j > 0:
					_bbxs = BoundingBoxes(
						bbxs[num_exemplars:].clone(), 
						format="XYXY", 
						canvas_size=(h, w)
					)
				else:
					_bbxs = bbxs.clone()

				_im, _pts, _bbxs = self.hflip(_im, _pts, _bbxs)

				padding = [k*w, j*h, 0, 0]
				_pts = F.pad(_pts, padding=padding)
				_bbxs = F.pad(_bbxs, padding=padding)

				row_im.append(_im)
				result_pts.append(_pts)
				result_bbxs.append(_bbxs)
			result_im.append(th.cat(row_im, dim=-1))
	
		result_im = th.cat(result_im, dim=-2)
		result_bbxs = BoundingBoxes(
			th.cat(result_bbxs, dim=0), 
			format="XYXY", 
			canvas_size=(h*num_tiles, w*num_tiles)
		)
		result_pts = Points(
			th.cat(result_pts, dim=0), 
			canvas_size=(h*num_tiles, w*num_tiles)
		)
		return result_im, result_bbxs, result_pts


	def forward(self, img, bboxes, points, num_exemplars=3):
		if th.rand(1) >= self.p:
			return img, bboxes, points
		
		avgbbox_w = (bboxes[:, 2] - bboxes[:, 0]).mean()
		avgbbox_h = (bboxes[:, 3] - bboxes[:, 1]).mean()
		avgbbox_a = ops.box_area(bboxes).mean()
		if avgbbox_w <= self.avgbbox_w_lim or avgbbox_h <= self.avgbbox_h_lim or avgbbox_a <= self.avgbbox_area_lim:
			return img, bboxes, points

		_, h, w = img.shape

		x_tile, y_tile = (th.rand(1) + 1, th.rand(1) + 1)
		num_tiles = max(
			int(x_tile.ceil()), 
			int(y_tile.ceil())
		)

		img, bboxes, points = self.make_tile(img, bboxes, points, num_tiles, num_exemplars)

		img = F.crop(img, top=0, left=0, height=int(y_tile*h), width=int(x_tile*w))
		points = F.crop(points, top=0, left=0, height=int(y_tile*h), width=int(x_tile*w))
		bboxes = F.crop(bboxes, top=0, left=0, height=int(y_tile*h), width=int(x_tile*w))

		img = F.resize(img, (h, w))
		points = F.resize(points, (h, w))
		bboxes = F.resize(bboxes, (h, w))

		return img, bboxes, points


class ZoominCrop(nn.Module):
	
	def __init__(self, leeway):
		super().__init__()
		self.leeway = leeway


	def forward(self, img, bboxes, points):
		_, h, w = img.shape

		(xmin, ymin), _ = th.min(bboxes.T[:2], dim=1)
		(xmax, ymax), _ = th.max(bboxes.T[2:], dim=1)

		xmin = int(max(0, (xmin - self.leeway).item()))
		ymin = int(max(0, (ymin - self.leeway).item()))
		xmax = int(min(w, (xmax + self.leeway).item()))
		ymax = int(min(h, (ymax + self.leeway).item()))

		img = F.crop(img, top=ymin, left=xmin, height=ymax-ymin, width=xmax-xmin)
		points = F.crop(points, top=ymin, left=xmin, height=ymax-ymin, width=xmax-xmin)
		bboxes = F.crop(bboxes, top=ymin, left=xmin, height=ymax-ymin, width=xmax-xmin)

		return img, bboxes, points
	

class MinMaxNormalize(nn.Module):

	def __init__(self):
		super().__init__()

	def forward(self, input):
		_max = input.max()
		_min = input.min()
		return (input - _min) / (_max - _min)
	