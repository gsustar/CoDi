import torch as th
import numpy as np

from scipy.ndimage import gaussian_filter
from torchvision.ops import box_area
import os
EXP_IGNORE_LARGE = int(os.environ.get("EXP_IGNORE_LARGE", "0"))

def compute_padding(image_size, target_size, center=False):
	image_h, image_w = image_size
	resize_h, resize_w = target_size
	pad_h = resize_h - image_h
	pad_w = resize_w - image_w
	pad_t = 0
	pad_l = 0
	if center:
		pad_t = pad_h // 2
		pad_l = pad_w // 2
	pad_b = pad_h - pad_t
	pad_r = pad_w - pad_l
	padding = [pad_l, pad_t, pad_r, pad_b]
	return padding


def unpad(image, padding):
	image_h, image_w = image.shape[-2:]
	pad_l, pad_t, pad_r, pad_b = padding
	return image[
		:, 
		:, 
		pad_t:(image_h - pad_b), 
		pad_l:(image_w - pad_r)
	]


def generate_density_map(size, points, sigma):
	h, w = size
	bitmap = np.zeros((h, w), dtype=np.float32)
	for point in points:
		x, y = int(point[0])-1, int(point[1])-1
		bitmap[y, x] = 1.0

	if sigma > 0.0:
		density_map = gaussian_filter(
			bitmap,
			sigma,
			truncate=3.0,
			mode='constant'
		)
	else:
		density_map = bitmap
	return th.from_numpy(density_map[None, :])

def generate_exponential_decay_map(size, points, rate):
	# Create coordinate grid
	yy, xx = np.indices(size)
	distance_map = np.full(size, np.inf)

	# Compute minimum Euclidean distance to the nearest center
	for point in points:
		x, y = int(point[0])-1, int(point[1])-1
		dist = np.sqrt((yy - y) ** 2 + (xx - x) ** 2)
		distance_map = np.minimum(distance_map, dist)

	# Exponential decay function
	decay_map = np.exp(-rate * distance_map)
	return th.from_numpy(decay_map[None, :]).float()


def generate_density_map_from_bboxes(size, bboxes):
	h, w = size

	_area = box_area(bboxes)
	order = th.argsort(_area, descending=True)
	bboxes = th.round(bboxes[order]).int()

	dm = th.zeros((h, w), dtype=th.float32)
	for bbox in bboxes:
		x_min, y_min, x_max, y_max = bbox

		x_min = int(np.clip(x_min, 0, w-1))
		y_min = int(np.clip(y_min, 0, h-1))
		x_max = int(np.clip(x_max, 0, w-1))
		y_max = int(np.clip(y_max, 0, h-1))

		# Create a meshgrid for this bbox region
		xs = np.arange(x_min, x_max + 1)
		ys = np.arange(y_min, y_max + 1)
		xv, yv = np.meshgrid(xs, ys)

		# Center of the box
		cx = (x_min + x_max) / 2
		cy = (y_min + y_max) / 2

		# Box width and height, avoid division by zero
		bw = max(x_max - x_min, 1)
		bh = max(y_max - y_min, 1)

		# Generate a Gaussian-like response
		sigma_x = bw / 6.0  # covers ~99% within the box
		sigma_y = bh / 6.0

		gaussian = np.exp(-(((xv - cx) ** 2) / (2 * sigma_x ** 2) + ((yv - cy) ** 2) / (2 * sigma_y ** 2)))

		# Place into density map
		current_patch = dm[y_min:y_max+1, x_min:x_max+1]
		updated_patch = np.maximum(current_patch, gaussian)

		# gaussian = th.from_numpy(gaussian)
		# updated_patch = th.from_numpy(updated_patch)

		dm[y_min:y_max+1, x_min:x_max+1] = updated_patch

	return dm[None, :].float()


def generate_fcos_centerness(size, bboxes):
	h, w = size

	_area = box_area(bboxes)
	order = th.argsort(_area, descending=True)
	bboxes = th.round(bboxes[order]).int()

	centerness_map = np.zeros((h, w), dtype=np.float32)
	for bbox in bboxes:
		l, t, r, b = bbox

		l = int(np.clip(l, 0, w - 1))
		r = int(np.clip(r, 0, w - 1))
		t = int(np.clip(t, 0, h - 1))
		b = int(np.clip(b, 0, h - 1))

		xs = np.arange(l, r + 1)
		ys = np.arange(t, b + 1)
		xv, yv = np.meshgrid(xs, ys)

		# Compute l, t, r, b distances from each pixel to box sides
		l_dist = xv - l
		t_dist = yv - t
		r_dist = r - xv
		b_dist = b - yv

		# Stack as per FCOS: [l, t, r, b]
		reg_targets = np.stack([l_dist, t_dist, r_dist, b_dist], axis=-1)

		# Apply FCOS centerness formula
		left_right = reg_targets[..., [0, 2]]
		top_bottom = reg_targets[..., [1, 3]]

		min_lr = np.min(left_right, axis=-1)
		max_lr = np.max(left_right, axis=-1)
		min_tb = np.min(top_bottom, axis=-1)
		max_tb = np.max(top_bottom, axis=-1)

		centerness = np.sqrt(
			(min_lr / (max_lr + 1e-6)) * (min_tb / (max_tb + 1e-6))
		)
		# current_patch = centerness_map[t:b+1, l:r+1]
		# updated_patch = np.maximum(current_patch, centerness)
		# centerness = th.from_numpy(centerness)

		# Overwrite the target region in the map
		centerness_map[t:b+1, l:r+1] = centerness
		# centerness_map[t:b+1, l:r+1] = updated_patch

	centerness_map = th.from_numpy(centerness_map)
	return centerness_map[None, :].float()


def compute_locations(h, w, stride, device):
	shifts_x = th.arange(
		0, w * stride, step=stride,
		dtype=th.float32, device=device
	)
	shifts_y = th.arange(
		0, h * stride, step=stride,
		dtype=th.float32, device=device
	)
	shift_y, shift_x = th.meshgrid(shifts_y, shifts_x, indexing="ij")
	locations = th.stack((shift_x, shift_y), dim=0) + stride // 2
	return locations

def generate_tlrb(locs, bboxes, center_sample=True, radius=5, point_sample=False, points=None, min_radius=1, force_square=False):
	assert not (center_sample and point_sample), "'center_sample' and 'point_sample' are mutually exlusive."
	dev = bboxes.device
	h, w = locs.shape[-2:]

	_area = box_area(bboxes)
	order = th.argsort(_area, descending=True)
	bboxes = th.round(bboxes[order]).int()

	t, l, r, b = th.zeros((4, h, w), device=dev, dtype=th.float32)
	if EXP_IGNORE_LARGE:
		wm = th.ones((h, w), device=dev, dtype=th.float32)
	else:
		wm = th.zeros((h, w), device=dev, dtype=th.float32)

	for bbox in bboxes:
		xmin, ymin, xmax, ymax = bbox
		c_xmin, c_ymin, c_xmax, c_ymax = bbox

		cx = (xmin + xmax) / 2
		cy = (ymin + ymax) / 2

		bbox_w = xmax - xmin
		bbox_h = ymax - ymin
		bbox_a = bbox_w * bbox_h

		if force_square:
			assert radius >= 1, "Forcing square requires radius >= 1"
			radius = max(min(radius, bbox_w // 4, bbox_h // 4), min_radius) 

		stride_x = radius if radius >= 1 else int(radius * bbox_w / 2)
		stride_y = radius if radius >= 1 else int(radius * bbox_h / 2)
		stride_x = max(stride_x, min_radius)
		stride_y = max(stride_y, min_radius)

		if point_sample:
			# assert points is not None and radius > 1
			assert points is not None
			bound_x = th.logical_and(points[:, 0] > xmin, points[:, 0] < xmax)
			bound_y = th.logical_and(points[:, 1] > ymin, points[:, 1] < ymax)
			bbox_filter = th.logical_and(bound_x, bound_y)
			
			candidate_points = points[bbox_filter]
			if len(candidate_points) == 0:
				continue

			point = candidate_points[
				th.argmin(
					th.linalg.norm(
						candidate_points - th.tensor([cx, cy]), dim=1
					)
				)
			]
			px, py = point
			c_xmin = px - stride_x
			c_ymin = py - stride_y
			c_xmax = px + stride_x
			c_ymax = py + stride_y

		if center_sample:
			c_xmin = cx - stride_x
			c_ymin = cy - stride_y
			c_xmax = cx + stride_x
			c_ymax = cy + stride_y

		c_xmin = c_xmin if c_xmin > xmin else xmin
		c_ymin = c_ymin if c_ymin > ymin else ymin
		c_xmax = xmax if c_xmax > xmax else c_xmax
		c_ymax = ymax if c_ymax > ymax else c_ymax

		c_xmin = int(c_xmin)
		c_ymin = int(c_ymin)
		c_xmax = int(c_xmax)
		c_ymax = int(c_ymax)

		t[c_ymin:c_ymax, c_xmin:c_xmax] = locs[1, c_ymin:c_ymax, c_xmin:c_xmax] - ymin
		l[c_ymin:c_ymax, c_xmin:c_xmax] = locs[0, c_ymin:c_ymax, c_xmin:c_xmax] - xmin
		r[c_ymin:c_ymax, c_xmin:c_xmax] = xmax - locs[0, c_ymin:c_ymax, c_xmin:c_xmax]
		b[c_ymin:c_ymax, c_xmin:c_xmax] = ymax - locs[1, c_ymin:c_ymax, c_xmin:c_xmax]
		if EXP_IGNORE_LARGE:
			if (bbox_w > 25 or bbox_h > 25):
				wm[c_ymin:c_ymax, c_xmin:c_xmax] = 0.0
		else:
			wm[c_ymin:c_ymax, c_xmin:c_xmax] = 1.0 / bbox_a

	if not EXP_IGNORE_LARGE:
		n_bg_pxls = th.where(wm == 0, 1.0, 0.0).sum().item()
		if n_bg_pxls == 0.0:
			n_bg_pxls = 1e-6
		wm = th.where(wm == 0, 1.0 / n_bg_pxls, wm)
	wm = wm[None, :, :].expand(4, -1, -1)

	t = t / h
	l = l / w
	r = r / w
	b = b / h

	tlrb = th.stack([t, l, r, b])
	return tlrb, wm


def map_imgname_to_cocoid(labels):
	all_coco_imgs = labels.imgs
	map_name2id = dict()
	for k, v in all_coco_imgs.items():
		img_id = v["id"]
		img_name = v["file_name"]
		map_name2id[img_name] = img_id
	return map_name2id