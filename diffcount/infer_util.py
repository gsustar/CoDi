import torch as th

from torchvision.transforms.v2 import functional as F

from torchvision import ops
from torchvision.tv_tensors import BoundingBoxFormat
from skimage.feature import peak_local_max

from .points import Points, resize_points, pad_points


def counting(d):
	assert d.dim() == 3
	c, h, w = d.shape
	assert c == 1

	d = d.clamp_(-1.0, 1.0)
	d = (d + 1.0) / 2.0
	d = d.detach().cpu().numpy().squeeze()

	threshold = d.mean() + 0.1
	d[d < threshold] = 0.0

	coords = peak_local_max(d, exclude_border=0)
	coords = Points(coords[:, [1, 0]], canvas_size=(h, w))
	cnt = len(coords)

	return cnt, coords


def collate_channels(inpt, mode="mean"):
	assert inpt.dim() in [3, 4]
	reduce_dim = 0
	if inpt.dim() == 4:
		reduce_dim = 1

	if mode == "mean":
		inpt = inpt.mean(dim=reduce_dim)
	elif mode == "max":
		inpt, _ = inpt.max(dim=reduce_dim)
	elif mode == "first":
		if reduce_dim == 1:
			inpt = inpt[:, 0]
		else:
			inpt = inpt[0]
	else:
		raise ValueError(f"Unsupported collate mode '{mode}'")

	if reduce_dim == 1:
		inpt = inpt.unsqueeze(reduce_dim)

	return inpt


def eval_preprocess(
	cond, 
	allow_resizing=False,
	force_resize=None,
):
	h, w = cond["img"].shape[-2:]
	assert h == w, "Just to make things easier"

	BBOX_AREA_LOWER_LIMIT = 1250

	do_resize = False
	avg_bbox_a = ops.box_area(cond["bboxes"].squeeze(0)).mean().item()
	if avg_bbox_a <= BBOX_AREA_LOWER_LIMIT:
		new_size = (1024, 1024)
		do_resize = True

	if force_resize is not None:
		assert isinstance(force_resize, int)
		do_resize = True
		new_size = (force_resize, force_resize)

	if allow_resizing and do_resize:

		if force_resize == -1:
			bboxes = cond["bboxes"]
			avgbbox_r = ((bboxes[..., 3] - bboxes[..., 1]) / (bboxes[..., 2] - bboxes[..., 0])).mean().item()
			new_size = (h, int(w * avgbbox_r))

		cond["img"] = F.resize(cond["img"], new_size)
		cond["bboxes"], _ = F.resize_bounding_boxes(cond["bboxes"].as_subclass(th.Tensor), canvas_size=(h, w), size=new_size)
		if "points" in cond:
			cond["points"], _ = resize_points(cond["points"].as_subclass(th.Tensor), canvas_size=(h, w), size=new_size)

		p = abs(new_size[0] - new_size[1])
		p, r = p // 2, p % 2

		padding = [p, 0, p+r, 0] # l t r b
		cond["img"] = F.pad(cond["img"], padding=padding, fill=-1.0)
		cond["bboxes"], _ = F.pad_bounding_boxes(cond["bboxes"], format=BoundingBoxFormat.XYXY, canvas_size=new_size, padding=padding)
		if "points" in cond:
			cond["points"], _ = pad_points(cond["points"], canvas_size=new_size, padding=padding)
	return cond


def ttn(coords, bboxes):
	if coords.nelement() == 0:
		return 1

	if bboxes.ndim == 3:
		bboxes = bboxes.squeeze(0)
	
	bbox_counts = []
	for bbox in bboxes:
		bbox_count = (
			(coords[:, 0] > bbox[0]) &
			(coords[:, 0] < bbox[2]) &
			(coords[:, 1] > bbox[1]) &
			(coords[:, 1] < bbox[3])
		).sum().item()
		bbox_counts.append(bbox_count)

	if len(set(bbox_counts)) == 1 and bbox_counts[0] > 0:
		return bbox_counts[0]
	return 1