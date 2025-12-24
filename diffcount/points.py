from __future__ import annotations

import torch

from typing import Any, Mapping, Optional, Sequence, Tuple, Union, List
from torchvision import tv_tensors
from torch.utils._pytree import tree_flatten
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.v2 import functional as F
from torchvision.transforms.v2.functional._utils import is_pure_tensor
from torchvision.transforms.v2.functional._geometry import (
	_parse_pad_padding, 
	_compute_resized_output_size,
	_get_inverse_affine_matrix,
	_compute_affine_output_size,
	_affine_parse_args
)



class Points(tv_tensors.TVTensor):
	""":class:`torch.Tensor` subclass for points with shape ``[N, 2]``.

	Args:
		data: Any data that can be turned into a tensor with :func:`torch.as_tensor`.
		canvas_size (two-tuple of ints): Height and width of the corresponding image or video.
		dtype (torch.dtype, optional): Desired data type of the points. If omitted, will be inferred from
			``data``.
		device (torch.device, optional): Desired device of the points. If omitted and ``data`` is a
			:class:`torch.Tensor`, the device is taken from it. Otherwise, the points is constructed on the CPU.
		requires_grad (bool, optional): Whether autograd should record operations on the points. If omitted and
			``data`` is a :class:`torch.Tensor`, the value is taken from it. Otherwise, defaults to ``False``.
	"""

	canvas_size: Tuple[int, int]

	@classmethod
	def _wrap(cls, tensor: torch.Tensor, *, canvas_size: Tuple[int, int], check_dims: bool = True) -> Points:  # type: ignore[override]
		if check_dims:
			if tensor.ndim == 1:
				tensor = tensor.unsqueeze(0)
			elif tensor.ndim != 2:
				raise ValueError(f"Expected a 1D or 2D tensor, got {tensor.ndim}D")
		points = tensor.as_subclass(cls)
		points.canvas_size = canvas_size
		return points


	def __new__(
		cls,
		data: Any,
		*,
		canvas_size: Tuple[int, int],
		dtype: Optional[torch.dtype] = None,
		device: Optional[Union[torch.device, str, int]] = None,
		requires_grad: Optional[bool] = None,
	) -> Points:
		tensor = cls._to_tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
		return cls._wrap(tensor, canvas_size=canvas_size)


	@classmethod
	def _wrap_output(
		cls,
		output: torch.Tensor,
		args: Sequence[Any] = (),
		kwargs: Optional[Mapping[str, Any]] = None,
	) -> Points:
		flat_params, _ = tree_flatten(args + (tuple(kwargs.values()) if kwargs else ()))  # type: ignore[operator]
		first_bbox_from_args = next(x for x in flat_params if isinstance(x, Points))
		canvas_size = first_bbox_from_args.canvas_size

		if isinstance(output, torch.Tensor) and not isinstance(output, Points):
			output = Points._wrap(output, canvas_size=canvas_size, check_dims=False)
		elif isinstance(output, (tuple, list)):
			output = type(output)(
				Points._wrap(part, canvas_size=canvas_size, check_dims=False) for part in output
			)
		return output


	def __repr__(self, *, tensor_contents: Any = None) -> str:  # type: ignore[override]
		return self._make_repr(canvas_size=self.canvas_size)


def resize_points(
	points: torch.Tensor,
	canvas_size: Tuple[int, int],
	size: Optional[List[int]],
	max_size: Optional[int] = None,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
	old_height, old_width = canvas_size
	new_height, new_width = _compute_resized_output_size(canvas_size, size=size, max_size=max_size)

	if (new_height, new_width) == (old_height, old_width):
		return points, canvas_size

	w_ratio = new_width / old_width
	h_ratio = new_height / old_height
	ratios = torch.tensor([w_ratio, h_ratio], device=points.device)
	return (
		points.mul(ratios).to(points.dtype),
		(new_height, new_width),
	)


@F.register_kernel(functional=F.resize, tv_tensor_cls=Points)
def _resize_points_dispatch(
	inpt: Points, size: Optional[List[int]], max_size: Optional[int] = None, **kwargs: Any
) -> Points:
	output, canvas_size = resize_points(
		inpt.as_subclass(torch.Tensor), inpt.canvas_size, size, max_size=max_size
	)
	return Points._wrap(output, canvas_size=canvas_size)


def horizontal_flip_points(
	points: torch.Tensor, canvas_size: Tuple[int, int]
) -> torch.Tensor:
	shape = points.shape
	points = points.clone().reshape(-1, 2)

	points[:, 0].sub_(canvas_size[1]).neg_()

	return points.reshape(shape)


@F.register_kernel(functional=F.hflip, tv_tensor_cls=Points)
def _horizontal_flip_points_dispatch(inpt: Points) -> Points:
	output = horizontal_flip_points(
		inpt.as_subclass(torch.Tensor), canvas_size=inpt.canvas_size
	)
	return Points._wrap(output, canvas_size=inpt.canvas_size)


def vertical_flip_points(
	points: torch.Tensor, canvas_size: Tuple[int, int]
) -> torch.Tensor:
	shape = points.shape
	points = points.clone().reshape(-1, 2)

	points[:, 1].sub_(canvas_size[0]).neg_()

	return points.reshape(shape)


@F.register_kernel(functional=F.vflip, tv_tensor_cls=Points)
def _vertical_flip_points_dispatch(inpt: Points) -> Points:
	output = vertical_flip_points(
		inpt.as_subclass(torch.Tensor), canvas_size=inpt.canvas_size
	)
	return Points._wrap(output, canvas_size=inpt.canvas_size)


def pad_points(
	points: torch.Tensor,
	canvas_size: Tuple[int, int],
	padding: List[int],
	padding_mode: str = "constant",
) -> Tuple[torch.Tensor, Tuple[int, int]]:
	if padding_mode not in ["constant"]:
		raise ValueError(f"Padding mode '{padding_mode}' is not supported with points")

	left, right, top, bottom = _parse_pad_padding(padding)
	pad = [left, top]

	points = points + torch.tensor(pad, dtype=points.dtype, device=points.device)

	height, width = canvas_size
	height += top + bottom
	width += left + right
	canvas_size = (height, width)

	return points, canvas_size


@F.register_kernel(functional=F.pad, tv_tensor_cls=Points)
def _pad_points_dispatch(
	inpt: Points, padding: List[int], padding_mode: str = "constant", **kwargs
) -> Points:
	output, canvas_size = pad_points(
		inpt.as_subclass(torch.Tensor),
		canvas_size=inpt.canvas_size,
		padding=padding,
		padding_mode=padding_mode,
	)
	return Points._wrap(output, canvas_size=canvas_size)


def crop_points(
	points: torch.Tensor,
	top: int,
	left: int,
	height: int,
	width: int,
) -> Tuple[torch.Tensor, Tuple[int, int]]:

	sub = [left, top]
	points = points - torch.tensor(sub, dtype=points.dtype, device=points.device)
	canvas_size = (height, width)
	points, _ = sanitize_points(points, canvas_size=canvas_size)

	return points, canvas_size


@F.register_kernel(functional=F.crop, tv_tensor_cls=Points)
def _crop_points_dispatch(
	inpt: Points, top: int, left: int, height: int, width: int
) -> Points:
	output, canvas_size = crop_points(
		inpt.as_subclass(torch.Tensor), top=top, left=left, height=height, width=width
	)
	return Points._wrap(output, canvas_size=canvas_size)


def _affine_points_with_expand(
	points: torch.Tensor,
	canvas_size: Tuple[int, int],
	angle: Union[int, float],
	translate: List[float],
	scale: float,
	shear: List[float],
	center: Optional[List[float]] = None,
	expand: bool = False,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
	if points.numel() == 0:
		return points, canvas_size

	original_shape = points.shape
	original_dtype = points.dtype
	points = points.clone() if points.is_floating_point() else points.float()
	dtype = points.dtype
	device = points.device
	points = points.reshape(-1, 2)

	angle, translate, shear, center = _affine_parse_args(
		angle, translate, scale, shear, InterpolationMode.NEAREST, center
	)

	if center is None:
		height, width = canvas_size
		center = [width * 0.5, height * 0.5]

	affine_vector = _get_inverse_affine_matrix(center, angle, translate, scale, shear, inverted=False)
	transposed_affine_matrix = (
		torch.tensor(
			affine_vector,
			dtype=dtype,
			device=device,
		)
		.reshape(2, 3)
		.T
	)
	# 1) Single point structure is similar to
	# [(xmin, ymin, 1), (xmax, ymin, 1), (xmax, ymax, 1), (xmin, ymax, 1)]
	points = torch.cat([points, torch.ones(points.shape[0], 1, device=device, dtype=dtype)], dim=-1)
	# 2) Now let's transform the points using affine matrix
	transformed_points = torch.matmul(points, transposed_affine_matrix)
	# 3) Reshape transformed points to [N boxes, 4 points, x/y coords]
	# and compute bounding box from 4 transformed points:
	transformed_points = transformed_points.reshape(-1, 2)
	# out_points_mins, out_points_maxs = torch.aminmax(transformed_points, dim=1)
	# out_bboxes = torch.cat([out_bbox_mins, out_bbox_maxs], dim=1)

	if expand:
		# Compute minimum point for transformed image frame:
		# Points are Top-Left, Top-Right, Bottom-Left, Bottom-Right points.
		height, width = canvas_size
		points = torch.tensor(
			[
				[0.0, 0.0, 1.0],
				[0.0, float(height), 1.0],
				[float(width), float(height), 1.0],
				[float(width), 0.0, 1.0],
			],
			dtype=dtype,
			device=device,
		)
		new_points = torch.matmul(points, transposed_affine_matrix)
		tr = torch.amin(new_points, dim=0, keepdim=True)
		# Translate points
		transformed_points.sub_(tr)
		# Estimate meta-data for image with inverted=True
		affine_vector = _get_inverse_affine_matrix(center, angle, translate, scale, shear)
		new_width, new_height = _compute_affine_output_size(affine_vector, width, height)
		canvas_size = (new_height, new_width)

	out_points = transformed_points.to(original_dtype)
	return out_points, canvas_size


def affine_points(
	points: torch.Tensor,
	canvas_size: Tuple[int, int],
	angle: Union[int, float],
	translate: List[float],
	scale: float,
	shear: List[float],
	center: Optional[List[float]] = None,
) -> torch.Tensor:
	out_points, _ = _affine_points_with_expand(
		points,
		canvas_size=canvas_size,
		angle=angle,
		translate=translate,
		scale=scale,
		shear=shear,
		center=center,
		expand=False,
	)
	return out_points


@F.register_kernel(functional=F.affine, tv_tensor_cls=Points)
def _affine_points_dispatch(
	inpt: Points,
	angle: Union[int, float],
	translate: List[float],
	scale: float,
	shear: List[float],
	center: Optional[List[float]] = None,
	**kwargs,
) -> Points:
	output = affine_points(
		inpt.as_subclass(torch.Tensor),
		canvas_size=inpt.canvas_size,
		angle=angle,
		translate=translate,
		scale=scale,
		shear=shear,
		center=center,
	)
	return Points._wrap(output, canvas_size=inpt.canvas_size)


def rotate_points(
	points: torch.Tensor,
	canvas_size: Tuple[int, int],
	angle: float,
	expand: bool = False,
	center: Optional[List[float]] = None,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
	return _affine_points_with_expand(
		points,
		canvas_size=canvas_size,
		angle=-angle,
		translate=[0.0, 0.0],
		scale=1.0,
		shear=[0.0, 0.0],
		center=center,
		expand=expand,
	)


@F.register_kernel(functional=F.rotate, tv_tensor_cls=Points)
def _rotate_points_dispatch(
	inpt: Points, angle: float, expand: bool = False, center: Optional[List[float]] = None, **kwargs
) -> Points:
	output, canvas_size = rotate_points(
		inpt.as_subclass(torch.Tensor),
		canvas_size=inpt.canvas_size,
		angle=angle,
		expand=expand,
		center=center,
	)
	return Points._wrap(output, canvas_size=canvas_size)


def sanitize_points(
	points: torch.Tensor,
	canvas_size: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
	
	if torch.jit.is_scripting() or is_pure_tensor(points):
		if canvas_size is None:
			raise ValueError(
				"canvas_size cannot be None if points is a pure tensor. "
				f"Got canvas_size={canvas_size}."
				"Set it to appropriate value or pass points as a Points object."
			)
		valid = _get_sanitize_points_mask(
			points, canvas_size=canvas_size
		)
		points = points[valid]
	else:
		if not isinstance(points, Points):
			raise ValueError("points must be a Points instance or a pure tensor.")
		if canvas_size is not None:
			raise ValueError(
				"canvas_size must be None when points is a Points instance. "
				f"Got canvas_size={canvas_size}. "
				"Leave it to None or pass points as a pure tensor."
			)
		valid = _get_sanitize_points_mask(
			points,
			canvas_size=points.canvas_size,
		)
		points = Points._wrap(points[valid], canvas_size=points.canvas_size)

	return points, valid


def _get_sanitize_points_mask(
	points: torch.Tensor,
	canvas_size: Tuple[int, int],
) -> torch.Tensor:

	image_h, image_w = canvas_size
	valid = (points >= 0).all(dim=-1)
	valid &= (points[:, 0] <= image_w)
	valid &= (points[:, 1] <= image_h)
	return valid