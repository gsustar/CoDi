import os
import csv
import json
import torch as th
import os.path as osp

from torchvision.transforms import ColorJitter

import torchvision.datasets as thdata
import torchvision.transforms.v2 as v2
import torchvision.transforms.functional as F

from PIL import Image
from pycocotools.coco import COCO
from torchvision.tv_tensors import BoundingBoxes
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import box_convert
from torch.utils.data import default_collate

from .points import Points
from .transforms import RandomMosaic, MinMaxNormalize
from .data_util import generate_density_map, map_imgname_to_cocoid, generate_tlrb, compute_locations, generate_exponential_decay_map, generate_density_map_from_bboxes, generate_fcos_centerness

class MNIST(Dataset):

	def __init__(
		self, 
		datadir, 
		split='train'
	):
		self.dataset = thdata.MNIST(
			root=datadir,
			train=True if split == 'train' else False,
			download=True,
			transform=v2.Compose([
				v2.ToTensor(), v2.Lambda(lambda x: x * 2.0 - 1.0)
			])
		)
	
	def __len__(self):
		return len(self.dataset)
	
	def __getitem__(self, index):
		img, text = self.dataset[index]
		text = th.tensor(text)
		return img, dict(text=text, count=0)


class FSC147(Dataset):

	def __init__(
			self,
			datadir,
			split="train",
			n_exemplars=3,
			image_size=512,
			hflip_p=0.0,
			vflip_p=0.0,
			cj_p=0.0,
			mosaic_p=0.0,
			mosaic_avgbbox_w_lim=50,
			mosaic_avgbbox_h_lim=50,
			mosaic_avgbbox_area_lim=720,
			sigma=0.5,
			target_minmax_norm=False,
			with_tlrb=True,
			tlrb_center_sample=False,
			tlrb_point_sample=False,
			tlrb_radius=8,
			predpoints_path=None,
			exp_decay_rate=None,
			use_bbox_center_as_point=False,
			force_square_tlrb=False
	):
		assert split in ["train", "val", "test", "sample"]

		self.datadir = datadir
		self.split = split
		self.n_exemplars = n_exemplars
		self.image_size = image_size
		self.hflip_p = hflip_p
		self.vflip_p = vflip_p
		self.cj_p = cj_p
		self.mosaic_p = mosaic_p
		self.sigma = sigma
		self.target_minmax_norm = target_minmax_norm
		self.with_tlrb = with_tlrb
		self.tlrb_center_sample = tlrb_center_sample
		self.tlrb_point_sample = tlrb_point_sample
		self.tlrb_radius = tlrb_radius
		self.predpoints_path = predpoints_path
		self.exp_decay_rate = exp_decay_rate
		self.use_bbox_center_as_point = use_bbox_center_as_point
		self.force_square_tlrb = force_square_tlrb

		self.resize = v2.Resize(size=(image_size, image_size))
		self.hflip = v2.RandomHorizontalFlip(p=hflip_p)
		self.vflip = v2.RandomVerticalFlip(p=vflip_p)
		self.cj = ColorJitter(0.2, 0.2, 0.2, 0.1)
		self.mosaic = RandomMosaic(
			p=mosaic_p, 
			avgbbox_h_lim=mosaic_avgbbox_h_lim, 
			avgbbox_w_lim=mosaic_avgbbox_w_lim, 
			avgbbox_area_lim=mosaic_avgbbox_area_lim, 
			hflip_p=0.5)
		self.sanitize_bboxes = v2.SanitizeBoundingBoxes(labels_getter=None)
		self.to_tensor = v2.Compose([
			v2.ToImage(),
			v2.ToDtype(th.float32, scale=True)
		])
		self.minmaxnorm = MinMaxNormalize()

		_splits = ["val", "test"] if self.split == "sample" else [self.split]
		self.img_names = []
		with open(osp.join(self.datadir, "Train_Test_Val_FSC_147.json"), "rb") as f:
			jsonfile = json.load(f)
			for s in _splits:
				self.img_names += jsonfile[s]

		self.labels = {
			"train": None,
			"val": None,
			"test": None
		}
		self.name2id = {
			"train": None,
			"val": None,
			"test": None
		}
		for s in _splits:
			self.labels[s] = COCO(osp.join(self.datadir, "FSC147bbox_annotation", f"instances_{s}.json"))
			self.name2id[s] = map_imgname_to_cocoid(self.labels[s])

		self.annotations = None
		with open(osp.join(self.datadir, "annotation_FSC147_384.json"), "rb") as f:
			self.annotations = {k: v for k, v in json.load(f).items() if k in self.img_names}

		self.img_classes = None
		with open(osp.join(self.datadir, "ImageClasses_FSC147.txt"), "r") as f:
			self.img_classes = {k: v for (k, v) in csv.reader(f, delimiter="\t")}

		self.predpoints_ann = None
		if self.split == "train" and self.predpoints_path is not None:
			with open(self.predpoints_path, "r") as f:
				self.predpoints_ann = {f"{k}.jpg": v for k, v in json.load(f).items() if f"{k}.jpg" in self.img_names}

		self.locs = compute_locations(
			h=self.image_size,
			w=self.image_size,
			stride=1,
			device="cpu"
		)


	def get_gt_bboxes(self, idx):
		for k, v in self.labels.items():
			if v is not None:
				try:
					coco_im_id = self.name2id[k][self.img_names[idx]]
					anno_ids = self.labels[k].getAnnIds([coco_im_id])
					annos = self.labels[k].loadAnns(anno_ids)
				except KeyError:
					continue
		xywh_bboxes = th.tensor([anno["bbox"] for anno in annos])
		xyxy_bboxes = box_convert(xywh_bboxes, "xywh", "xyxy")
		return xyxy_bboxes


	def _shared_transform(self, img, bboxes, points):
		# ToImage
		img = self.to_tensor(img)
		# Resize
		img, bboxes, points = self.resize(img, bboxes, points)
		return img, bboxes, points


	def _train_transform(self, img, bboxes, points):
		img, bboxes, points = self._shared_transform(img, bboxes, points)

		# RandomMosaicing
		img, bboxes, points = self.mosaic(img, bboxes, points, self.n_exemplars)
		# RandomHorizontalFlip
		img, bboxes, points = self.hflip(img, bboxes, points)
		# RandomVerticalFlip
		img, bboxes, points = self.vflip(img, bboxes, points)
		# RandomColorJitter
		if th.rand(1) < self.cj_p:
			img = self.cj(img)

		return img, bboxes, points


	def _val_transform(self, img, bboxes, points):
		img, bboxes, points = self._shared_transform(img, bboxes, points)
		return img, bboxes, points


	def _test_transform(self, img, bboxes, points):
		img, bboxes, points = self._shared_transform(img, bboxes, points)
		return img, bboxes, points
	

	def _sample_transform(self, img, bboxes, points):
		img, bboxes, points = self._shared_transform(img, bboxes, points)
		# img, bboxes, points = v2.Resize(size=(256, 256))(img, bboxes, points)
		# img, bboxes, points = v2.Pad(padding=[128, 128, 128, 128])(img, bboxes, points)
		return img, bboxes, points


	def transform(self, img, bboxes, points):
		if self.split == "train":
			img, bboxes, points = self._train_transform(img, bboxes, points)
		elif self.split == "val":
			img, bboxes, points = self._val_transform(img, bboxes, points)
		elif self.split == "test":
			img, bboxes, points = self._test_transform(img, bboxes, points)
		elif self.split == "sample":
			img, bboxes, points = self._sample_transform(img, bboxes, points)
		
		bboxes = self.sanitize_bboxes(bboxes)
		img = (img * 2.0 - 1.0).clamp(-1.0, 1.0)

		return img, bboxes, points
	

	def target_transform(self, target, tlrb):
		# MinMax Normalization
		if self.target_minmax_norm:
			target = self.minmaxnorm(target)

		target = (target * 2.0 - 1.0).clamp(-1.0, 1.0)
		tlrb = (tlrb * 2.0 - 1.0).clamp(-1.0, 1.0)
		return target, tlrb


	def get_by_name(self, name):
		return self.__getitem__(self.img_names.index(name))


	def __len__(self):
		return len(self.img_names)


	def __getitem__(self, index):
		if th.is_tensor(index):
			index = index.tolist()

		img = Image.open(
			osp.join(
				self.datadir,
				'images_384_VarV2',
				self.img_names[index]
			)
		).convert('RGB')
		w, h = img.size

		bboxes = th.as_tensor(
			self.annotations[self.img_names[index]]['box_examples_coordinates'], 
			dtype=th.float32
		)
		assert len(bboxes) >= self.n_exemplars, f'Not enough examplars for image {self.img_names[index]}'

		bboxes = bboxes[:, [0, 2], :].reshape(-1, 4)
		# bboxes = bboxes[th.randperm(bboxes.shape[0])]
		bboxes = bboxes[:self.n_exemplars, ...]	# (x_min, y_min, x_max, y_max)
		bboxes = BoundingBoxes(
			th.cat([
				bboxes, 
				self.get_gt_bboxes(index)
			]), 
			format="XYXY", 
			canvas_size=(h, w)
		)
		
		points = th.as_tensor(self.annotations[self.img_names[index]]['points'])
		points = Points(points, canvas_size=(h, w))

		predpoints = th.as_tensor([[]])
		if self.predpoints_ann is not None:
			assert self.mosaic_p == 0.0
			assert not self.use_bbox_center_as_point
			predpoints_info = self.predpoints_ann[self.img_names[index]]
			predpoints = th.as_tensor(predpoints_info["points"])
			cs = predpoints_info["canvas_size"]
			predpoints = Points(predpoints, canvas_size=(cs, cs))
			predpoints = v2.functional.resize(predpoints, size=(h, w))
			npredp = len(predpoints)
			points = Points(
				th.cat([predpoints, points], dim=0),
				canvas_size=(h, w)
			)

		text = self.img_classes[self.img_names[index]]
		img_id = osp.splitext(self.img_names[index])[0]
		# target_count = th.tensor(len(points), dtype=th.float32)

		img, bboxes, points = self.transform(img, bboxes, points)

		if self.predpoints_ann is not None:
			predpoints, points = (
				points[:npredp],
				points[npredp:]
			)

		if self.use_bbox_center_as_point and self.split == "train":
			_bboxes = bboxes[self.n_exemplars:]
			points = th.stack([(_bboxes[:, 0] + _bboxes[:, 2]) / 2, (_bboxes[:, 1] + _bboxes[:, 3]) / 2], dim=1)
			points = Points(points, canvas_size=(self.image_size, self.image_size))
	
		bboxes, gt_bboxes = (
			bboxes[:self.n_exemplars], 
			bboxes[self.n_exemplars:]
		)

		if self.exp_decay_rate is not None:
			target = generate_exponential_decay_map(
				size=(self.image_size, self.image_size),
				points=points,
				rate=self.exp_decay_rate
			)
		else:
			if self.sigma == "fcos":
				target = generate_fcos_centerness(
					size=(self.image_size, self.image_size),
					bboxes=gt_bboxes
				)
			elif self.sigma == "bbox":
				target = generate_density_map_from_bboxes(
					size=(self.image_size, self.image_size),
					bboxes=gt_bboxes
				)
			else:
				target = generate_density_map(
					size=(self.image_size, self.image_size),
					points=points,
					sigma=self.sigma
				)
		tlrb, tlrb_wm = generate_tlrb(
			locs=self.locs,
			bboxes=gt_bboxes,
			center_sample=self.tlrb_center_sample,
			radius=self.tlrb_radius,
			point_sample=self.tlrb_point_sample,
			points=points,
			force_square=self.force_square_tlrb,
		)
		tlrb_mask = (tlrb > 0.0).float()[0].unsqueeze(0)
		target, tlrb = self.target_transform(
			target, tlrb
		)

		if self.with_tlrb:
			target = th.cat((target, tlrb), dim=0)

		assert target.shape[-2:] == img.shape[-2:], "target shape does not match image shape."
		target_count = th.tensor(len(points), dtype=th.float32)

		# from .plot_util import draw_bboxes, draw_result
		# import matplotlib.pyplot as plt
		# res = draw_result(img, th.zeros_like(img), target_count=target_count, pred_coords=points)
		# res = draw_bboxes(res, gt_bboxes)
		# plt.imshow(res)
		# plt.savefig("gtbboxes.png")

		# res1 = draw_result(img, th.zeros_like(img), target_count=target_count, pred_coords=points)
		# res1 = draw_bboxes(res1, bboxes)
		# plt.imshow(res1)
		# plt.savefig("exemplars.png")
		
		return target, dict(bboxes=bboxes,
					  		img=img,
							tlrb_mask=tlrb_mask,
							tlrb_wm=tlrb_wm,
					  		count=target_count,
					    	id=img_id,
							text=text,
							points=points.float(),
							predpoints=predpoints,
							gt_bboxes=gt_bboxes,
							og_size=(h, w),
							)


class MCAC(Dataset):

	def __init__(
		self,
		datadir,
		split="train",
		M1=False,
		occ_limit=70,
		n_exemplars=3,
		image_size=512,
		hflip_p=0.0,
		vflip_p=0.0,
		mosaic_p=0.0,
		mosaic_avgbbox_w_lim=0,
		mosaic_avgbbox_h_lim=0,
		mosaic_avgbbox_area_lim=0,
		sigma=0.5,
		target_minmax_norm=False,
		with_tlrb=True,
		tlrb_center_sample=False,
		tlrb_point_sample=False,
		tlrb_radius=8,
		force_square_tlrb=False
	):
		self.datadir = datadir
		# if split == "sample":
		# 	logger.log("'sample' set is not supported yet, switching to test set...")
		# 	split = "test"
		self.split = split
		self.splitdir = osp.join(self.datadir, self.split)
		self.image_size = image_size
		self.n_exemplars = n_exemplars
		self.hflip_p = hflip_p
		self.vflip_p = vflip_p
		self.mosaic_p = mosaic_p
		self.sigma = sigma
		self.target_minmax_norm = target_minmax_norm
		self.with_tlrb = with_tlrb
		self.tlrb_center_sample = tlrb_center_sample
		self.tlrb_point_sample = tlrb_point_sample
		self.tlrb_radius = tlrb_radius
		self.force_square_tlrb = force_square_tlrb

		self.occ_limit = occ_limit
		self.crop_size = 672
		self.MCAC_exclude_imgs_with_num_classes_over = -1
		if M1:
			self.MCAC_exclude_imgs_with_num_classes_over = 1

		self.prepare_image_names()

		self.crop = v2.CenterCrop(size=672)
		self.resize = v2.Resize(size=(image_size, image_size))
		self.hflip = v2.RandomHorizontalFlip(p=hflip_p)
		self.vflip = v2.RandomVerticalFlip(p=vflip_p)
		self.mosaic = RandomMosaic(
			p=mosaic_p, 
			avgbbox_h_lim=mosaic_avgbbox_h_lim, 
			avgbbox_w_lim=mosaic_avgbbox_w_lim, 
			avgbbox_area_lim=mosaic_avgbbox_area_lim, 
			hflip_p=0.5)
		self.sanitize_bboxes = v2.SanitizeBoundingBoxes(labels_getter=None)
		self.to_tensor = v2.Compose([
			v2.ToImage(),
			v2.ToDtype(th.float32, scale=True)
		])
		self.minmaxnorm = MinMaxNormalize()

		self.locs = compute_locations(
			h=self.image_size,
			w=self.image_size,
			stride=1,
			device="cpu"
		)


	def prepare_image_names(self):

		def get_countable_classes_for_image(info):
			countable_classes = []
			for i, c in enumerate(info["countables"]):
				occs = th.tensor(c["occlusions_crop672"])
				inds = occs < self.occ_limit
				cnt = len(occs[inds])
				if cnt > 0:
					countable_classes.append(i)
			return countable_classes

		self.img_names = []
		for f in os.listdir(self.splitdir):
			if os.path.isdir(self.splitdir + "/" + f):
				with open(osp.join(self.splitdir, f, "info_with_occ_bbox.json"), "r") as infofile:
					info = json.load(infofile)
					countable_classes = get_countable_classes_for_image(info)
					num_classes = len(countable_classes)
					if (self.MCAC_exclude_imgs_with_num_classes_over != -1
						and num_classes > self.MCAC_exclude_imgs_with_num_classes_over
					):
						continue
					self.img_names.extend([f"{f}_{i}" for i in countable_classes])


	def get_by_name(self, name):
		assert len(name.split("_")) == 2, "'imgid_classid' string required"
		return self.__getitem__(self.img_names.index(name))
	

	def get_exemplar_bboxes(self, all_bboxes, occlusions):
		sort_ixs = th.argsort(occlusions, stable=True)
		if self.split == "train":
			occ_mask = occlusions < 30
			if th.sum(occ_mask) >= self.n_exemplars:
				valid_bboxes = all_bboxes[occ_mask]
				valid_bboxes = valid_bboxes[th.randperm(len(valid_bboxes))]
				exemplar_bboxes = valid_bboxes[:self.n_exemplars]
				return exemplar_bboxes

		# pick 3 least occluded bboxes (lowest index to break ties)
		exemplar_bboxes = all_bboxes[sort_ixs][:self.n_exemplars]
		return exemplar_bboxes


	def _shared_transform(self, img, bboxes, points):
		# ToImage
		img = self.to_tensor(img)
		# CenterCrop
		img = self.crop(img)
		# img, bboxes, points = self.crop(img, bboxes, points)
		# Resize
		img, bboxes, points = self.resize(img, bboxes, points)
		return img, bboxes, points


	def _train_transform(self, img, bboxes, points):
		img, bboxes, points = self._shared_transform(img, bboxes, points)

		# RandomMosaicing
		img, bboxes, points = self.mosaic(img, bboxes, points)
		# RandomHorizontalFlip
		img, bboxes, points = self.hflip(img, bboxes, points)
		# RandomVerticalFlip
		img, bboxes, points = self.vflip(img, bboxes, points)

		return img, bboxes, points


	def _val_transform(self, img, bboxes, points):
		img, bboxes, points = self._shared_transform(img, bboxes, points)
		return img, bboxes, points


	def _test_transform(self, img, bboxes, points):
		img, bboxes, points = self._shared_transform(img, bboxes, points)
		return img, bboxes, points
	

	def _sample_transform(self, img, bboxes, points):
		img, bboxes, points = self._shared_transform(img, bboxes, points)
		return img, bboxes, points


	def transform(self, img, bboxes, points):
		if self.split == "train":
			img, bboxes, points = self._train_transform(img, bboxes, points)
		elif self.split == "val":
			img, bboxes, points = self._val_transform(img, bboxes, points)
		elif self.split == "test":
			img, bboxes, points = self._test_transform(img, bboxes, points)
		elif self.split == "sample":
			img, bboxes, points = self._sample_transform(img, bboxes, points)
		
		bboxes = self.sanitize_bboxes(bboxes)
		img = (img * 2.0 - 1.0).clamp(-1.0, 1.0)

		return img, bboxes, points
	

	def target_transform(self, target, tlrb):
		# MinMax Normalization
		if self.target_minmax_norm:
			target = self.minmaxnorm(target)

		target = (target * 2.0 - 1.0).clamp(-1.0, 1.0)
		tlrb = (tlrb * 2.0 - 1.0).clamp(-1.0, 1.0)
		return target, tlrb


	def __len__(self):
		return len(self.img_names)


	def __getitem__(self, index):
		if th.is_tensor(index):
			index = index.tolist()

		img_id, cls_ix = self.img_names[index].split("_")
		cls_ix = int(cls_ix)
		img = Image.open(
			osp.join(
				self.splitdir,
				img_id,
				"img.png"
			)
		).convert('RGB')

		infofile = open(
			osp.join(
				self.splitdir,
				img_id,
				"info_with_occ_bbox.json"
			)
		)
		with open(osp.join(self.splitdir, img_id, "info_with_occ_bbox.json"), "r") as infofile:
			info = json.load(infofile)
			countables = info["countables"][cls_ix]

			occlusions = th.as_tensor(
				countables["occlusions_crop672"],
				dtype=th.float32
			)
			occ_mask = occlusions < self.occ_limit
			occlusions = occlusions[occ_mask]

			all_bboxes = th.as_tensor(
				countables["bboxes_crop672"],
				dtype=th.float32
			)
			all_bboxes = th.stack([
				all_bboxes[:, 1, 0],
				all_bboxes[:, 0, 0],
				all_bboxes[:, 1, 1],
				all_bboxes[:, 0, 1],
			]).T
			all_bboxes = all_bboxes[occ_mask]

			bboxes = self.get_exemplar_bboxes(all_bboxes, occlusions)
			num_exemplar_bboxes = len(bboxes)

			all_bboxes = BoundingBoxes(
				th.cat([
					bboxes,
					all_bboxes
				]),
				format="XYXY", 
				canvas_size=(self.crop_size, self.crop_size)
			)
			points = th.as_tensor(
				countables["centers_crop672"],
				dtype=th.float32
			)[:, :2]
			points[:, 0] = points[:, 0] * self.crop_size
			points[:, 1] = (self.crop_size - 1) - points[:, 1] * self.crop_size
			points = points.int()
			points = th.clip(
				points, 0, self.crop_size - 1
			)
			points = points[occ_mask]
			points = Points(points, canvas_size=(self.crop_size, self.crop_size))

			img, all_bboxes, points = self.transform(img, all_bboxes, points)

			bboxes, all_bboxes = (
				all_bboxes[:num_exemplar_bboxes], 
				all_bboxes[num_exemplar_bboxes:]
			)
			bboxes = th.cat([
				bboxes,
				th.zeros(self.n_exemplars - num_exemplar_bboxes, 4)
			])

			text = countables["obj_class"]
			target_count = th.tensor(len(points), dtype=th.float32)


		target = generate_density_map(
			size=(self.image_size, self.image_size),
			points=points,
			sigma=self.sigma
		)
		tlrb, tlrb_wm = generate_tlrb(
			locs=self.locs,
			bboxes=all_bboxes,
			center_sample=self.tlrb_center_sample,
			radius=self.tlrb_radius,
			point_sample=self.tlrb_point_sample,
			points=points,
			force_square=self.force_square_tlrb,
		)
		tlrb_mask = (tlrb > 0.0).float()[0].unsqueeze(0)
		# print(tlrb)
		target, tlrb = self.target_transform(
			target, tlrb
		)

		if self.with_tlrb:
			target = th.cat((target, tlrb), dim=0)

		# import matplotlib.pyplot as plt
		# from diffcount.plot_util import draw_bboxes
		# print(img.shape)
		# img = draw_bboxes(img.unsqueeze(0), bboxes)

		# fig, axs = plt.subplots(1, 3)
		# axs[0].imshow(img)
		# axs[1].imshow(target.squeeze(0))
		# axs[2].imshow(tlrb[0].squeeze(0))
		# axs[0].scatter(
		# 	points[:, 0], 
		# 	points[:, 1], 
		# 	marker="P", s=12, c="red", 
		# 	edgecolor="black", linewidths=0.5
		# )
		# plt.show()
		assert target.shape[-2:] == img.shape[-2:], "target shape does not match image shape."
		return target, dict(bboxes=bboxes,
					  		img=img,
					  		count=target_count,
					    	id=f"{img_id}_{cls_ix}",
							text=text,
							points=points.float(),
							tlrb_mask=tlrb_mask,
							tlrb_wm=tlrb_wm,
							gt_bboxes=all_bboxes,
							predpoints=th.tensor([[]], dtype=th.float32),
							og_size=(self.crop_size, self.crop_size),
							)


class FSCD_LVIS(Dataset):

	def __init__(
		self,
		datadir,
		split="train",
		unseen=False,
		n_exemplars=3,
		image_size=512,
		hflip_p=0.0,
		vflip_p=0.0,
		mosaic_p=0.0,
		mosaic_avgbbox_w_lim=0,
		mosaic_avgbbox_h_lim=0,
		mosaic_avgbbox_area_lim=0,
		sigma=0.5,
		target_minmax_norm=False,
		with_tlrb=False,
		tlrb_center_sample=False,
		tlrb_point_sample=False,
		tlrb_radius=8,
		force_square_tlrb=False,
	):
		self.datadir = datadir
		self.split = split
		self.n_exemplars = n_exemplars
		self.image_size = image_size
		self.hflip_p = hflip_p
		self.vflip_p = vflip_p
		self.mosaic_p = mosaic_p
		self.sigma = sigma
		self.target_minmax_norm = target_minmax_norm
		# assert not with_tlrb, "with_tlrb is not supported yet for FSCD_LVIS dataset"
		self.with_tlrb = with_tlrb
		self.tlrb_center_sample = tlrb_center_sample
		self.tlrb_point_sample = tlrb_point_sample
		self.tlrb_radius = tlrb_radius
		self.force_square_tlrb = force_square_tlrb

		self.resize = v2.Resize(size=(image_size, image_size))
		self.hflip = v2.RandomHorizontalFlip(p=hflip_p)
		self.vflip = v2.RandomVerticalFlip(p=vflip_p)
		self.mosaic = RandomMosaic(
			p=mosaic_p, 
			avgbbox_h_lim=mosaic_avgbbox_h_lim, 
			avgbbox_w_lim=mosaic_avgbbox_w_lim, 
			avgbbox_area_lim=mosaic_avgbbox_area_lim, 
			hflip_p=0.5)
		self.sanitize_bboxes = v2.SanitizeBoundingBoxes(labels_getter=None)
		self.to_tensor = v2.Compose([
			v2.ToImage(),
			v2.ToDtype(th.float32, scale=True)
		])
		self.minmaxnorm = MinMaxNormalize()

		self.locs = compute_locations(
			h=self.image_size,
			w=self.image_size,
			stride=1,
			device="cpu"
		)

		self.imgdir = osp.join(self.datadir, "images")
		unseen_prefix = "unseen_" if unseen else ""

		self.imgdir = osp.join(self.datadir, "images")

		self.img_names = os.listdir(self.imgdir)
		_splits = ["test", "val"] if split == "sample" else [split]
		_img_names = []
		_annotations = []
		for s in _splits:
			if unseen and s == "val":
				continue
			self.annfile = osp.join(self.datadir, "annotations", f"{unseen_prefix}count_{s}.json")
			with open(self.annfile, "rb") as f:
				jsonfile = json.load(f)
				split_anns = [ann for ann in jsonfile["annotations"] if ann["file_name"] in self.img_names]
				split_img_names = [ann["file_name"] for ann in split_anns]
				_annotations.extend(split_anns)
				_img_names.extend(split_img_names)
		self.img_names = _img_names
		self.annotations = _annotations

		self.labels = {
			"train": None,
			"val": None,
			"test": None
		}
		self.name2id = {
			"train": None,
			"val": None,
			"test": None
		}
		for s in _splits:
			self.labels[s] = COCO(osp.join(self.datadir, "annotations", f"instances_{s}.json"))
			self.name2id[s] = map_imgname_to_cocoid(self.labels[s])

	def get_gt_bboxes(self, idx):
		for k, v in self.labels.items():
			if v is not None:
				try:
					coco_im_id = self.name2id[k][self.img_names[idx]]
					anno_ids = self.labels[k].getAnnIds([coco_im_id])
					annos = self.labels[k].loadAnns(anno_ids)
				except KeyError:
					continue
		xywh_bboxes = th.tensor([anno["bbox"] for anno in annos])
		xyxy_bboxes = box_convert(xywh_bboxes, "xywh", "xyxy")
		return xyxy_bboxes

	def _shared_transform(self, img, bboxes, points):
		# ToImage
		img = self.to_tensor(img)
		# Resize
		img, bboxes, points = self.resize(img, bboxes, points)
		return img, bboxes, points


	def _train_transform(self, img, bboxes, points):
		img, bboxes, points = self._shared_transform(img, bboxes, points)

		# RandomMosaicing
		img, bboxes, points = self.mosaic(img, bboxes, points)
		# RandomHorizontalFlip
		img, bboxes, points = self.hflip(img, bboxes, points)
		# RandomVerticalFlip
		img, bboxes, points = self.vflip(img, bboxes, points)

		return img, bboxes, points


	def _val_transform(self, img, bboxes, points):
		img, bboxes, points = self._shared_transform(img, bboxes, points)
		return img, bboxes, points


	def _test_transform(self, img, bboxes, points):
		img, bboxes, points = self._shared_transform(img, bboxes, points)
		return img, bboxes, points


	def _sample_transform(self, img, bboxes, points):
		img, bboxes, points = self._shared_transform(img, bboxes, points)
		# img, bboxes, points = v2.Resize(size=(256, 256))(img, bboxes, points)
		# img, bboxes, points = v2.Pad(padding=[128, 128, 128, 128])(img, bboxes, points)
		return img, bboxes, points


	def transform(self, img, bboxes, points):
		if self.split == "train":
			img, bboxes, points = self._train_transform(img, bboxes, points)
		elif self.split == "val":
			img, bboxes, points = self._val_transform(img, bboxes, points)
		elif self.split == "test":
			img, bboxes, points = self._test_transform(img, bboxes, points)
		elif self.split == "sample":
			img, bboxes, points = self._sample_transform(img, bboxes, points)
		
		bboxes = self.sanitize_bboxes(bboxes)
		img = (img * 2.0 - 1.0).clamp(-1.0, 1.0)

		return img, bboxes, points


	def target_transform(self, target, tlrb):
		# MinMax Normalization
		if self.target_minmax_norm:
			target = self.minmaxnorm(target)

		target = (target * 2.0 - 1.0).clamp(-1.0, 1.0)
		tlrb = (tlrb * 2.0 - 1.0).clamp(-1.0, 1.0)
		return target, tlrb


	def get_by_name(self, name):
		return self.__getitem__(self.img_names.index(name))

	def __len__(self):
		return len(self.img_names)

	def __getitem__(self, index):
		if th.is_tensor(index):
			index = index.tolist()

		img = Image.open(
			osp.join(
				self.imgdir,
				self.img_names[index]
			)
		).convert('RGB')
		w, h = img.size

		bboxes = th.as_tensor(
			self.annotations[index]['boxes'], 
			dtype=th.float32
		)
		# print(self.annotations[index])
		assert len(bboxes) >= self.n_exemplars, f'Not enough examplars for image {self.img_names[index]}'
		bboxes = box_convert(bboxes, "xywh", "xyxy")

		bboxes[:, 0] = th.clip(bboxes[:, 0], 0, w-1)
		bboxes[:, 1] = th.clip(bboxes[:, 1], 0, h-1)
		bboxes[:, 2] = th.clip(bboxes[:, 2], 0, w-1)
		bboxes[:, 3] = th.clip(bboxes[:, 3], 0, h-1)

		# bboxes = bboxes[th.randperm(bboxes.shape[0])]
		bboxes = bboxes[:self.n_exemplars, ...]	# (x_min, y_min, x_max, y_max)
		# bboxes = BoundingBoxes(
		# 	bboxes, 
		# 	format="XYXY", 
		# 	canvas_size=(h, w)
		# )
		bboxes = BoundingBoxes(
			th.cat([
				bboxes, 
				self.get_gt_bboxes(index)
			]), 
			format="XYXY", 
			canvas_size=(h, w)
		)

		points = th.as_tensor(self.annotations[index]['points']).float()
		# add 1.0 to all points to compensate for minus 1.0 in generate density map (turns out LVIS alredy accounts for start index 0)
		points += 1
		points = Points(points, canvas_size=(h, w))

		# text = self.img_classes[self.img_names[index]]
		img_id = osp.splitext(self.img_names[index])[0]
		target_count = th.tensor(len(points), dtype=th.float32)

		img, bboxes, points = self.transform(img, bboxes, points)
		# bboxes = bboxes[:self.n_exemplars, ...]
		bboxes, gt_bboxes = (
			bboxes[:self.n_exemplars], 
			bboxes[self.n_exemplars:]
		)
		# print(gt_bboxes.shape)
		target = generate_density_map(
			size=(self.image_size, self.image_size),
			points=points,
			sigma=self.sigma
		)
		# target = self.target_transform(
		# 	target
		# )

		tlrb, tlrb_wm = generate_tlrb(
			locs=self.locs,
			bboxes=gt_bboxes,
			center_sample=self.tlrb_center_sample,
			radius=self.tlrb_radius,
			point_sample=self.tlrb_point_sample,
			points=points,
			force_square=self.force_square_tlrb,
		)
		tlrb_mask = (tlrb > 0.0).float()[0].unsqueeze(0)
		target, tlrb = self.target_transform(
			target, tlrb
		)

		if self.with_tlrb:
			target = th.cat((target, tlrb), dim=0)

		assert target.shape[-2:] == img.shape[-2:], "target shape does not match image shape."
		return target, dict(bboxes=bboxes,
							img=img,
							count=target_count,
							id=img_id,
							text="",
							points=points.float(),
							tlrb_mask=tlrb_mask,
							tlrb_wm=tlrb_wm,
							gt_bboxes=gt_bboxes,
							predpoints=th.tensor([[]], dtype=th.float32),
							og_size=(h, w)
							)



class CA44(Dataset):

	def __init__(
		self, 
		datadir, 
		image_size=512, 
		split='train', 
		n_exemplars=3,
		mosaic_p=0.0,
		mosaic_avgbbox_w_lim=50,
		mosaic_avgbbox_h_lim=50,
		mosaic_avgbbox_area_lim=720,
		hflip_p=0.0,
		vflip_p=0.0,
		cj_p=0.0,
		sigma=0.5,
		target_minmax_norm=False,
		with_tlrb=True,
		tlrb_center_sample=False,
		tlrb_point_sample=False,
		tlrb_radius=8,
		use_bbox_center_as_point=False,
		force_square_tlrb=False,
		zero_shot=False, 
		return_ids=False,
	):
		# check if datadir + '/3exampler_'+split+'_filtered.json' exists
		self.split = split if split != 'val' else 'valid'
		if not os.path.exists(datadir + '/3exampler_'+self.split+'_filtered.json'):
			self.image_names = []
		else:
			self.datadir = datadir
			# self.split = split
			self.n_exemplars = n_exemplars
			self.image_size = image_size
			self.hflip_p = hflip_p
			self.vflip_p = vflip_p
			self.cj_p = cj_p
			self.mosaic_p = mosaic_p
			self.sigma = sigma
			self.target_minmax_norm = target_minmax_norm
			self.with_tlrb = with_tlrb
			self.tlrb_center_sample = tlrb_center_sample
			self.tlrb_point_sample = tlrb_point_sample
			self.tlrb_radius = tlrb_radius
			self.use_bbox_center_as_point = use_bbox_center_as_point
			self.force_square_tlrb = force_square_tlrb
			self.zero_shot = zero_shot
			self.return_ids = return_ids

			self.resize = v2.Resize(size=(image_size, image_size))
			self.hflip = v2.RandomHorizontalFlip(p=hflip_p)
			self.vflip = v2.RandomVerticalFlip(p=vflip_p)
			self.cj = ColorJitter(0.2, 0.2, 0.2, 0.1)
			self.mosaic = RandomMosaic(
				p=mosaic_p, 
				avgbbox_h_lim=mosaic_avgbbox_h_lim, 
				avgbbox_w_lim=mosaic_avgbbox_w_lim, 
				avgbbox_area_lim=mosaic_avgbbox_area_lim, 
				hflip_p=0.5)
			self.sanitize_bboxes = v2.SanitizeBoundingBoxes(labels_getter=None)
			self.to_tensor = v2.Compose([
				v2.ToImage(),
				v2.ToDtype(th.float32, scale=True)
			])
			self.minmaxnorm = MinMaxNormalize()

			with open(datadir + '/3exampler_'+self.split+'_filtered.json', 'r') as f:
				self.exemplars = json.load(f)

			#open COCO annotations
			with open(datadir + '/CA_mini10_'+self.split+'_filtered.json', 'r') as f:
				self.annotations = json.load(f)

			#coco format 
			self.labels = COCO(os.path.join(datadir, 'CA_mini10_' + self.split + '_filtered.json'))
			# get image names
			self.image_names = [self.annotations['images'][i]['file_name'] for i in range(len(self.annotations['images']))]
			self.map_name_2_id = self.mapping()

			# filter out images with no annotations
			image_names = []
			exemplars = {}
			annotations = []
			for i in range(len(self.image_names)):
				if self.labels.getAnnIds([i]) != []:
					image_names.append(self.image_names[i])
					exemplars[self.image_names[i]] = self.exemplars[self.image_names[i]]
					annotations.append(self.annotations['images'][i])
			self.image_names = image_names
			self.exemplars = exemplars  
			self.annotations['images'] = annotations

			self.locs = compute_locations(
				h=self.image_size,
				w=self.image_size,
				stride=1,
				device="cpu"
			)
		
	def mapping(self, ):
		all_coco_imgs = self.labels.imgs
		map_name_2_id = dict()
		for k, v in all_coco_imgs.items():
			img_id = v["id"]
			img_name = v["file_name"]
			map_name_2_id[img_name] = img_id
		return map_name_2_id    
	
	def get_gt_bboxes(self, idx):
		idx = self.map_name_2_id[self.image_names[idx]]
		anno_ids = self.labels.getAnnIds([idx])
		annotations = self.labels.loadAnns(anno_ids)
		bboxes = []
		for a in annotations:
			bboxes.append((th.tensor(a['bbox'])))
		if len(bboxes) == 0:
			print("issues")
		bboxes=th.stack(bboxes)
		return box_convert(bboxes, in_fmt='xywh', out_fmt='xyxy')
	

	def _shared_transform(self, img, bboxes, points):
		# ToImage
		img = self.to_tensor(img)
		# Resize
		img, bboxes, points = self.resize(img, bboxes, points)
		return img, bboxes, points


	def _train_transform(self, img, bboxes, points):
		img, bboxes, points = self._shared_transform(img, bboxes, points)

		# RandomMosaicing
		img, bboxes, points = self.mosaic(img, bboxes, points, self.n_exemplars)
		# RandomHorizontalFlip
		img, bboxes, points = self.hflip(img, bboxes, points)
		# RandomVerticalFlip
		img, bboxes, points = self.vflip(img, bboxes, points)
		# RandomColorJitter
		if th.rand(1) < self.cj_p:
			img = self.cj(img)

		return img, bboxes, points


	def _val_transform(self, img, bboxes, points):
		img, bboxes, points = self._shared_transform(img, bboxes, points)
		return img, bboxes, points


	def _test_transform(self, img, bboxes, points):
		img, bboxes, points = self._shared_transform(img, bboxes, points)
		return img, bboxes, points
	

	def _sample_transform(self, img, bboxes, points):
		img, bboxes, points = self._shared_transform(img, bboxes, points)
		return img, bboxes, points


	def transform(self, img, bboxes, points):
		if self.split == "train":
			img, bboxes, points = self._train_transform(img, bboxes, points)
		elif self.split == "valid":
			img, bboxes, points = self._val_transform(img, bboxes, points)
		elif self.split == "test":
			img, bboxes, points = self._test_transform(img, bboxes, points)
		elif self.split == "sample":
			img, bboxes, points = self._sample_transform(img, bboxes, points)
		
		bboxes = self.sanitize_bboxes(bboxes)
		img = (img * 2.0 - 1.0).clamp(-1.0, 1.0)

		return img, bboxes, points
	

	def target_transform(self, target, tlrb):
		# MinMax Normalization
		if self.target_minmax_norm:
			target = self.minmaxnorm(target)

		target = (target * 2.0 - 1.0).clamp(-1.0, 1.0)
		tlrb = (tlrb * 2.0 - 1.0).clamp(-1.0, 1.0)
		return target, tlrb


	def get_by_name(self, name):
		return self.__getitem__(self.img_names.index(name))
	
	def __getitem__(self, idx):
		self.annotations['images'][idx]
		img = Image.open(os.path.join(
					self.datadir,
					self.split,
					self.image_names[idx]
				)).convert("RGB")
		w, h = img.size
		img = self.to_tensor(img)
		bboxes = box_convert(th.tensor(
			self.exemplars[self.image_names[idx]],
			dtype=th.float32
		), in_fmt='xywh', out_fmt='xyxy')[:self.n_exemplars]
		gt_bboxes = self.get_gt_bboxes(idx).to(th.float32)

		bboxes = BoundingBoxes(
			th.cat([
				bboxes, 
				gt_bboxes
			]), 
			format="XYXY", 
			canvas_size=(h, w)
		)

		img_id = os.path.splitext(self.image_names[idx])[0]
		points = (bboxes[:, :2] + bboxes[:, 2:]) / 2.0  # center points
		points = Points(points, canvas_size=(h, w))
		target_count = th.tensor(len(points), dtype=th.float32)

		img, bboxes, points = self.transform(img, bboxes, points)
		bboxes, gt_bboxes = (
			bboxes[:self.n_exemplars], 
			bboxes[self.n_exemplars:]
		)

		target = generate_density_map(
			size=(self.image_size, self.image_size),
			points=points,
			sigma=self.sigma
		)
		tlrb, tlrb_wm = generate_tlrb(
			locs=self.locs,
			bboxes=gt_bboxes,
			center_sample=self.tlrb_center_sample,
			radius=self.tlrb_radius,
			point_sample=self.tlrb_point_sample,
			points=points,
			force_square=self.force_square_tlrb,
		)
		tlrb_mask = (tlrb > 0.0).float()[0].unsqueeze(0)
		target, tlrb = self.target_transform(
			target, tlrb
		)

		if self.with_tlrb:
			target = th.cat((target, tlrb), dim=0)

		assert target.shape[-2:] == img.shape[-2:], "target shape does not match image shape."
		target_count = th.tensor(len(points), dtype=th.float32)

		return target, dict(bboxes=bboxes,
							img=img,
							tlrb_mask=tlrb_mask,
							tlrb_wm=tlrb_wm,
							count=target_count,
							id=img_id,
							text="",
							points=points.float(),
							predpoints=th.tensor([[]], dtype=th.float32),
							gt_bboxes=gt_bboxes,
							og_size=(h, w),
							)

	def __len__(self):
		return len(self.image_names)

def full_dataset(
	datadir, 
	image_size=512, 
	split='train', 
	n_exemplars=3,
	mosaic_p=0.0,
	mosaic_avgbbox_w_lim=50,
	mosaic_avgbbox_h_lim=50,
	mosaic_avgbbox_area_lim=720,
	hflip_p=0.0,
	vflip_p=0.0,
	cj_p=0.0,
	sigma=0.5,
	target_minmax_norm=False,
	with_tlrb=True,
	tlrb_center_sample=False,
	tlrb_point_sample=False,
	tlrb_radius=8,
	use_bbox_center_as_point=False,
	force_square_tlrb=False,
	zero_shot=False, 
	return_ids=False,
):
	subdatasets = os.listdir(datadir)
	datasets = []
	for subdataset in subdatasets:
		for subusbdataset in os.listdir(os.path.join(datadir, subdataset)):
			#for split in ['train', 'val', 'test']:
			data = CA44(
				datadir=os.path.join(datadir, subdataset, subusbdataset), 
				image_size=image_size, 
				split=split, 
				n_exemplars=n_exemplars,
				mosaic_p=mosaic_p,
				mosaic_avgbbox_w_lim=mosaic_avgbbox_w_lim,
				mosaic_avgbbox_h_lim=mosaic_avgbbox_h_lim,
				mosaic_avgbbox_area_lim=mosaic_avgbbox_area_lim,
				hflip_p=hflip_p,
				vflip_p=vflip_p,
				cj_p=cj_p,
				sigma=sigma,
				target_minmax_norm=target_minmax_norm,
				with_tlrb=with_tlrb,
				tlrb_center_sample=tlrb_center_sample,
				tlrb_point_sample=tlrb_point_sample,
				tlrb_radius=tlrb_radius,
				use_bbox_center_as_point=use_bbox_center_as_point,
				force_square_tlrb=force_square_tlrb,
				zero_shot=zero_shot, 
				return_ids=return_ids,
			) 
			if len(data.image_names) > 0:
				datasets.append(data)
				#print("----->",subusbdataset,split)
	return th.utils.data.ConcatDataset(datasets)

class FullCA44(Dataset):
	def __init__(
		self, 
		datadir, 
		image_size=512, 
		split='train', 
		n_exemplars=3,
		mosaic_p=0.0,
		mosaic_avgbbox_w_lim=50,
		mosaic_avgbbox_h_lim=50,
		mosaic_avgbbox_area_lim=720,
		hflip_p=0.0,
		vflip_p=0.0,
		cj_p=0.0,
		sigma=0.5,
		target_minmax_norm=False,
		with_tlrb=True,
		tlrb_center_sample=False,
		tlrb_point_sample=False,
		tlrb_radius=8,
		use_bbox_center_as_point=False,
		force_square_tlrb=False,
		zero_shot=False, 
		return_ids=False,
	):
		self.dataset = full_dataset(
			datadir=datadir, 
			image_size=image_size, 
			split=split, 
			n_exemplars=n_exemplars,
			mosaic_p=mosaic_p,
			mosaic_avgbbox_w_lim=mosaic_avgbbox_w_lim,
			mosaic_avgbbox_h_lim=mosaic_avgbbox_h_lim,
			mosaic_avgbbox_area_lim=mosaic_avgbbox_area_lim,
			hflip_p=hflip_p,
			vflip_p=vflip_p,
			cj_p=cj_p,
			sigma=sigma,
			target_minmax_norm=target_minmax_norm,
			with_tlrb=with_tlrb,
			tlrb_center_sample=tlrb_center_sample,
			tlrb_point_sample=tlrb_point_sample,
			tlrb_radius=tlrb_radius,
			use_bbox_center_as_point=use_bbox_center_as_point,
			force_square_tlrb=force_square_tlrb,
			zero_shot=zero_shot, 
			return_ids=return_ids,
		)

	def __getitem__(self, idx):
		return self.dataset[idx]
	
	def __len__(self):
		return len(self.dataset)


def max_seq_collate(batch):
	max_points = max(map(lambda b: len(b[1]["points"]), batch))
	has_predpoints = "predpoints" in batch[0][1]
	has_gtbboxes = "gt_bboxes" in batch[0][1]
	if has_predpoints:
		max_predpoints = max(map(lambda b: len(b[1]["predpoints"]), batch))
	if has_gtbboxes:
		max_gtbboxes = max(map(lambda b: len(b[1]["gt_bboxes"]), batch))
	for _, cond in batch:
		p = max_points - len(cond["points"])
		cond["points"] = F.pad(cond["points"], padding=[0, 0, 0, p])
		if has_gtbboxes:
			cond["gt_bboxes"] = F.pad(cond["gt_bboxes"], padding=[0, 0, 0, max_gtbboxes - len(cond["gt_bboxes"])])
		if has_predpoints:
			pp = max_predpoints - len(cond["predpoints"])
			cond["predpoints"] =  F.pad(cond["predpoints"], padding=[0, 0, 0, pp])
	return default_collate(batch)


def load_data(
	*,
	dataset,
	batch_size,
	sampler=None,
	num_workers=0,
	shuffle=False,
	pin_memory=True
):
	"""
	For a dataset, create a dataloader of (target, kwargs) pairs.

	Each images is an NCHW float tensor, and the kwargs dict contains zero or
	more keys, each of which map to a batched Tensor of their own.

	:param dataset: The dataset to iterate over.
	:param batch_size: the batch size of each returned pair.
	"""
	loader = DataLoader(
		dataset=dataset,
		sampler=sampler,
		batch_size=batch_size, 
		num_workers=num_workers, 
		shuffle=shuffle,
		pin_memory=pin_memory,
		collate_fn=max_seq_collate,
	)
	return loader