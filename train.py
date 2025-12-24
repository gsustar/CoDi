import argparse
import datetime
import pprint
import shutil
import os

import os.path as osp
import torch as th
import torch.distributed as dist

from diffcount import logger
from diffcount.script_util import (
	create_model,
	create_diffusion,
	create_data,
	create_conditioner,
	create_vae,
	namespace_to_dict,
	parse_config,
	assert_config,
	seed_everything,
)
from diffcount.train_util import TrainLoop

def main():

	world_size = int(os.getenv("SLURM_NTASKS", 1))
	rank = int(os.getenv("SLURM_PROCID", 0))
	gpu = rank % th.cuda.device_count()
	print(f"{os.getenv('SLURMD_NODENAME', '')}: RANK={rank}, GPU={gpu}", flush=True)

	th.cuda.set_device(gpu)
	dev = th.device(gpu)

	dist.init_process_group(
		backend='nccl', 
		init_method='env://',
		world_size=world_size, 
		rank=rank
	)

	args = parse_args()
	config = parse_config(args.config)
	assert_config(config)
	now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

	if config.train.seed is not None:
		seed_everything(config.train.seed)

	config.log.logdir = osp.join(
		config.log.logdir,
		config.data.dataset.name,
		config.name,
		now
	) if config.log.logdir else None

	if rank == 0:
		logger.configure(
			dir=config.log.logdir, 
			format_strs=["stdout", "log", "wandb"],
			wandb_kwargs=dict(
				project="diffcount",
				name=f"{config.name}:{rank}",
				group=f"{config.name}:{now}",
				config=namespace_to_dict(config),
				mode=config.log.wandb_mode,
			),
			log_suffix=f"_train{rank}"
		)
		shutil.copy(args.config, osp.join(logger.get_dir(), "config.yaml"))
	logger.log(pprint.pformat(config))

	logger.log("creating model...")
	model = create_model(config.model)

	logger.log(f"moving model to '{dev}'...")
	model.to(dev)

	logger.log("creating diffusion...")
	diffusion = create_diffusion(config.diffusion)

	logger.log("creating VAE...")
	vae = create_vae(
		getattr(config, "vae", None), device=dev
	)

	logger.log("creating data...")
	train_data, val_data, _ = create_data(config.data, train=True, distributed=True)

	logger.log("creating conditioner...")
	conditioner = create_conditioner(
		getattr(config, "conditioner", []),
		train=True,
	)
	conditioner.to(dev)

	if not hasattr(config.train, 'zeroshot_adapt'):
		config.train.zeroshot_adapt = False
	logger.log("entering training loop...")
	TrainLoop(
		model=model,
		diffusion=diffusion,
		data=train_data,
		val_data=val_data,
		conditioner=conditioner,
		vae=vae,
		batch_size=config.data.dataloader.params.batch_size,
		lr=config.train.lr,
		ema_rate=config.train.ema_rate,
		log_interval=config.log.log_interval,
		save_interval=config.log.save_interval,
		validation_interval=config.train.validation_interval,
		resume_checkpoint=config.train.resume_checkpoint,
		weight_decay=config.train.weight_decay,
		num_epochs=config.train.num_epochs,
		device=dev,
		grad_clip=config.train.grad_clip,
		lr_scheduler=config.train.lr_scheduler,
		overfit=config.train.overfit,
		mixed_precision=config.train.mixed_precision,
		single_ch_vae=config.vae.single_ch,
		zeroshot_adapt=config.train.zeroshot_adapt,
	).run_loop()

	dist.destroy_process_group()


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str)
	return parser.parse_args()


if __name__ == "__main__":
	main()
