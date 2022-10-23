import argparse
import os
import time
import warnings
from typing import Type

import torch
import torch.distributed as torch_dist
import torch.utils.data
import torchvision.models.resnet as resnet
from composer import Trainer
from composer import optim as composer_optim
from composer.algorithms import BlurPool, ChannelsLast, ProgressiveResizing
from composer.callbacks import LRMonitor, SpeedMonitor
from composer.loggers import (FileLogger, LogLevel, ProgressBarLogger)
from composer.utils import reproducibility

from callbacks import CheckpointSaver
from datasets import get_ffcv_loaders
from head_adder import HeadAdder
from mosaic_model import MosaicModel
from schedulers import CosineAnnealingWithWarmupScheduler


def warning_on_one_line(message: str, category: Type[Warning], filename: str, lineno: int, file=None, line=None):
    # From https://stackoverflow.com/questions/26430861/make-pythons-warnings-warn-not-mention-itself
    return f'{category.__name__}: {message} (source: {filename}:{lineno})\n'

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='imagenet_ffcv')
    parser.add_argument('--train_bs', type=int, default=2048)
    parser.add_argument('--test_bs', type=int, default=2048)

    parser.add_argument('--arch', type=str, default='resnet50')
    parser.add_argument('--new_head', type=bool, default=False)
    parser.add_argument('--num_classes', type=int, default=1000)

    # Algorithms
    parser.add_argument('--smoothing', type=float, default=0.08)
    parser.add_argument('--surgery_time', type=str, required=True)
    parser.add_argument('--bs_multiplier', type=int, default=1)

    parser.add_argument('--epochs', type=str, default='90ep', help='k ep or k ba')
    parser.add_argument('--scale_schedule_ratio', type=float, default=0.4, help='fraction of total training of 90ep')
    parser.add_argument('--eval_interval', type=str, default='1ep', help='how often to do evaluation')

    parser.add_argument('--warm', type=str, default='8ep')

    parser.add_argument('--save_interval', type=str, default='2ep', help='how often to save checkpoints')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint path')

    parser.add_argument('--save_folder', type=str, default=None)
    parser.add_argument('--precision', type=str, default='amp', help='amp / fp16 / fp32')

    # run related
    parser.add_argument('--name', type=str, required=True, help='Name this run')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--id', type=str, default=None, help='wandb id of run (use it for resuming a prev run)')
    parser.add_argument('--resume', action='store_true', help='Auto-resume an existing run. Provided to both wandb & Trainer')

    parser.add_argument('--seed', type=int, default=42)

    ## distributed
    parser.add_argument("--local_rank", default=-1, type=int, 
    help="needed as an arg if torch.distributed.launch is used without --use_env flag")

    return parser

def init_distributed_mode(args):
    # For all slurm env vars, can check here: https://hpcc.umd.edu/hpcc/help/slurmenv.html
    args.is_slurm_job = "SLURM_JOB_ID" in os.environ
    if args.is_slurm_job:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.world_size = int(os.environ["SLURM_JOB_NUM_NODES"]) * int(
            os.environ["SLURM_TASKS_PER_NODE"][0]
        )
        args.local_rank = args.rank % torch.cuda.device_count()
    else:
        # multi-GPU job (local or multi-node) - jobs started with torch.distributed.launch
        # read environment variables
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        if args.local_rank == -1: # torch.run was used or --use_env flag was used with torch.distributed.launch
            args.local_rank = int(os.environ['LOCAL_RANK']) 

    # Set env variables required by the composer Trainer
    os.environ['RANK'] = str(args.rank)
    os.environ['WORLD_SIZE'] = str(args.world_size)
    os.environ['LOCAL_RANK'] = str(args.local_rank)
    if args.is_slurm_job:
        os.environ['LOCAL_WORLD_SIZE'] = str(torch.cuda.device_count())
        os.environ['NODE_RANK'] = os.environ['SLURM_NODEID']
    else:
        # os.environ['NODE_RANK'] = os.environ['GROUP_RANK']
        pass

    torch_dist.init_process_group(backend="nccl", init_method="env://", world_size=args.world_size, rank=args.rank)
    torch.cuda.set_device(args.local_rank)
    return

def get_optimizer(model, args):
    params = model.parameters()
    optimizer = composer_optim.DecoupledSGDW(
        params,
        lr=2.048,
        momentum=0.875,
        nesterov=False,
        weight_decay=5e-4,
        dampening=0.0
    )
    scheduler = CosineAnnealingWithWarmupScheduler(
        t_warmup=args.warm,
        surgery_time=args.surgery_time,
        bs_multiplier=args.bs_multiplier,
        alpha_f=0.0
    )
    return optimizer, scheduler


def main(args) -> None:
    warnings.formatwarning = warning_on_one_line
    init_distributed_mode(args)
    # reproducibility.configure_deterministic_mode()
    reproducibility.seed_all(args.seed)

    backbone = getattr(resnet, args.arch)(num_classes=args.num_classes)
    model = MosaicModel(backbone, args)
    optimizer, scheduler = get_optimizer(model, args)
    train_dataloader, test_dataloader = get_ffcv_loaders(args)

    # algorithms
    blurpool = BlurPool(
        blur_first=True,
        min_channels=16,
        replace_convs=True,
        replace_maxpools=True
    )
    progressize_resize = ProgressiveResizing(
        delay_fraction=0.4,
        finetune_fraction=0.2,
        initial_scale=0.5,
        mode='resize',
        size_increment=4,
    )
    head_adder = HeadAdder(surgery_time=args.surgery_time)
    algorithms = [ChannelsLast(), progressize_resize, blurpool, head_adder]

    # init Trainer
    trainer = Trainer(
        run_name=args.name,
        model=model,
        optimizers=optimizer,
        schedulers=scheduler,
        max_duration=args.epochs,
        scale_schedule_ratio=args.scale_schedule_ratio,
        train_dataloader=train_dataloader,
        # train_subset_num_batches=10,
        eval_dataloader=test_dataloader,
        eval_interval=args.eval_interval,
        algorithms=algorithms,
        callbacks=[
            SpeedMonitor(window_size=100),
            LRMonitor(),
            CheckpointSaver(  # Custom checkpoint class to dump a model ckpt before head addition
                folder='experiments/{run_name}/checkpoints',
                save_interval=args.save_interval,
                overwrite=True,
                num_checkpoints_to_keep=1,
                artifact_name=None
            )
        ],
        loggers=[
            FileLogger(filename='experiments/{run_name}/logs-rank{rank}.txt'),
            ProgressBarLogger(
                log_to_console=True,
                console_log_level=LogLevel.EPOCH),
        ],
        save_folder=args.save_folder,
        autoresume=args.resume,
        device='gpu',
        precision=args.precision,
        grad_accum=2,
        grad_clip_norm=-1,
        seed=args.seed,
        load_path=args.ckpt
    )

    start_time = time.perf_counter()
    trainer.fit()
    end_time = time.perf_counter()
    print(f'\nNumber of epochs trained: {trainer.state.timestamp.epoch}')
    print(f"It took {end_time - start_time:0.4f} seconds to train")

    torch_dist.destroy_process_group()

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
