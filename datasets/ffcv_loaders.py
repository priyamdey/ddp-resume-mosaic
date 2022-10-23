
from pathlib import Path

import ffcv
import ffcv.transforms as transforms
import numpy as np
import torch
import torchvision.transforms as T
from composer.datasets.ffcv_utils import ffcv_monkey_patches
from ffcv.fields.decoders import (CenterCropRGBImageDecoder, IntDecoder,
                                  RandomResizedCropRGBImageDecoder)


class ContiguousTensor(torch.nn.Module):
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return img.contiguous(memory_format=torch.channels_last)


def get_ffcv_loaders(args, head_flag=False):
    this_device=torch.device(f'cuda:{args.local_rank}')
    IMAGENET_CHANNEL_MEAN = (0.485 * 255, 0.456 * 255, 0.406 * 255)
    IMAGENET_CHANNEL_STD = (0.229 * 255, 0.224 * 255, 0.225 * 255)
    train_image_pipeline = [
        RandomResizedCropRGBImageDecoder((176, 176)),
        transforms.RandomHorizontalFlip(),
    ]
    test_image_pipeline = [
        CenterCropRGBImageDecoder((224, 224), ratio=224/232), # ratio=crop_size/resize_size
    ]
    common_transform = [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.ToDevice(this_device, non_blocking=True),
        transforms.ToTorchImage(),
        transforms.NormalizeImage(
            np.array(IMAGENET_CHANNEL_MEAN), 
            np.array(IMAGENET_CHANNEL_STD),
            np.float32),
    ]
    train_image_pipeline.extend(common_transform)
    test_image_pipeline.extend(common_transform)
    label_pipeline = [
        IntDecoder(), 
        transforms.ToTensor(), 
        transforms.Squeeze(),
        transforms.ToDevice(this_device, non_blocking=True)
    ]

    ffcv_monkey_patches()
    train_dataloader = ffcv.Loader(
        Path(args.data) / 'train.ffcv',
        batch_size=args.train_bs * (2 if head_flag else 1),
        num_workers=8,
        order=ffcv.loader.OrderOption.RANDOM,
        distributed=True,
        seed=args.seed,
        pipelines={
            'image': train_image_pipeline,
            'label': label_pipeline
        },
        batches_ahead=2,
        drop_last=True,
    )
    test_dataloader = ffcv.Loader(
        Path(args.data) / 'val.ffcv',
        batch_size=args.test_bs,
        num_workers=8,
        order=ffcv.loader.OrderOption.SEQUENTIAL,
        distributed=True,
        seed=args.seed,
        pipelines={
            'image': test_image_pipeline,
            'label': label_pipeline
        },
        batches_ahead=2,
        drop_last=False,
    )
    return train_dataloader, test_dataloader
