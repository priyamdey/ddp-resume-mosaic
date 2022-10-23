# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Core Exponential Moving Average (EMA) classes and functions."""

from __future__ import annotations

import copy
import logging
from typing import Optional, Union

import torch
import torch.nn as nn
from composer.core import Algorithm, Event, State, Time, TimeUnit
from composer.core.data_spec import ensure_data_spec
from composer.loggers import Logger
from composer.trainer.devices import Device, DeviceCPU, DeviceGPU
from composer.utils.module_surgery import update_params_in_optimizer
from torch.utils.data import DataLoader, DistributedSampler

from datasets.ffcv_loaders import get_ffcv_loaders

log = logging.getLogger(__name__)

__all__ = ['HeadAdder']


class HeadAdder(Algorithm):

    def __init__(self, surgery_time: str):
        self.surgery_time = surgery_time

        # Check timestrings are parsable and convert into time object
        self.surgery_time = Time.from_timestring(surgery_time)

        # Verify that the time strings have supported units.
        if self.surgery_time.unit not in [TimeUnit.EPOCH]:
            raise ValueError(f'split_start must be EPOCH, got {self.surgery_time.unit}')
        

    def _get_time(self, state: State):
        """helper function to retrieve either the epoch or the duration depending on the units"""
        unit = self.surgery_time.unit

        if unit == TimeUnit.EPOCH:
            return state.timestamp.epoch
        elif unit == TimeUnit.DURATION:
            time_elapsed = state.get_elapsed_duration()
            assert time_elapsed is not None, 'Time should have been set on BATCH_END or EPOCH_END.'
            return time_elapsed
        else:
            raise ValueError('units must be in epoch or duration.')

    def match(self, event: Event, state: State) -> bool:
        if event != Event.EPOCH_START:
            return False
        return self._get_time(state) == self.surgery_time

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        main_module = state.model.module
        orig_model = copy.deepcopy(main_module.model)
        args = main_module.args
        args.new_head = True
        device = get_device('gpu')
        linear_head = nn.Linear(orig_model.fc.out_features, args.num_classes)
        linear_head = device.module_to_device(linear_head)
        setattr(orig_model, main_module.NEW_HEAD, linear_head)
        state.model.module.init(orig_model, args)
        state.evaluators[0].metrics = state.model.module.metrics(train=False)
        update_params_in_optimizer(state.optimizers[0].param_groups[0]['params'], state.model.module.parameters(), state.optimizers)
        print('Updated Optimizer')

        state.schedulers[0].optimizer = state.optimizers[0]
        print('Updated Scheduler')

        train_loader, _ = get_ffcv_loaders(args, head_flag=True)
        data_spec = ensure_data_spec(train_loader)
        state.set_dataloader(data_spec.dataloader, 'train')
        state.train_dataloader = state.dataloader
        state.grad_accum = args.bs_multiplier * state.grad_accum
        spin_dataloaders(state)  # not sure if this need to be called here


def spin_dataloaders(state):  # copied from trainer
    dataloader = state.dataloader
    assert dataloader is not None, 'train dataloader is set on state after FIT_START'
    for epoch in range(int(state.timestamp.epoch)):
        if isinstance(dataloader, DataLoader) and isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch)
        for _ in dataloader:
            break


def get_device(device: Optional[Union[str, Device]]):
    if not device:
        device = DeviceGPU() if torch.cuda.is_available() else DeviceCPU()
    elif isinstance(device, str):
        if device.lower() == 'cpu':
            device = DeviceCPU()
        elif device.lower() == 'gpu':
            device = DeviceGPU()
        else:
            raise ValueError(f'device ({device}) must be one of (cpu, gpu).')
    return device
