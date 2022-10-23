from functools import partial
from typing import Any, Optional, Tuple, Union

import torch
from composer import ComposerModel
from composer.algorithms.label_smoothing import smooth_labels
from composer.loss import binary_cross_entropy_with_logits
from torch import Tensor
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import Accuracy

from metrics import HeadAccuracy


class MosaicModel(ComposerModel):
    ORIGINAL = 'original'
    NEW_HEAD = 'new_head'
    num_classes: Optional[int] = None

    def __init__(self, module: torch.nn.Module, args) -> None:
        super().__init__()
        self.init(module, args)
        
    def init(self, module, args):
        self.args = args
        self._loss_fn = binary_cross_entropy_with_logits

        self.model = module

        # Metrics for training
        self.train_metrics = Accuracy()

        # Metrics for validation
        head_metrics = {MosaicModel.ORIGINAL: Accuracy()}
        if self.args.new_head:
            head_metrics = {
                MosaicModel.ORIGINAL: HeadAccuracy(0), 
                MosaicModel.NEW_HEAD: HeadAccuracy(1)
            }
        self.val_metrics = MetricCollection(head_metrics)

        if hasattr(module, 'num_classes'):
            self.num_classes = getattr(module, 'num_classes')

    def loss(self, outputs: Any, batch: Tuple[Any, Tensor], *args, **kwargs) -> Tensor:
        loss_fn = partial(self._loss_fn, *args, **kwargs)
        smoothing_fn = partial(smooth_labels, smoothing=self.args.smoothing)
        if hasattr(self.model, MosaicModel.NEW_HEAD):
            out1, out2, tgt = outputs
            tgt = smoothing_fn(out1, tgt)
            loss = loss_fn(out1, tgt) + loss_fn(out2, tgt)
        else:
            out, tgt = outputs
            tgt = smoothing_fn(out, tgt)
            loss = loss_fn(out, tgt)
        return loss

    def metrics(self, train: bool = False) -> Union[Metric, MetricCollection]:
        return self.train_metrics if train else self.val_metrics

    def forward(self, batch: Tuple[Tensor, Any]) -> Tensor:
        inputs, targets = batch
        out1 = self.model(inputs)
        if hasattr(self.model, MosaicModel.NEW_HEAD):
            out2 = getattr(self.model, MosaicModel.NEW_HEAD)(out1.detach())
            return out1, out2, targets
        return out1, targets

    def validate(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        inputs, targets = batch
        out = self.model(inputs)
        if hasattr(self.model, MosaicModel.NEW_HEAD):
            out2 = getattr(self.model, MosaicModel.NEW_HEAD)(out.detach())
            out = torch.cat([out, out2])
            targets = targets.reshape(-1).repeat(2)
        return out, targets
