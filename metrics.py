from torch import Tensor
from torchmetrics.classification import Accuracy


class HeadAccuracy(Accuracy):
    def __init__(self, head_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head_id = head_id

    def update(self, preds: Tensor, target: Tensor) -> None:
        pred, tgt = preds.chunk(2)[self.head_id], target.chunk(2)[self.head_id]
        super().update(pred, tgt)
