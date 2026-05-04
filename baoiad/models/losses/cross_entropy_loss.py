"""Cross Entropy Loss module registered in MODELS registry."""
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from baoiad.registry import MODELS


@MODELS.register_module(force=True)
class CrossEntropyLoss(BaseModule):
    """Registered Cross Entropy loss.

    Args:
        reduction (str): Reduction mode ('mean', 'sum', 'none').
        loss_weight (float): Weight factor for this loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, **kwargs):
        return self.loss_weight * F.cross_entropy(pred, target, reduction=self.reduction)
