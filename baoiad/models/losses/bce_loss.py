"""BCE loss modules registered in MODELS registry."""
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from baoiad.registry import MODELS


@MODELS.register_module()
class BCEWithLogitsLoss(BaseModule):
    """Registered BCE with logits loss.

    Args:
        reduction (str): 'mean', 'sum', or 'none'.
        loss_weight (float): Weight factor.
    """

    def __init__(self, reduction='mean', loss_weight=1.0, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, **kwargs):
        return self.loss_weight * F.binary_cross_entropy_with_logits(
            pred, target, reduction=self.reduction)


@MODELS.register_module()
class BCELoss(BaseModule):
    """Registered BCE loss (expects sigmoid-activated inputs).

    Args:
        reduction (str): 'mean', 'sum', or 'none'.
        loss_weight (float): Weight factor.
    """

    def __init__(self, reduction='mean', loss_weight=1.0, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, **kwargs):
        return self.loss_weight * F.binary_cross_entropy(
            pred, target, reduction=self.reduction)
