"""L1 loss modules registered in MODELS registry."""
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from baoiad.registry import MODELS


@MODELS.register_module(force=True)
class L1Loss(BaseModule):
    """Registered L1 loss.

    Args:
        reduction (str): 'mean', 'sum', or 'none'.
        loss_weight (float): Weight factor.
    """

    def __init__(self, reduction='mean', loss_weight=1.0, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, **kwargs):
        return self.loss_weight * F.l1_loss(pred, target, reduction=self.reduction)


@MODELS.register_module(force=True)
class SmoothL1Loss(BaseModule):
    """Registered Smooth L1 loss.

    Args:
        beta (float): Threshold for switching between L1 and L2.
        reduction (str): 'mean', 'sum', or 'none'.
        loss_weight (float): Weight factor.
    """

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, **kwargs):
        return self.loss_weight * F.smooth_l1_loss(
            pred, target, beta=self.beta, reduction=self.reduction)
