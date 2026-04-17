"""Cosine distance loss module."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from baoiad.registry import MODELS


@MODELS.register_module()
class CosineDistanceLoss(BaseModule):
    """1 - cosine_similarity, averaged over spatial dims.

    Args:
        reduction (str): 'mean', 'sum', or 'none'.
        loss_weight (float): Weight factor.
    """

    def __init__(self, reduction='mean', loss_weight=1.0, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, **kwargs):
        cos = F.cosine_similarity(pred, target, dim=1)
        loss = 1.0 - cos
        if self.reduction == 'mean':
            return self.loss_weight * loss.mean()
        elif self.reduction == 'sum':
            return self.loss_weight * loss.sum()
        return self.loss_weight * loss
