"""Focal Loss modules registered in MODELS registry."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from baoiad.registry import MODELS


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = 'none',
) -> torch.Tensor:
    """Sigmoid focal loss (equivalent to torchvision.ops.sigmoid_focal_loss).

    Args:
        inputs: Predicted logits of any shape.
        targets: Ground truth of the same shape as inputs.
        alpha: Weighting factor. Use -1 for no weighting.
        gamma: Focusing parameter.
        reduction: 'none', 'mean', or 'sum'.
    """
    p = torch.sigmoid(inputs)
    bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = bce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


@MODELS.register_module()
class BinaryFocalLoss(BaseModule):
    """Focal loss for binary classification / segmentation.

    Args:
        alpha (float): Weighting factor. Default 0.25.
        gamma (float): Focusing parameter. Default 2.0.
        reduction (str): 'mean', 'sum', or 'none'.
        loss_weight (float): Weight factor.
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', loss_weight=1.0, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, **kwargs):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        if self.reduction == 'mean':
            return self.loss_weight * focal.mean()
        elif self.reduction == 'sum':
            return self.loss_weight * focal.sum()
        return self.loss_weight * focal


@MODELS.register_module()
class FocalLoss(BaseModule):
    """Multiclass Focal Loss (cross-entropy based).

    Args:
        alpha (float): Weighting factor. Default 1.0.
        gamma (float): Focusing parameter. Default 2.0.
        reduction (str): 'mean', 'sum', or 'none'.
        loss_weight (float): Weight factor.
    """

    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', loss_weight=1.0, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, **kwargs):
        ce = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce)
        focal = self.alpha * ((1 - pt) ** self.gamma) * ce
        if self.reduction == 'mean':
            return self.loss_weight * focal.mean()
        elif self.reduction == 'sum':
            return self.loss_weight * focal.sum()
        return self.loss_weight * focal
