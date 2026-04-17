"""Binary Dice Loss for segmentation tasks.

Shared implementation used by multiple detectors (AdaCLIP, AA-CLIP).
"""

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from baoiad.registry import MODELS


@MODELS.register_module()
class BinaryDiceLoss(BaseModule):
    """Binary Dice loss for anomaly segmentation.

    Computes 1 - mean(Dice score) over the batch, where the Dice score for
    each sample is::

        dice = (2 * |pred ∩ target| + smooth) / (|pred| + |target| + smooth)

    Args:
        smooth (float): Smoothing constant to avoid division by zero.
            Default: 1.0.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute binary Dice loss.

        Args:
            inputs: Predicted probabilities, shape ``(N, ...)``.
            targets: Ground-truth binary masks, shape ``(N, ...)``.

        Returns:
            Scalar loss tensor.
        """
        n = targets.size(0)
        inputs_flat = inputs.reshape(n, -1)
        targets_flat = targets.reshape(n, -1)
        intersection = inputs_flat * targets_flat
        dice = (2 * intersection.sum(1) + self.smooth) / (
            inputs_flat.sum(1) + targets_flat.sum(1) + self.smooth
        )
        return 1 - dice.mean()
