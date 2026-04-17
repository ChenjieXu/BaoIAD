"""Multi-scale feature pooling neck."""

from typing import Tuple

import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from torch import Tensor

from baoiad.registry import MODELS


@MODELS.register_module()
class MultiScalePooling(BaseModule):
    """Pool multi-scale features to a unified spatial size and concatenate.

    Args:
        output_size: Target spatial size for adaptive average pooling.
    """

    def __init__(self, output_size: int = 28, init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.output_size = output_size
        self.pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, feats: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        """Pool each feature map to output_size and return as tuple.

        Args:
            feats: Tuple of feature tensors from different backbone layers.

        Returns:
            Tuple of pooled feature tensors.
        """
        pooled = []
        for feat in feats:
            if feat.shape[-2:] != (self.output_size, self.output_size):
                feat = self.pool(feat)
            pooled.append(feat)
        return tuple(pooled)
