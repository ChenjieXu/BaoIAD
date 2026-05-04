"""Anomaly detection segmentation head registered in MODELS registry."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from baoiad.registry import MODELS


@MODELS.register_module(force=True)
class AnomalySegmentationHead(BaseModule):
    """Simple convolutional segmentation head for anomaly map prediction.

    Takes concatenated feature maps and outputs a 1-channel anomaly map.

    Args:
        in_channels (int): Number of input channels.
        mid_channels (int): Hidden channel dimension. Default 64.
        out_channels (int): Output channels. Default 1.
        num_convs (int): Number of conv layers. Default 2.
    """

    def __init__(self, in_channels, mid_channels=64, out_channels=1, num_convs=2, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        layers = []
        ch = in_channels
        for i in range(num_convs - 1):
            layers.extend([
                nn.Conv2d(ch, mid_channels, 3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
            ])
            ch = mid_channels
        layers.append(nn.Conv2d(ch, out_channels, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@MODELS.register_module(force=True)
class AnomalyClassificationHead(BaseModule):
    """Simple FC head for image-level anomaly classification.

    Args:
        in_channels (int): Input feature dimension.
        num_classes (int): Number of output classes. Default 1 (binary).
        pool_type (str): 'avg' or 'max' spatial pooling. Default 'avg'.
    """

    def __init__(self, in_channels, num_classes=1, pool_type='avg', init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        """
        Args:
            x: Feature tensor (B, C, H, W) or (B, C).
        Returns:
            Logits (B, num_classes).
        """
        if x.dim() == 4:
            x = self.pool(x).flatten(1)
        return self.fc(x)


@MODELS.register_module(force=True)
class ProjectionHead(BaseModule):
    """MLP projection head for contrastive/self-supervised methods.

    Args:
        in_channels (int): Input feature dimension.
        proj_dim (int): Projection dimension. Default 128.
        hidden_dim (int): Hidden dimension. Default 512.
    """

    def __init__(self, in_channels, proj_dim=128, hidden_dim=512, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x):
        if x.dim() > 2:
            x = x.flatten(1)
        return self.net(x)
