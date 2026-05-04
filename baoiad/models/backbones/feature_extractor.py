"""Generic multi-scale feature extractor wrapping torchvision backbones.

Registered as 'FeatureExtractor' in MODELS registry for mmdetection-style
config-driven backbone construction via ``MODELS.build(cfg)``.
"""

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from baoiad.registry import MODELS


@MODELS.register_module(force=True)
class FeatureExtractor(BaseModule):
    """Multi-scale feature extractor built from torchvision backbones.

    Extracts intermediate features from ResNet-family models at specified layers.

    Args:
        backbone_name (str): Name recognized by ``backbone_utils.build_backbone``
            (e.g. ``'wide_resnet50_2'``, ``'resnet18'``).
        pretrained (bool): Load ImageNet-pretrained weights. Default True.
        out_indices (tuple[int]): 1-indexed layer indices to return features from.
            1 → layer1, 2 → layer2, 3 → layer3, 4 → layer4. Default ``(1, 2, 3)``.
        frozen (bool): Freeze all parameters. Default True.

    Example::

        # Config-driven:
        backbone = dict(type='FeatureExtractor', backbone_name='wide_resnet50_2',
                        out_indices=(1, 2, 3), frozen=True)
        fe = MODELS.build(backbone)
        features = fe(images)  # list of [B, C_i, H_i, W_i]
    """

    def __init__(self, backbone_name='wide_resnet50_2', pretrained=True,
                 out_indices=(1, 2, 3), frozen=True, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        from baoiad.models.backbone_utils import build_backbone, get_channel_dims

        net = build_backbone(backbone_name, pretrained=pretrained)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layers = nn.ModuleList([net.layer1, net.layer2, net.layer3, net.layer4])
        self.out_indices = tuple(out_indices)

        ch = get_channel_dims(backbone_name)
        self.channels = tuple(ch[i - 1] for i in self.out_indices)
        self.out_channels = self.channels

        if frozen:
            self.eval()
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x):
        """Extract multi-scale features.

        Returns:
            list[Tensor]: Feature maps from layers specified by ``out_indices``.
        """
        x = self.stem(x)
        features = []
        for i, layer in enumerate(self.layers, start=1):
            x = layer(x)
            if i in self.out_indices:
                features.append(x)
        return features

    def train(self, mode=True):
        """Keep frozen layers in eval mode."""
        if mode and not any(p.requires_grad for p in self.parameters()):
            # Frozen: stay in eval mode
            return super().train(False)
        return super().train(mode)
