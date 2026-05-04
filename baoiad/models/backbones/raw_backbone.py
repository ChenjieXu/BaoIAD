"""Raw torchvision backbone wrapper for detectors that decompose layers.

Some detectors (e.g., PyramidFlow, SuperSimpleNet, GraphCore) need direct access
to individual layers (conv1, bn1, layer1, ...) from a torchvision ResNet.
This wrapper provides them through the registry.
"""
import torch.nn as nn
from mmengine.model import BaseModule
from torchvision import models as tv_models

from baoiad.registry import MODELS

_BACKBONE_REGISTRY = {
    'resnet18': (tv_models.resnet18, tv_models.ResNet18_Weights.IMAGENET1K_V1),
    'resnet34': (tv_models.resnet34, tv_models.ResNet34_Weights.IMAGENET1K_V1),
    'resnet50': (tv_models.resnet50, tv_models.ResNet50_Weights.IMAGENET1K_V1),
    'resnet101': (tv_models.resnet101, tv_models.ResNet101_Weights.IMAGENET1K_V1),
    'wide_resnet50_2': (tv_models.wide_resnet50_2, tv_models.Wide_ResNet50_2_Weights.IMAGENET1K_V1),
    'wide_resnet101_2': (tv_models.wide_resnet101_2, tv_models.Wide_ResNet101_2_Weights.IMAGENET1K_V1),
}

_CHANNEL_DIMS = {
    'resnet18': (64, 128, 256, 512),
    'resnet34': (64, 128, 256, 512),
    'resnet50': (256, 512, 1024, 2048),
    'resnet101': (256, 512, 1024, 2048),
    'wide_resnet50_2': (256, 512, 1024, 2048),
    'wide_resnet101_2': (256, 512, 1024, 2048),
}


@MODELS.register_module(force=True)
class RawBackbone(BaseModule):
    """Raw torchvision backbone that exposes individual layers.

    Unlike FeatureExtractor which returns feature tensors, this exposes
    the raw network for detectors that need to decompose it.

    Attributes:
        conv1, bn1, relu, maxpool: Stem layers.
        layer1, layer2, layer3, layer4: ResNet stages.
        channel_dims (tuple): Per-layer channel dimensions.

    Args:
        backbone_name (str): Backbone name. Default 'wide_resnet50_2'.
        pretrained (bool): Load pretrained weights. Default True.
        frozen (bool): Freeze all parameters. Default True.
    """

    def __init__(self, backbone_name='wide_resnet50_2', pretrained=True, frozen=True, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if backbone_name not in _BACKBONE_REGISTRY:
            raise ValueError(
                f"Unknown backbone '{backbone_name}'. "
                f"Supported: {list(_BACKBONE_REGISTRY.keys())}"
            )
        constructor, weights = _BACKBONE_REGISTRY[backbone_name]
        net = constructor(weights=weights if pretrained else None)

        # Expose individual layers
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        self.channel_dims = _CHANNEL_DIMS[backbone_name]
        self.out_channels = self.channel_dims

        if frozen:
            self.eval()
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x):
        """Forward through all layers, returning multi-scale features."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        features = []
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            x = layer(x)
            features.append(x)
        return features

    def train(self, mode=True):
        if mode and not any(p.requires_grad for p in self.parameters()):
            return super().train(False)
        return super().train(mode)
