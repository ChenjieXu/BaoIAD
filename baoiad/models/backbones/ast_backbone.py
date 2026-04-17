"""AST feature extractors: EfficientNet block-level and generic timm extractors.

Moved from detectors/ast.py to centralize backbone/feature extraction logic.
"""

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from baoiad.registry import MODELS


@MODELS.register_module()
class EfficientNetFeatureExtractor(BaseModule):
    """Extract features from a specific layer of EfficientNet-b5."""

    def __init__(self, model_name='tf_efficientnet_b5', layer_idx=35, pretrained=True, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        import timm
        self.net = timm.create_model(model_name, pretrained=pretrained, features_only=False)
        self.layer_idx = layer_idx
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        x = self.net.conv_stem(x)
        x = self.net.bn1(x)
        for idx, block in enumerate(self.net.blocks):
            if isinstance(block, nn.Sequential):
                for sub_idx, sub_block in enumerate(block):
                    x = sub_block(x)
            else:
                x = block(x)
        return x


@MODELS.register_module()
class EfficientNetLayerExtractor(BaseModule):
    """Extract features from EfficientNet at a specific block index (flat enumeration)."""

    def __init__(self, model_name='tf_efficientnet_b5', extract_layer=35, pretrained=True, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        import timm
        self.net = timm.create_model(model_name, pretrained=pretrained, features_only=False)
        self.extract_layer = extract_layer
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        x = self.net.conv_stem(x)
        x = self.net.bn1(x)
        block_idx = 0
        for stage in self.net.blocks:
            if isinstance(stage, nn.Sequential):
                for sub_block in stage:
                    x = sub_block(x)
                    if block_idx == self.extract_layer:
                        return x
                    block_idx += 1
            else:
                x = stage(x)
                if block_idx == self.extract_layer:
                    return x
                block_idx += 1
        return x


@MODELS.register_module()
class GenericFeatureExtractor(BaseModule):
    """Extract features from any timm model using features_only mode.

    Works with ResNet, EfficientNet, and other timm backbones.
    Returns features from a specified output index.
    """

    def __init__(self, model_name='wide_resnet50_2', out_index=-1, pretrained=True, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.backbone = MODELS.build(dict(
            type='TIMMBackbone', model_name=model_name,
            pretrained=pretrained, features_only=True, frozen=True,
        ))
        self.out_index = out_index

    @torch.no_grad()
    def forward(self, x):
        feats = self.backbone(x)
        return feats[self.out_index]
