"""CSFlow multi-scale EfficientNet-B5 feature extractor.

Registered as 'CSFlowFeatureExtractor' in MODELS registry.
"""
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from baoiad.registry import MODELS


@MODELS.register_module()
class CSFlowFeatureExtractor(BaseModule):
    """Multi-scale EfficientNet-B5 feature extractor for CS-Flow.

    Extracts features at multiple scales by resizing input and passing
    through EfficientNet-B5 features[:7] (304 channels output).

    Args:
        n_scales (int): Number of scales. Default 3.
        input_size (tuple): Input image size (H, W). Default (256, 256).
        frozen (bool): Freeze all parameters. Default True.
    """

    def __init__(self, n_scales=3, input_size=(256, 256), frozen=True, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.n_scales = n_scales
        self.input_size = input_size

        from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights
        backbone = efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1)
        # torchvision's `features.6.8` node is exactly the output of `features[:7]`.
        self.features = backbone.features[:7]

        if frozen:
            self.eval()
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x):
        output = []
        for scale in range(self.n_scales):
            if scale > 0:
                feat_s = F.interpolate(
                    x, size=(self.input_size[0] // (2 ** scale),
                             self.input_size[1] // (2 ** scale)),
                )
            else:
                feat_s = x
            feat_s = self.features(feat_s)
            output.append(feat_s)
        return output

    def train(self, mode=True):
        if mode and not any(p.requires_grad for p in self.parameters()):
            return super().train(False)
        return super().train(mode)
