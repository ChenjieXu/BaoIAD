"""DINOv2 backbone wrapper registered in MODELS registry.

Wraps the DINOv2 ViT encoder for config-driven construction.
Uses timm for Python 3.8+ compatibility (torch.hub DINOv2 requires 3.10+).
"""
import logging
import os

import timm
import torch
import torch.nn as nn

from mmengine.model import BaseModule
from baoiad.registry import MODELS

logger = logging.getLogger(__name__)

# Mapping from torch.hub DINOv2 names to timm model names
DINOV2_HUB_TO_TIMM = {
    'dinov2_vits14': 'vit_small_patch14_dinov2.lvd142m',
    'dinov2_vitb14': 'vit_base_patch14_dinov2.lvd142m',
    'dinov2_vitl14': 'vit_large_patch14_dinov2.lvd142m',
    'dinov2_vitg14': 'vit_giant_patch14_dinov2.lvd142m',
    'dinov2_vits14_reg': 'vit_small_patch14_reg4_dinov2.lvd142m',
    'dinov2_vitb14_reg': 'vit_base_patch14_reg4_dinov2.lvd142m',
    'dinov2_vitl14_reg': 'vit_large_patch14_reg4_dinov2.lvd142m',
    'dinov2_vitg14_reg': 'vit_giant_patch14_reg4_dinov2.lvd142m',
}

DINOV2_ARCHITECTURES = {
    "s": {"embed_dim": 384, "num_heads": 6, "depth": 12,
          "target_layers": (2, 5, 8, 11)},
    "b": {"embed_dim": 768, "num_heads": 12, "depth": 12,
          "target_layers": (2, 5, 8, 11)},
    "l": {"embed_dim": 1024, "num_heads": 16, "depth": 24,
          "target_layers": (4, 11, 17, 23)},
    "g": {"embed_dim": 1536, "num_heads": 24, "depth": 40,
          "target_layers": (9, 19, 29, 39)},
}


class _DINOv2Encoder(nn.Module):
    """Wrapper around timm DINOv2 model that provides the same API as
    ``torch.hub.load('facebookresearch/dinov2', ...)``.

    Specifically, ``forward_features()`` returns a dict with key
    ``'x_norm_patchtokens'`` containing ``(B, N_patches, D)`` tensors,
    matching the interface expected by downstream consumers like MuSc.
    """

    def __init__(self, timm_model):
        super().__init__()
        self._model = timm_model
        self.embed_dim = timm_model.embed_dim
        self.num_register_tokens = 0

    def forward(self, x):
        """Return CLS token embedding (B, D), matching torch.hub DINOv2."""
        out = self._model.forward_features(x)
        # timm returns (B, N_prefix + N_patches, D)
        cls_token = out[:, 0, :]
        return cls_token

    def forward_features(self, x):
        """Return dict with 'x_norm_patchtokens' for torch.hub API compat."""
        out = self._model.forward_features(x)
        # Strip prefix tokens (CLS + optional registers)
        n_prefix = self._model.num_prefix_tokens
        patch_tokens = out[:, n_prefix:, :]
        return {'x_norm_patchtokens': patch_tokens}


@MODELS.register_module()
class DINOv2Backbone(BaseModule):
    """DINOv2 ViT encoder registered in MODELS registry.

    Loads a DINOv2 model via timm with pretrained weights. Accepts both
    torch.hub style names (e.g. ``'dinov2_vitb14'``) and timm model names
    (e.g. ``'vit_base_patch14_dinov2.lvd142m'``).

    Args:
        model_name (str): DINOv2 variant. Default ``'dinov2_vitb14'``.
        frozen (bool): Freeze all parameters. Default ``True``.
        pretrained (bool): Load pretrained weights. Default ``True``.
    """

    def __init__(self, model_name='dinov2_vitb14', frozen=True,
                 pretrained=True, init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        encoder = self._build_local_hub_model(model_name, pretrained)
        if encoder is None:
            timm_name = DINOV2_HUB_TO_TIMM.get(model_name, model_name)
            logger.info(f"Loading DINOv2 backbone via timm: {timm_name}")
            timm_model = timm.create_model(
                timm_name, pretrained=pretrained, dynamic_img_size=True,
            )
            encoder = _DINOv2Encoder(timm_model)
        self.encoder = encoder

        if frozen:
            self.eval()
            for p in self.parameters():
                p.requires_grad = False

    def _build_local_hub_model(self, model_name, pretrained):
        """Try the official local torch.hub DINOv2 path before timm."""
        this_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(this_dir)))
        candidate_roots = [
            os.path.join(project_root, '.refs', 'UniVAD_ref', 'models', 'dinov2'),
            os.path.join(project_root, '.refs', 'UniVAD', 'models', 'dinov2'),
        ]
        for root in candidate_roots:
            if os.path.isdir(root):
                logger.info(f'Loading DINOv2 backbone via local torch.hub path: {root}')
                return torch.hub.load(root, model_name, pretrained=pretrained, source='local')
        return None

    def forward(self, x):
        return self.encoder(x)

    def train(self, mode=True):
        if mode and not any(p.requires_grad for p in self.parameters()):
            return super().train(False)
        return super().train(mode)
