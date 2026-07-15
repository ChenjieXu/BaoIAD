"""ViTAD encoder backbones: DistilledVisionTransformer and ViTEncoder.

Registered in MODELS registry for config-driven construction.
"""
import os

import numpy as np
import torch
import torch.nn as nn
from functools import partial

from baoiad.registry import MODELS

try:
    import timm
    from timm.models.vision_transformer import VisionTransformer
    from timm.models.layers import trunc_normal_
except ImportError:
    raise ImportError("timm is required for ViTAD backbones. Install with: pip install timm")


# Predefined encoder configs for convenience
_ENCODER_CONFIGS = {
    'deit_small_distilled_patch16_224': {
        'type': 'DistilledVisionTransformerBackbone',
        'patch_size': 16, 'embed_dim': 384, 'depth': 12,
        'num_heads': 6, 'mlp_ratio': 4, 'qkv_bias': True,
        'pretrained_url': 'https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth',
    },
    'deit_tiny_distilled_patch16_224': {
        'type': 'DistilledVisionTransformerBackbone',
        'patch_size': 16, 'embed_dim': 192, 'depth': 12,
        'num_heads': 3, 'mlp_ratio': 4, 'qkv_bias': True,
        'pretrained_url': 'https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth',
    },
    'vit_small_patch16_224_dino': {
        'type': 'ViTEncoderBackbone',
        'timm_model': 'vit_small_patch16_224.dino',
        'patch_size': 16, 'embed_dim': 384, 'depth': 12, 'num_heads': 6,
    },
    'vit_base_patch16_224_dino': {
        'type': 'ViTEncoderBackbone',
        'timm_model': 'vit_base_patch16_224.dino',
        'patch_size': 16, 'embed_dim': 768, 'depth': 12, 'num_heads': 12,
    },
}


@MODELS.register_module(force=True)
class DistilledVisionTransformerBackbone(VisionTransformer):
    """DeiT encoder that extracts intermediate features from specified blocks.

    Args:
        teachers (list[int]): Block indices to extract teacher features from.
        neck (list[int]): Block indices for neck features.
        img_size (int): Input image size. Default 256.
        patch_size (int): Patch size. Default 16.
        embed_dim (int): Embedding dimension. Default 384.
        depth (int): Number of transformer blocks. Default 12.
        num_heads (int): Number of attention heads. Default 6.
        mlp_ratio (float): MLP ratio. Default 4.
        qkv_bias (bool): QKV bias. Default True.
        pretrained_url (str | None): URL for pretrained weights. Default None.
        pretrained (bool): Whether to load pretrained weights. Default True.
    """

    def __init__(self, teachers=(3, 6, 9), neck=(12,), img_size=256,
                 patch_size=16, embed_dim=384, depth=12, num_heads=6,
                 mlp_ratio=4, qkv_bias=True,
                 pretrained_url=None, pretrained=True,
                 **kwargs):
        # Filter out non-VisionTransformer kwargs
        kwargs.pop('data_preprocessor', None)
        kwargs.pop('init_cfg', None)
        super().__init__(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
            depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **kwargs)
        self.teachers = list(teachers)
        self.neck = list(neck)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

        if pretrained and pretrained_url:
            from baoiad.runtime import require_network

            from urllib.parse import urlparse

            filename = os.path.basename(urlparse(pretrained_url).path)
            cached_path = os.path.join(torch.hub.get_dir(), 'checkpoints', filename)
            if not os.path.isfile(cached_path):
                require_network('download ViTAD backbone weights', url=pretrained_url)
            checkpoint = torch.hub.load_state_dict_from_url(
                url=pretrained_url, map_location="cpu", check_hash=True)
            self.load_state_dict(checkpoint["model"], strict=False)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        out, neck = [], []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            fea = x[:, 2:]
            if (idx + 1) in self.neck:
                neck.append(fea)
            if (idx + 1) in self.teachers:
                B, L, C = fea.shape
                H = int(np.sqrt(L))
                fea_2d = fea.view(B, H, H, C).permute(0, 3, 1, 2).contiguous()
                out.append(fea_2d)
        return out, neck


@MODELS.register_module(force=True)
class ViTEncoderBackbone(VisionTransformer):
    """Standard ViT encoder that extracts intermediate features.

    Args:
        teachers (list[int]): Block indices to extract teacher features from.
        neck (list[int]): Block indices for neck features.
        img_size (int): Input image size. Default 256.
        patch_size (int): Patch size. Default 16.
        embed_dim (int): Embedding dimension. Default 384.
        depth (int): Number of transformer blocks. Default 12.
        num_heads (int): Number of attention heads. Default 6.
        timm_model (str | None): timm model name for pretrained weights. Default None.
        pretrained (bool): Whether to load pretrained weights. Default True.
    """

    def __init__(self, teachers=(3, 6, 9), neck=(12,), img_size=256,
                 patch_size=16, embed_dim=384, depth=12, num_heads=6,
                 timm_model=None, pretrained=True,
                 **kwargs):
        kwargs.pop('data_preprocessor', None)
        kwargs.pop('init_cfg', None)

        # Align architecture with timm variant defaults when a model name is given.
        if timm_model:
            # Preserve RNG state so this metadata-only helper model does not
            # perturb the downstream fusion/decoder initialization order.
            with torch.random.fork_rng(devices=[]):
                ref = timm.create_model(timm_model, pretrained=False, img_size=img_size)
            embed_dim = ref.embed_dim
            depth = len(ref.blocks)
            if 'num_classes' not in kwargs:
                kwargs['num_classes'] = getattr(ref, 'num_classes', 1000)
            if len(ref.blocks) > 0:
                num_heads = ref.blocks[0].attn.num_heads
                if 'mlp_ratio' not in kwargs:
                    mlp_in = ref.blocks[0].mlp.fc1.in_features
                    mlp_out = ref.blocks[0].mlp.fc1.out_features
                    kwargs['mlp_ratio'] = float(mlp_out) / float(mlp_in)
                if 'qkv_bias' not in kwargs and hasattr(ref.blocks[0].attn, 'qkv'):
                    kwargs['qkv_bias'] = ref.blocks[0].attn.qkv.bias is not None
            if hasattr(ref.patch_embed, 'patch_size'):
                ps = ref.patch_embed.patch_size
                patch_size = ps[0] if isinstance(ps, tuple) else ps
            if 'norm_layer' not in kwargs and hasattr(ref, 'norm') and hasattr(ref.norm, 'eps'):
                kwargs['norm_layer'] = partial(nn.LayerNorm, eps=ref.norm.eps)
            del ref

        super().__init__(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
            depth=depth, num_heads=num_heads, **kwargs)
        self.teachers = list(teachers)
        self.neck = list(neck)

        if pretrained and timm_model:
            # Use timm variant weights directly, keeping interpolation behavior consistent.
            try:
                with torch.random.fork_rng(devices=[]):
                    ref = timm.create_model(timm_model, pretrained=True, img_size=img_size)
                self.load_state_dict(ref.state_dict(), strict=False)
                del ref
            except Exception:
                # Fallback: load from local HF cache (no internet needed)
                import os
                from safetensors.torch import load_file
                hf_home = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
                cache_dir = os.path.join(hf_home, 'hub',
                                         f'models--timm--{timm_model.replace(".", "-")}')
                # Try both dot and underscore variants of the repo name
                alt_cache_dir = os.path.join(hf_home, 'hub',
                                             f'models--timm--{timm_model}')
                for cdir in [cache_dir, alt_cache_dir]:
                    snapshot_dir = os.path.join(cdir, 'snapshots')
                    if os.path.isdir(snapshot_dir):
                        for snap in os.listdir(snapshot_dir):
                            st_file = os.path.join(snapshot_dir, snap, 'model.safetensors')
                            if os.path.isfile(st_file):
                                sd = load_file(st_file)
                                self.load_state_dict(sd, strict=False)
                                break
                        break

    def forward(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)

        out, neck = [], []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            fea = x[:, 1:, :]
            if (i + 1) in self.neck:
                neck.append(fea)
            if (i + 1) in self.teachers:
                B, L, C = fea.shape
                H = int(np.sqrt(L))
                fea_2d = fea.view(B, H, H, C).permute(0, 3, 1, 2).contiguous()
                out.append(fea_2d)
        return out, neck


def get_vitad_encoder_config(encoder_name, img_size=256, teachers=(3, 6, 9),
                              neck=(12,), pretrained=True):
    """Get a backbone config dict for a named ViTAD encoder.

    Convenience function for backward compatibility with encoder_name strings.
    """
    if encoder_name not in _ENCODER_CONFIGS:
        raise ValueError(f"Unknown encoder: {encoder_name}. "
                         f"Supported: {list(_ENCODER_CONFIGS.keys())}")
    cfg = dict(_ENCODER_CONFIGS[encoder_name])
    cfg['img_size'] = img_size
    cfg['teachers'] = list(teachers)
    cfg['neck'] = list(neck)
    cfg['pretrained'] = pretrained
    return cfg
