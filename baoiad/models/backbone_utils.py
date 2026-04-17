"""Utility for building backbones.

Provides :func:`normalize_backbone_cfg` to convert legacy string backbone
names into config dicts, and the legacy :func:`build_backbone` /
:func:`get_channel_dims` helpers for any code not yet migrated.
"""
from __future__ import annotations

import copy
from typing import Union

import torch.nn as nn
from torchvision import models as tv_models

from baoiad.registry import MODELS


# ---------------------------------------------------------------------------
# Legacy registry (kept for backward compatibility of build_backbone)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# New unified helpers
# ---------------------------------------------------------------------------

def normalize_backbone_cfg(
    backbone: Union[str, dict],
    default_out_indices: tuple = (0, 1, 2, 3),
    default_frozen: bool = True,
) -> dict:
    """Normalize a backbone specification to a config dict.

    If *backbone* is already a dict it is returned as-is (deep-copied).
    If it is a string (legacy), it is converted to a ``FeatureExtractor``
    config dict.

    Args:
        backbone: Either a model name string or a config dict with
            ``type`` key.
        default_out_indices: Default ``out_indices`` when converting from
            string.
        default_frozen: Default ``frozen`` flag when converting from string.

    Returns:
        A config dict suitable for ``MODELS.build()``.
    """
    if isinstance(backbone, dict):
        cfg = copy.deepcopy(backbone)
        cfg.pop('_delete_', None)  # strip MMEngine config-merge marker
        return cfg
    # Legacy string name
    canonical = backbone.split('.')[0] if '.' in backbone else backbone
    return dict(
        type='FeatureExtractor',
        backbone_name=canonical,
        pretrained=True,
        out_indices=default_out_indices,
        frozen=default_frozen,
    )


def build_feature_extractor(
    backbone: Union[str, dict],
    default_out_indices: tuple = (0, 1, 2, 3),
    default_frozen: bool = True,
) -> nn.Module:
    """Build a feature-extractor backbone from a string or config dict.

    Returns:
        An ``nn.Module`` with ``.out_channels`` attribute (tuple of ints).
    """
    cfg = normalize_backbone_cfg(backbone, default_out_indices, default_frozen)
    return MODELS.build(cfg)


# ---------------------------------------------------------------------------
# Legacy helpers (deprecated, kept for backward compat)
# ---------------------------------------------------------------------------

def build_backbone(name: str, pretrained: bool = True):
    """Build a torchvision backbone by name (DEPRECATED).

    Prefer :func:`build_feature_extractor` for new code.
    """
    canonical = name.split('.')[0] if '.' in name else name
    if canonical in _BACKBONE_REGISTRY and name not in _BACKBONE_REGISTRY:
        name = canonical
    if name not in _BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone '{name}'. Supported: {list(_BACKBONE_REGISTRY.keys())}"
        )
    constructor, weights = _BACKBONE_REGISTRY[name]
    return constructor(weights=weights if pretrained else None)


def get_channel_dims(name: str) -> tuple:
    """Get per-layer channel dimensions for a backbone (DEPRECATED).

    Prefer using ``backbone.out_channels`` from a built FeatureExtractor.
    """
    canonical = name.split('.')[0] if '.' in name else name
    if canonical in _CHANNEL_DIMS and name not in _CHANNEL_DIMS:
        name = canonical
    if name not in _CHANNEL_DIMS:
        raise ValueError(
            f"Unknown backbone '{name}'. Supported: {list(_CHANNEL_DIMS.keys())}"
        )
    return _CHANNEL_DIMS[name]


# Re-exports for convenience — avoid direct torchvision imports in detectors
from torchvision.models.resnet import Bottleneck as ResNetBottleneck  # noqa: F401
