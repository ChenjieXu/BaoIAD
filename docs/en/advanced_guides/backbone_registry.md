# Backbone Registry

BaoIAD supports multiple backbone types, each registered with the `MODELS` registry. This page documents the available backbones and how to add new ones.

## Available Backbones

### TIMMBackbone

The most commonly used backbone. Wraps `timm.create_model(features_only=True)` to extract multi-scale features.

```python
backbone=dict(
    type='TIMMBackbone',
    model_name='wide_resnet50_2',  # Any timm model name
    pretrained=True,
    features_only=True,
    out_indices=(2, 3),            # Feature layer indices
    frozen=True,                    # Freeze backbone weights
)
```

| Parameter | Description |
|-----------|-------------|
| `model_name` | Any model name from [timm](https://huggingface.co/docs/timm/) |
| `pretrained` | Load ImageNet pretrained weights |
| `features_only` | Must be `True` for feature extraction |
| `out_indices` | Feature layers to extract (varies by method) |
| `frozen` | Freeze all backbone parameters |

Common model names and their `out_indices`:

| Model | Typical `out_indices` | Used By |
|-------|----------------------|---------|
| `wide_resnet50_2` | `(2, 3)` | PatchCore, RD, SPADE, SimpleNet |
| `wide_resnet50_2` | `(1, 2, 3)` | PaDiM, CFA, STFPM |
| `resnet18` | `(1, 2, 3)` | FastFlow, STFPM |
| `efficientnet_b4` | `(1, 2, 3, 4)` | UniAD, MambaAD |

### OpenCLIPBackbone

Wraps OpenCLIP models for vision-language methods.

```python
backbone=dict(
    type='OpenCLIPBackbone',
    model_name='ViT-L-14-336',
    pretrained='openai',
    out_indices=(1, 2, 3, ..., 24),  # Transformer layer indices
    frozen=True,
)
```

Used by: WinCLIP, AnomalyCLIP, AnoVL, MuSc, AdaCLIP, AACLIP.

### DINOv2Backbone

Wraps DINOv2 ViT models for self-supervised feature extraction.

```python
backbone=dict(
    type='DINOv2Backbone',
    model_name='dinov2_vitb14_reg',
    frozen=True,
)
```

Used by: Dinomaly.

### ViTEncoderBackbone / DistilledVisionTransformerBackbone

Specialized ViT backbones for ViTAD.

```python
backbone=dict(
    type='DistilledVisionTransformerBackbone',
    img_size=256,
    patch_size=16,
    embed_dim=384,
    depth=12,
    num_heads=6,
)
```

Used by: ViTAD.

### EfficientNetFeatureExtractor / GenericFeatureExtractor

Feature extractors used by AST and related methods.

```python
backbone=dict(
    type='EfficientNetFeatureExtractor',
    model_name='efficientnet_b4',
    out_indices=(2, 3),
    frozen=True,
)
```

Used by: AST.

### CSFlowFeatureExtractor

Specialized feature extractor for CSFlow that operates on cross-scale features.

Used by: CSFlow.

### MuScCLIPBackbone

Multi-layer CLIP backbone with positional embedding resize for MuSc.

Used by: MuSc.

### RawBackbone

Legacy passthrough backbone. Prefer `TIMMBackbone` for new implementations.

## Shared Backbone Configs

Common backbone configurations are defined in `configs/_base_/backbones/` for reuse:

```python
# configs/_base_/backbones/wide_resnet50_unified.py
backbone=dict(
    type='TIMMBackbone',
    model_name='wide_resnet50_2',
    pretrained=True,
    features_only=True,
    out_indices=(2, 3),
    frozen=True,
)
```

Inherit these in method configs to ensure consistency:

```python
_base_ = ['../_base_/backbones/wide_resnet50_unified.py']
```

## Adding a New Backbone

1. Create `baoiad/models/backbones/my_backbone.py`:

```python
import torch.nn as nn
from baoiad.registry import MODELS


@MODELS.register_module()
class MyBackbone(nn.Module):
    """Custom backbone for feature extraction."""

    def __init__(self, out_indices=(2, 3), frozen=True):
        super().__init__()
        self.out_indices = out_indices
        self.frozen = frozen
        # Define layers ...

    def forward(self, x):
        """Extract multi-scale features.

        Returns:
            tuple[Tensor]: Multi-scale feature maps.
        """
        feats = []
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx in self.out_indices:
                feats.append(x)
        return tuple(feats)
```

2. Register in `baoiad/models/backbones/__init__.py`.

3. Create a shared config in `configs/_base_/backbones/my_backbone.py`.

## Feature Layer Indices

The `out_indices` parameter controls which feature layers are extracted. Different methods require different layers:

```{important}
`out_indices` uses 0-based indexing. For ResNet-style networks:
- `(0,)` = layer1 (stride 4)
- `(1,)` = layer2 (stride 8)
- `(2,)` = layer3 (stride 16)
- `(3,)` = layer4 (stride 32)

Different implementations may use different conventions. Always verify by checking the feature map sizes.
```