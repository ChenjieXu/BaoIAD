# Backbone Registry

BaoIAD provides several backbone wrappers registered in the `MODELS` registry (from `baoiad/registry.py`). Each backbone can be configured via dict in model configs and built with `MODELS.build(cfg)`.

## Available Backbones

| Registry Name | Module | Description |
|---------------|--------|-------------|
| `TIMMBackbone` | `baoiad.models.backbones.timm_backbone` | Generic timm model wrapper |
| `FeatureExtractor` | `baoiad.models.backbones.feature_extractor` | Multi-scale torchvision feature extractor |
| `OpenCLIPBackbone` | `baoiad.models.backbones.clip_backbone` | OpenCLIP model wrapper |
| `DINOv2Backbone` | `baoiad.models.backbones.dinov2_backbone` | DINOv2 ViT encoder |
| `DinomalyEncoder` | `baoiad.models.backbones.dinomaly_backbone` | DINOv2 encoder for Dinomaly |
| `RawBackbone` | `baoiad.models.backbones.raw_backbone` | Raw torchvision backbone with exposed layers |
| `DistilledVisionTransformerBackbone` | `baoiad.models.backbones.vitad_backbone` | DeiT distilled ViT encoder |
| `ViTEncoderBackbone` | `baoiad.models.backbones.vitad_backbone` | Standard ViT encoder |
| `MuScCLIPBackbone` | `baoiad.models.backbones.musc_clip_backbone` | MuSc-specific CLIP backbone |
| `MuScDINOv2Backbone` | `baoiad.models.backbones.musc_clip_backbone` | MuSc-specific DINOv2 backbone |
| `EfficientNetFeatureExtractor` | `baoiad.models.backbones.ast_backbone` | EfficientNet multi-scale feature extractor |
| `EfficientNetLayerExtractor` | `baoiad.models.backbones.ast_backbone` | EfficientNet single-layer extractor |
| `GenericFeatureExtractor` | `baoiad.models.backbones.ast_backbone` | Generic multi-scale extractor |
| `SAASaliencyBackbone` | `baoiad.models.backbones.saa_saliency_backbone` | SAA saliency backbone |

## TIMMBackbone

The most commonly used backbone. Wraps any [timm](https://github.com/huggingface/pytorch-image-models) model with multi-scale feature extraction.

```python
backbone = dict(
    type='TIMMBackbone',
    model_name='wide_resnet50_2',   # Any timm model name
    pretrained=True,
    features_only=True,             # Return intermediate features
    out_indices=(1, 2, 3),          # Feature levels to extract
    frozen=True,                    # Freeze all parameters
)
```

### Supported Architectures

Common `model_name` values used across BaoIAD methods:

| Model | Used by | Feature channels (levels 1-3) |
|-------|---------|-------------------------------|
| `wide_resnet50_2` | PatchCore, PaDiM, SimpleNet, RD | (256, 512, 1024) |
| `resnet18` | EfficientAD | (64, 128, 256) |
| `convnext_base` | CFA | varies |
| `tf_efficientnet_b4` | DRAEM | varies |
| `wide_resnet50_2.tv2_in1k` | Alternative WRN-50 | (256, 512, 1024) |

### Selective Freezing

Use `frozen_names` to freeze specific submodules while keeping others trainable:

```python
backbone = dict(
    type='TIMMBackbone',
    model_name='wide_resnet50_2',
    pretrained=True,
    features_only=True,
    out_indices=(1, 2, 3),
    frozen=False,                    # Don't freeze everything
    frozen_names=['layer1', 'layer2'],  # Freeze only these layers
    frozen_names_eval=True,          # Keep frozen layers in eval mode during training
)
```

### Caching and Offline Use

`TIMMBackbone` resolves pretrained weights from local caches before attempting download:

1. Checks `torch.hub.get_dir()/checkpoints/` for known filenames
2. Checks HuggingFace cache for safetensors
3. Falls back to timm's online download

Set `HF_HUB_OFFLINE=1` to enforce offline-only mode.

## FeatureExtractor

Multi-scale feature extractor built from torchvision ResNet-family models. Returns intermediate feature maps from specified layers.

```python
backbone = dict(
    type='FeatureExtractor',
    backbone_name='wide_resnet50_2',
    pretrained=True,
    out_indices=(1, 2, 3),   # 1→layer1, 2→layer2, 3→layer3, 4→layer4
    frozen=True,
)
```

**Supported `backbone_name` values**: `resnet18`, `resnet34`, `resnet50`, `resnet101`, `wide_resnet50_2`, `wide_resnet101_2`.

**Channel dimensions** (for `out_indices`):

| Backbone | layer1 | layer2 | layer3 | layer4 |
|----------|--------|--------|--------|--------|
| ResNet-18/34 | 64 | 128 | 256 | 512 |
| ResNet-50/101 | 256 | 512 | 1024 | 2048 |
| Wide-ResNet-50/101 | 256 | 512 | 1024 | 2048 |

## OpenCLIPBackbone

Wraps OpenCLIP models for CLIP-based anomaly detection methods.

```python
backbone = dict(
    type='OpenCLIPBackbone',
    model_name='ViT-L-14-336',   # OpenCLIP model name
    pretrained='openai',          # Pretrained weights source
    frozen=True,
    force_quick_gelu=True,        # Required for OpenAI models
)
```

**Common model names**: `ViT-L-14-336`, `ViT-L-14`, `ViT-B-16`, `ViT-B-16-plus-240`.

Key properties:
- `.visual` — Access the visual encoder directly
- `.tokenize(texts)` — Tokenize text inputs
- `.encode_image(x)` — Encode images to CLIP features
- `.encode_text(t)` — Encode text tokens to CLIP features

## DINOv2Backbone

DINOv2 ViT encoder loaded via timm (or local torch.hub).

```python
backbone = dict(
    type='DINOv2Backbone',
    model_name='dinov2_vitb14',   # DINOv2 variant
    frozen=True,
    pretrained=True,
)
```

**Supported model names**:

| Name | Architecture | Embed dim | Depth |
|------|-------------|-----------|-------|
| `dinov2_vits14` | ViT-S/14 | 384 | 12 |
| `dinov2_vitb14` | ViT-B/14 | 768 | 12 |
| `dinov2_vitl14` | ViT-L/14 | 1024 | 24 |
| `dinov2_vitg14` | ViT-g/14 | 1536 | 40 |
| `dinov2_vits14_reg` | ViT-S/14 + registers | 384 | 12 |
| `dinov2_vitb14_reg` | ViT-B/14 + registers | 768 | 12 |
| `dinov2_vitl14_reg` | ViT-L/14 + registers | 1024 | 24 |
| `dinov2_vitg14_reg` | ViT-g/14 + registers | 1536 | 40 |

## RawBackbone

Exposes individual layers of a torchvision ResNet for methods that need layer-level access.

```python
backbone = dict(
    type='RawBackbone',
    backbone_name='wide_resnet50_2',
    pretrained=True,
    frozen=True,
)
```

Exposes `conv1`, `bn1`, `relu`, `maxpool`, `layer1`–`layer4` as direct attributes. Used by methods like PyramidFlow and SuperSimpleNet that decompose the backbone.

## DistilledVisionTransformerBackbone and ViTEncoderBackbone

ViT encoders for methods like ViTAD that need intermediate block features.

```python
backbone = dict(
    type='DistilledVisionTransformerBackbone',
    teachers=(3, 6, 9),       # Block indices for teacher features
    neck=(12,),                # Block indices for neck features
    img_size=256,
    patch_size=16,
    embed_dim=384,
    depth=12,
    num_heads=6,
    pretrained_url='https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth',
    pretrained=True,
)
```

Or use the convenience function:

```python
from baoiad.models.backbones.vitad_backbone import get_vitad_encoder_config

backbone = get_vitad_encoder_config('deit_small_distilled_patch16_224', img_size=256)
```

## Configuring Backbones in a Model

Backbone selection is configured in the model's `backbone` dict. Here is a complete example:

```python
model = dict(
    type='PatchCore',
    backbone=dict(
        type='TIMMBackbone',
        model_name='wide_resnet50_2',
        pretrained=True,
        features_only=True,
        out_indices=(1, 2, 3),
        frozen=True,
    ),
    neck=dict(type='MultiScalePooling', output_size=1),
    head=dict(type='MemoryBankHead', coreset_ratio=0.1),
)
```

To switch backbones, simply change the `type` and parameters:

```python
# Switch from WRN-50 to ResNet-18
model = dict(
    type='PatchCore',
    backbone=dict(
        type='FeatureExtractor',
        backbone_name='resnet18',
        out_indices=(1, 2, 3),
        frozen=True,
    ),
    neck=dict(type='MultiScalePooling', output_size=1),
    head=dict(type='MemoryBankHead', coreset_ratio=0.1),
)
```
