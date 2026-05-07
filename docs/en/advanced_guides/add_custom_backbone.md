# Add a Custom Backbone

Backbones in BaoIAD extract multi-scale features from input images. Most methods reuse existing wrappers (`TIMMBackbone`, `OpenCLIPBackbone`, `DINOv2Backbone`), but you can add a custom backbone when you need a model not available through timm or OpenCLIP.

## Backbone Interface

A backbone must:

1. Be registered in the `MODELS` registry
2. Inherit from `mmengine.model.BaseModule`
3. Implement `forward(x)` returning a tuple of feature tensors (one per scale)

Each output tensor has shape `(B, C_i, H_i, W_i)` where `i` indexes the feature level.

## Available Backbones

| Backbone class | Source | Registered name |
|---|---|---|
| `TIMMBackbone` | timm models | `TIMMBackbone` |
| `OpenCLIPBackbone` | OpenCLIP models | `OpenCLIPBackbone` |
| `DINOv2Backbone` | DINOv2 models | `DINOv2Backbone` |
| `DinomalyEncoder` | Dinomaly ViT | `DinomalyEncoder` |
| `ViTEncoderBackbone` | DeiT-style ViT | `ViTEncoderBackbone` |
| `FeatureExtractor` | CNN feature extraction | `FeatureExtractor` |
| `RawBackbone` | Raw nn.Module wrapper | `RawBackbone` |

## Example: Wrapping a Custom Model

Create `baoiad/models/backbones/my_backbone.py`:

```python
"""Custom backbone wrapping a hypothetical model."""

from typing import Sequence

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from baoiad.registry import MODELS


@MODELS.register_module()
class MyCustomBackbone(BaseModule):
    """A custom backbone that extracts multi-scale features.

    Args:
        model_name: Name of the model architecture.
        pretrained: Whether to load pretrained weights.
        out_indices: Feature level indices to return (0-based).
        frozen: Whether to freeze all parameters.
        init_cfg: Initialization config.
    """

    def __init__(
        self,
        model_name: str = 'resnet18',
        pretrained: bool = True,
        out_indices: Sequence[int] = (1, 2, 3),
        frozen: bool = True,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.out_indices = tuple(out_indices)

        # Build the internal model
        self.net = self._build_model(model_name, pretrained)

        # Expose output channel info (used by some heads/necks)
        self.out_channels = self._get_out_channels()

        if frozen:
            self.eval()
            for p in self.parameters():
                p.requires_grad = False

    def _build_model(self, model_name: str, pretrained: bool) -> nn.Module:
        # Replace with your actual model construction logic
        import torchvision.models as models
        model_fn = getattr(models, model_name)
        model = model_fn(pretrained=pretrained)
        return model

    def _get_out_channels(self) -> tuple:
        # Return channel counts for each output index
        # Example for ResNet: layer1=64, layer2=128, layer3=256, layer4=512
        channel_map = {0: 64, 1: 128, 2: 256, 3: 512}
        return tuple(channel_map[i] for i in self.out_indices)

    def forward(self, x: torch.Tensor) -> tuple:
        """Extract multi-scale features.

        Args:
            x: Input tensor (B, 3, H, W).

        Returns:
            Tuple of feature tensors, one per out_indices level.
        """
        feats = []
        # Example: walk through ResNet layers
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        for i, layer_name in enumerate(['layer1', 'layer2', 'layer3', 'layer4']):
            layer = getattr(self.net, layer_name)
            x = layer(x)
            if i in self.out_indices:
                feats.append(x)

        return tuple(feats)

    def train(self, mode: bool = True):
        """Keep frozen parameters in eval mode."""
        result = super().train(mode)
        if mode and not any(p.requires_grad for p in self.parameters()):
            return super().train(False)
        return result
```

## Register the Backbone

Add to `baoiad/models/backbones/__init__.py`:

```python
from .my_backbone import MyCustomBackbone  # noqa: F401
```

## Using the Backbone in a Config

Reference your backbone by its registered class name:

```python
model = dict(
    type='MyDetector',
    backbone=dict(
        type='MyCustomBackbone',
        model_name='resnet18',
        pretrained=True,
        out_indices=(1, 2, 3),
        frozen=True,
    ),
    freeze_backbone=True,
)
```

## Using TIMMBackbone for timm Models

For most CNN and ViT models available in the [timm](https://github.com/huggingface/pytorch-image-models) library, use the built-in `TIMMBackbone` wrapper:

```python
backbone=dict(
    type='TIMMBackbone',
    model_name='wide_resnet50_2',   # any timm model name
    pretrained=True,
    features_only=True,             # return intermediate features
    out_indices=(2, 3),             # which feature levels
    frozen=True,                    # freeze all parameters
)
```

`TIMMBackbone` automatically resolves cached pretrained weights from `~/.cache/torch/hub/checkpoints/` and falls back to Hugging Face Hub cache. It exposes `out_channels` and `reduction` from timm's `feature_info`.

### Partial Freezing

To freeze only specific submodules while leaving others trainable:

```python
backbone=dict(
    type='TIMMBackbone',
    model_name='wide_resnet50_2',
    pretrained=True,
    features_only=True,
    out_indices=(2, 3),
    frozen=False,
    frozen_names=('layer1', 'layer2', 'layer3'),  # freeze these only
    frozen_names_eval=True,  # keep frozen layers in eval mode during training
)
```

## Feature Extraction in Models

When a backbone is used inside a `BaseADModel` subclass, feature extraction goes through `extract_feat()`:

```python
# From BaseADModel.extract_feat():
def extract_feat(self, batch_inputs):
    ctx = torch.no_grad() if self.freeze_backbone else torch.enable_grad()
    with ctx:
        feats = self.backbone(batch_inputs)
    if isinstance(feats, Tensor):
        feats = (feats,)
    if self.neck is not None:
        feats = self.neck(feats)
    return feats
```

The backbone output is always normalized to a tuple of tensors. If a neck (e.g., `MultiScalePooling`) is configured, it processes the tuple further.

## Key Conventions

- **Frozen backbone**: Most AD methods freeze the backbone. Set `frozen=True` and override `train()` to keep it in eval mode during training.
- **Multi-scale output**: Return a tuple of tensors, one per `out_indices` level. This is what necks and heads expect.
- **`out_channels` attribute**: Expose channel counts so heads/necks can determine their input dimensions automatically.
