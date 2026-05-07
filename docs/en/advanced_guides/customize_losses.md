# Customizing Losses

BaoIAD provides a set of registered loss modules that can be used directly in model configs or extended for custom methods.

## Built-In Losses

All losses are registered in the `MODELS` registry (from `baoiad/registry.py`) and can be built via `MODELS.build(cfg)`.

| Registry Name | Module | Description |
|---------------|--------|-------------|
| `MSELoss` | `baoiad.models.losses.mse_loss` | Mean Squared Error |
| `L1Loss` | `baoiad.models.losses.l1_loss` | L1 (Mean Absolute Error) |
| `SmoothL1Loss` | `baoiad.models.losses.l1_loss` | Smooth L1 (Huber-like) |
| `BCEWithLogitsLoss` | `baoiad.models.losses.bce_loss` | Binary Cross-Entropy with logits |
| `BCELoss` | `baoiad.models.losses.bce_loss` | Binary Cross-Entropy (sigmoid inputs) |
| `CrossEntropyLoss` | `baoiad.models.losses.cross_entropy_loss` | Cross-Entropy for classification |
| `FocalLoss` | `baoiad.models.losses.focal_loss` | Multiclass Focal Loss |
| `BinaryFocalLoss` | `baoiad.models.losses.focal_loss` | Binary Focal Loss |
| `CosineDistanceLoss` | `baoiad.models.losses.cosine_loss` | 1 − cosine_similarity |
| `SSIMLoss` | `baoiad.models.losses.ssim_loss` | 1 − SSIM for reconstruction |
| `BinaryDiceLoss` | `baoiad.models.losses.dice_loss` | Dice loss for segmentation |

## Common API

All loss modules follow the same interface:

```python
class SomeLoss(BaseModule):
    def __init__(self, reduction='mean', loss_weight=1.0):
        ...

    def forward(self, pred, target, **kwargs) -> Tensor:
        return self.loss_weight * base_loss(pred, target, reduction=self.reduction)
```

**Common arguments:**

| Arg | Type | Default | Description |
|-----|------|---------|-------------|
| `reduction` | `str` | `'mean'` | `'mean'`, `'sum'`, or `'none'` |
| `loss_weight` | `float` | `1.0` | Scalar multiplier on the loss |

## Loss Details

### MSELoss

```python
dict(type='MSELoss', reduction='mean', loss_weight=1.0)
```

Standard MSE: `loss_weight * F.mse_loss(pred, target, reduction)`.

Used by: knowledge distillation methods (RD, EfficientAD) for teacher-student feature matching.

### L1Loss and SmoothL1Loss

```python
dict(type='L1Loss', reduction='mean', loss_weight=1.0)
dict(type='SmoothL1Loss', beta=1.0, reduction='mean', loss_weight=1.0)
```

- `L1Loss`: `F.l1_loss(pred, target, reduction)`
- `SmoothL1Loss`: `F.smooth_l1_loss(pred, target, beta, reduction)` — switches between L1 and L2 based on `beta`.

### BCEWithLogitsLoss and BCELoss

```python
dict(type='BCEWithLogitsLoss', reduction='mean', loss_weight=1.0)  # Raw logits
dict(type='BCELoss', reduction='mean', loss_weight=1.0)            # After sigmoid
```

- `BCEWithLogitsLoss`: Applies sigmoid internally. Use when the model outputs raw logits.
- `BCELoss`: Expects sigmoid-activated inputs.

### CrossEntropyLoss

```python
dict(type='CrossEntropyLoss', reduction='mean', loss_weight=1.0)
```

Wraps `F.cross_entropy(pred, target, reduction)`. Target should be class indices.

### FocalLoss and BinaryFocalLoss

```python
dict(type='FocalLoss', alpha=1.0, gamma=2.0, reduction='mean', loss_weight=1.0)
dict(type='BinaryFocalLoss', alpha=0.25, gamma=2.0, reduction='mean', loss_weight=1.0)
```

- `FocalLoss`: Multiclass, cross-entropy based. `alpha` weights the loss, `gamma` down-weights easy examples.
- `BinaryFocalLoss`: Binary, logits-based. Uses `pt = exp(-bce)` formulation.

A standalone function `sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction='none')` is also available.

### CosineDistanceLoss

```python
dict(type='CosineDistanceLoss', reduction='mean', loss_weight=1.0)
```

Computes `1.0 - F.cosine_similarity(pred, target, dim=1)`. Used for feature matching in knowledge distillation methods.

### SSIMLoss

```python
dict(type='SSIMLoss', window_size=11, loss_weight=1.0)
```

Computes `1.0 - SSIM(pred, target)` using a Gaussian window of size `window_size` with `sigma=1.5`. Used by reconstruction methods (DRAEM, MemSeg).

### BinaryDiceLoss

```python
dict(type='BinaryDiceLoss', smooth=1.0)
```

Computes `1 - mean(Dice score)` per sample. Used for anomaly segmentation masks.

## Using Losses in Configs

### Single Loss

```python
model = dict(
    type='MyDetector',
    backbone=dict(...),
    head=dict(
        type='MyHead',
        loss=dict(type='MSELoss', loss_weight=1.0),
    ),
)
```

### Multiple Losses

Many AD methods combine multiple losses:

```python
model = dict(
    type='MyReconstructionDetector',
    backbone=dict(...),
    head=dict(
        type='MyHead',
        loss_recon=dict(type='SSIMLoss', loss_weight=1.0),
        loss_seg=dict(type='BinaryFocalLoss', alpha=0.25, gamma=2.0, loss_weight=0.5),
        loss_l2=dict(type='MSELoss', loss_weight=0.5),
    ),
)
```

In the head's `loss()` method, combine them:

```python
def loss(self, feats, data_samples):
    losses = {}
    if hasattr(self, 'loss_recon'):
        losses['loss_recon'] = self.loss_recon(reconstructed, original)
    if hasattr(self, 'loss_seg'):
        losses['loss_seg'] = self.loss_seg(pred_mask, gt_mask)
    return losses
```

## Building Losses Programmatically

```python
from baoiad.registry import MODELS

# Build from config dict
loss_fn = MODELS.build(dict(type='MSELoss', reduction='mean', loss_weight=0.5))

# Use directly
import torch
pred = torch.randn(4, 256, 14, 14)
target = torch.randn(4, 256, 14, 14)
loss = loss_fn(pred, target)
```

## Creating a Custom Loss

To add a new loss function:

1. Create a new module in `baoiad/models/losses/`:

```python
# baoiad/models/losses/my_loss.py
import torch.nn.functional as F
from mmengine.model import BaseModule
from baoiad.registry import MODELS


@MODELS.register_module(force=True)
class MyCustomLoss(BaseModule):
    """My custom loss function.

    Args:
        margin (float): Margin parameter. Default 1.0.
        reduction (str): 'mean', 'sum', or 'none'.
        loss_weight (float): Weight factor.
    """

    def __init__(self, margin=1.0, reduction='mean', loss_weight=1.0, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.margin = margin
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, **kwargs):
        loss = F.relu(self.margin - (target - pred))
        if self.reduction == 'mean':
            return self.loss_weight * loss.mean()
        elif self.reduction == 'sum':
            return self.loss_weight * loss.sum()
        return self.loss_weight * loss
```

2. Import it in `baoiad/models/losses/__init__.py`:

```python
from baoiad.models.losses.my_loss import MyCustomLoss  # noqa: F401
```

3. Use in configs:

```python
model = dict(
    type='MyDetector',
    head=dict(
        type='MyHead',
        loss=dict(type='MyCustomLoss', margin=2.0, loss_weight=0.5),
    ),
)
```

## Loss Registry

All losses are registered under the `MODELS` registry (not a separate registry). This means the same `MODELS.build()` mechanism used for backbones, necks, and heads also builds loss functions:

```python
from baoiad.registry import MODELS

# List all registered losses
losses = [name for name in MODELS.module_dict if 'Loss' in name or 'loss' in name.lower()]
```
