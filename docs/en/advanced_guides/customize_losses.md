# Customizing Losses

BaoIAD provides a set of registered loss modules that can be used in detector `loss()` methods.

## Built-in Losses

All losses are in `baoiad/models/losses/` and registered with the `MODELS` registry.

### CosineLoss

Cosine distance between feature tensors, commonly used in knowledge distillation methods (RD, STFPM, EfficientAD).

```python
loss = dict(
    type='CosineLoss',
    reduction='mean',
)
```

### MSELoss

Mean squared error loss for feature matching and reconstruction.

```python
loss = dict(
    type='MSELoss',
    reduction='mean',
)
```

### L1Loss

L1 (mean absolute error) loss.

```python
loss = dict(
    type='L1Loss',
    reduction='mean',
)
```

### BCELoss

Binary cross-entropy loss for classification-based methods.

```python
loss = dict(
    type='BCELoss',
    reduction='mean',
)
```

### CrossEntropyLoss

Cross-entropy loss with optional label smoothing.

```python
loss = dict(
    type='CrossEntropyLoss',
    reduction='mean',
    label_smoothing=0.0,
)
```

### FocalLoss

Focal loss for handling class imbalance (used by DRAEM).

```python
loss = dict(
    type='FocalLoss',
    gamma=2.0,
    alpha=0.25,
    reduction='mean',
)
```

:::{warning}
DRAEM's FocalLoss expects softmax probabilities, not raw logits. Ensure the input is properly normalized.
:::

### DiceLoss

Dice loss for segmentation (used by DSR, DeSTSeg).

```python
loss = dict(
    type='DiceLoss',
    smooth=1.0,
    reduction='mean',
)
```

### SSIMLoss

Structural Similarity Index loss for reconstruction quality (used by DRAEM, DSR).

```python
loss = dict(
    type='SSIMLoss',
    window_size=11,
    reduction='mean',
)
```

## Using Losses in Detectors

### Direct Instantiation

```python
from baoiad.registry import MODELS

cosine_loss = MODELS.build(dict(type='CosineLoss'))
loss_value = cosine_loss(teacher_feats, student_feats)
```

### In Config

Define losses in the model config and build them in `__init__`:

```python
# Config
model = dict(
    type='MyDetector',
    loss=dict(type='CosineLoss'),
)

# Detector code
class MyDetector(BaseADModel):
    def __init__(self, loss, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = MODELS.build(loss)

    def loss(self, batch_inputs, data_samples):
        feats = self.extract_feat(batch_inputs)
        loss_val = self.loss_fn(feats[0], feats[1])
        return {'loss': loss_val}
```

## Adding a Custom Loss

1. Create `baoiad/models/losses/my_loss.py`:

```python
import torch
import torch.nn as nn
from baoiad.registry import MODELS


@MODELS.register_module()
class MyLoss(nn.Module):
    """Custom loss function.

    Args:
        alpha: Weight parameter.
        reduction: Reduction method ('mean', 'sum', 'none').
    """

    def __init__(self, alpha: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, target):
        """Compute loss.

        Args:
            pred: Predicted tensor.
            target: Target tensor.

        Returns:
            Loss value.
        """
        diff = pred - target
        loss = self.alpha * diff.abs().pow(2)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
```

2. Register in `baoiad/models/losses/__init__.py`:

```python
from .my_loss import MyLoss
```

3. Use in config:

```python
model = dict(
    type='MyDetector',
    loss=dict(type='MyLoss', alpha=0.5),
)
```

## Loss Combinations

Many methods combine multiple losses:

```python
def loss(self, batch_inputs, data_samples):
    # Feature matching loss
    feat_loss = self.cosine_loss(teacher_feats, student_feats)

    # Reconstruction loss
    recon_loss = self.mse_loss(reconstructed, original)

    # Combined with weights
    total_loss = feat_loss + 0.1 * recon_loss

    return {'loss': total_loss, 'feat_loss': feat_loss, 'recon_loss': recon_loss}
```

All returned loss tensors are logged automatically by MMEngine's `LoggerHook`.