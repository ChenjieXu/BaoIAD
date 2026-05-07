# Add a Custom Loss Function

Losses in BaoIAD are registered as `MODELS` (following the mmdetection convention). They are `BaseModule` subclasses that compute a scalar loss tensor from predictions and targets.

## Built-in Losses

BaoIAD ships with these loss modules in `baoiad/models/losses/`:

| Module | Class name | Description |
|---|---|---|
| `mse_loss.py` | `MSELoss` | Mean squared error |
| `l1_loss.py` | `L1Loss`, `SmoothL1Loss` | L1 and smooth L1 |
| `bce_loss.py` | `BCEWithLogitsLoss`, `BCELoss` | Binary cross-entropy |
| `focal_loss.py` | `BinaryFocalLoss`, `FocalLoss` | Focal loss (binary and multiclass) |
| `cosine_loss.py` | `CosineDistanceLoss` | Cosine distance |
| `ssim_loss.py` | `SSIMLoss` | Structural similarity |
| `dice_loss.py` | `BinaryDiceLoss` | Dice loss for segmentation |
| `cross_entropy_loss.py` | `CrossEntropyLoss` | Cross-entropy |

## Pattern

All loss modules follow the same pattern:

1. Inherit from `mmengine.model.BaseModule`
2. Register with `@MODELS.register_module()`
3. Accept a `loss_weight` parameter
4. Implement `forward(pred, target, **kwargs) -> Tensor`

## Example: Contrastive Loss

Create `baoiad/models/losses/contrastive_loss.py`:

```python
"""Contrastive loss for anomaly detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from baoiad.registry import MODELS


@MODELS.register_module()
class ContrastiveLoss(BaseModule):
    """Margin-based contrastive loss.

    Pulls positive (normal) features together and pushes negative
    (anomalous) features apart by at least ``margin``.

    Args:
        margin: Minimum distance between positive and negative pairs.
        reduction: 'mean', 'sum', or 'none'.
        loss_weight: Global weight applied to the loss.
    """

    def __init__(
        self,
        margin: float = 1.0,
        reduction: str = 'mean',
        loss_weight: float = 1.0,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.margin = margin
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            pred: Feature embeddings (N, D).
            target: Binary labels (N,), where 0=normal, 1=anomaly.

        Returns:
            Scalar loss tensor.
        """
        # Pairwise squared distances
        diff = pred.unsqueeze(1) - pred.unsqueeze(0)  # (N, N, D)
        dist_sq = (diff ** 2).sum(dim=-1)  # (N, N)

        # Same-class pairs (positive): minimize distance
        # Different-class pairs (negative): maximize distance up to margin
        label_eq = (target.unsqueeze(1) == target.unsqueeze(0))  # (N, N)

        # Positive loss: mean squared distance for same-class pairs
        pos_mask = label_eq & ~torch.eye(
            len(target), dtype=torch.bool, device=target.device)
        pos_loss = (
            dist_sq[pos_mask].mean() if pos_mask.any()
            else torch.tensor(0.0, device=pred.device)
        )

        # Negative loss: hinge loss for different-class pairs
        neg_mask = ~label_eq
        neg_dist = torch.sqrt(dist_sq[neg_mask] + 1e-8)
        neg_loss = (
            F.relu(self.margin - neg_dist).mean() if neg_mask.any()
            else torch.tensor(0.0, device=pred.device)
        )

        loss = pos_loss + neg_loss

        if self.reduction == 'sum':
            return self.loss_weight * loss * pred.shape[0]
        elif self.reduction == 'none':
            return self.loss_weight * loss
        return self.loss_weight * loss
```

## Register the Loss

Add to `baoiad/models/losses/__init__.py`:

```python
from baoiad.models.losses.contrastive_loss import ContrastiveLoss  # noqa: F401
```

And add `'ContrastiveLoss'` to the `__all__` list.

## Using the Loss in a Model

Loss modules are typically constructed directly inside a model's `__init__` and called during `forward(mode='loss')`:

```python
from baoiad.registry import MODELS
from baoiad.models.base_ad_model import BaseADModel

@MODELS.register_module()
class MyModel(BaseADModel):
    def __init__(self, backbone, loss_cfg=None, **kwargs):
        super().__init__(backbone=backbone, **kwargs)
        loss_cfg = loss_cfg or dict(type='MSELoss')
        self.loss_fn = MODELS.build(loss_cfg)

    def forward(self, inputs, data_samples=None, mode='tensor'):
        feats = self.extract_feat(inputs)
        if mode == 'loss':
            reconstructed = self.decoder(feats)
            target = self._get_targets(feats)
            loss = self.loss_fn(reconstructed, target)
            return {'loss': loss}
```

Or configure it in the config file and have the model build it:

```python
# configs/mydetector/mydetector_256_mvtec.py
model = dict(
    type='MyModel',
    backbone=dict(type='TIMMBackbone', ...),
    loss_cfg=dict(
        type='ContrastiveLoss',
        margin=1.0,
        loss_weight=0.5,
    ),
)
```

## Example: Weighted Multi-Loss

For methods that combine multiple losses, build each from config:

```python
@MODELS.register_module()
class MultiLossModel(BaseADModel):
    def __init__(self, backbone, losses=None, **kwargs):
        super().__init__(backbone=backbone, **kwargs)
        losses = losses or [dict(type='MSELoss')]
        self.loss_fns = nn.ModuleList([MODELS.build(l) for l in losses])

    def _compute_losses(self, pred, target):
        total = 0.0
        loss_dict = {}
        for i, fn in enumerate(self.loss_fns):
            l = fn(pred, target)
            total = total + l
            loss_dict[f'loss_{i}'] = l
        loss_dict['loss'] = total
        return loss_dict
```

Config:

```python
model = dict(
    type='MultiLossModel',
    backbone=dict(type='TIMMBackbone', ...),
    losses=[
        dict(type='MSELoss', loss_weight=1.0),
        dict(type='SSIMLoss', loss_weight=0.5),
        dict(type='BinaryDiceLoss', loss_weight=0.3),
    ],
)
```
