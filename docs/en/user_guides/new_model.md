# Adding a New Detector

This guide walks through adding a new anomaly detection method to BaoIAD.

## Step 1: Choose the Base Class

BaoIAD provides 6 specialized sub-classes of `BaseADModel`. Choose the one that matches your method's paradigm:

| Base Class | When to Use | Key Methods to Implement |
|------------|-------------|--------------------------|
| `MemoryBankADModel` | Feature matching (kNN, coreset) | `build_memory_bank()` |
| `KnowledgeDistillationADModel` | Teacher-student discrepancy | `loss()`, `predict()` |
| `FlowBasedADModel` | Normalizing flows on features | `loss()`, `predict()` |
| `ReconstructionADModel` | Autoencoder/reconstruction | `loss()`, `predict()` |
| `VisionLanguageADModel` | CLIP-based zero/few-shot | `loss()`, `predict()` |
| `DiscriminatorADModel` | Feature discrimination | `loss()`, `predict()` |
| `BaseADModel` | None of the above | `loss()`, `predict()` |

## Step 2: Create the Detector File

Create `baoiad/models/detectors/my_method.py`:

```python
"""MyMethod anomaly detector."""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from baoiad.registry import MODELS
from baoiad.models.base_ad_model import KnowledgeDistillationADModel
from baoiad.models.predict_utils import build_predict_results
from baoiad.structures import ADDataSample


@MODELS.register_module()
class MyMethod(KnowledgeDistillationADModel):
    """My method for anomaly detection.

    Args:
        backbone: Backbone config dict.
        neck: Neck config dict.
        head: Head config dict.
        my_param: Description of custom parameter.
    """

    def __init__(
        self,
        backbone: Optional[dict] = None,
        neck: Optional[dict] = None,
        head: Optional[dict] = None,
        my_param: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(backbone=backbone, neck=neck, head=head, **kwargs)
        self.my_param = my_param

    def loss(self, batch_inputs: Tensor, data_samples: List[ADDataSample]) -> Dict[str, Tensor]:
        """Compute training loss.

        Args:
            batch_inputs: Input images, shape (B, C, H, W).
            data_samples: Ground-truth data samples.

        Returns:
            Dictionary of loss tensors.
        """
        teacher_feats = self.extract_feat(batch_inputs)
        student_feats = self.head(teacher_feats)
        loss = self.compute_discrepancy(teacher_feats, student_feats)
        return {'loss': loss}

    def predict(self, batch_inputs: Tensor, data_samples: List[ADDataSample]) -> List[ADDataSample]:
        """Predict anomaly scores.

        Args:
            batch_inputs: Input images, shape (B, C, H, W).
            data_samples: Data samples to populate with predictions.

        Returns:
            Updated data samples with pred_score and pred_anomaly_map.
        """
        teacher_feats = self.extract_feat(batch_inputs)
        student_feats = self.head(teacher_feats)
        anomaly_map = self.compute_anomaly_map(teacher_feats, student_feats)

        return build_predict_results(
            data_samples=data_samples,
            anomaly_map=anomaly_map,
        )
```

### Key Points

1. **Always use `@MODELS.register_module()`** to register the detector.
2. **Use `build_predict_results()`** from `baoiad.models.predict_utils` to construct predictions uniformly.
3. **Implement `loss()`** for training and **`predict()`** for inference.
4. **Backbone features** are accessed via `self.extract_feat(batch_inputs)`, which handles freezing automatically.

## Step 3: Register in `__init__.py`

Add the import to `baoiad/models/detectors/__init__.py`:

```python
from .my_method import MyMethod
```

## Step 4: Create Config

Create `configs/my_method/my_method_wrn50_256_mvtec_strict.py`:

```python
_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/mvtec_ad.py',
    '../_base_/schedules/schedule_100e.py',
]

model = dict(
    type='MyMethod',
    backbone=dict(
        type='TIMMBackbone',
        model_name='wide_resnet50_2',
        pretrained=True,
        features_only=True,
        out_indices=(2, 3),
        frozen=True,
    ),
    neck=dict(
        type='MultiScalePooling',
        output_size=28,
    ),
    head=dict(
        type='MyMethodHead',
        my_param=0.1,
    ),
    freeze_backbone=True,
)
```

## Step 5: Add Tests

Create `tests/test_models/test_detectors/test_my_method.py`:

```python
"""Tests for MyMethod detector."""

import pytest
import torch

import baoiad
from baoiad import register_all_modules
from baoiad.registry import MODELS

register_all_modules()


def test_my_method_forward():
    """Test forward pass in loss mode."""
    model = MODELS.build(dict(
        type='MyMethod',
        backbone=dict(type='_DummyBackbone'),
        head=dict(type='_DummyHead'),
    ))
    model.train()

    batch_inputs = torch.randn(2, 3, 64, 64)
    data_samples = [baoiad.structures.ADDataSample() for _ in range(2)]
    for ds in data_samples:
        ds.gt_label = 0

    losses = model(batch_inputs, data_samples, mode='loss')
    assert 'loss' in losses


def test_my_method_predict():
    """Test forward pass in predict mode."""
    model = MODELS.build(dict(
        type='MyMethod',
        backbone=dict(type='_DummyBackbone'),
        head=dict(type='_DummyHead'),
    ))
    model.eval()

    batch_inputs = torch.randn(2, 3, 64, 64)
    data_samples = [baoiad.structures.ADDataSample() for _ in range(2)]
    for ds in data_samples:
        ds.gt_label = 0

    results = model(batch_inputs, data_samples, mode='predict')
    assert len(results) == 2
```

:::{note}
Use `_DummyBackbone` and `_DummyHead` from `tests/test_models/test_base_ad_model.py` for unit tests. These are lightweight stubs that avoid loading real pretrained weights.
:::

## Step 6: Verify

```bash
# Run the test
pytest tests/test_models/test_detectors/test_my_method.py -v

# Quick smoke test
python tools/train.py configs/my_method/my_method_wrn50_256_mvtec_strict.py \
    --work-dir runs/test_my_method \
    --cfg-options train_cfg.max_epochs=2
```
