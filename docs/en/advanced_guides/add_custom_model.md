# Add a Custom Model

This tutorial walks through implementing a new anomaly detection model and integrating it into BaoIAD. You will learn the model base-class hierarchy, the `MODELS` registry, config wiring, and the benchmark inventory.

## Prerequisites

- BaoIAD installed (`pip install -e ".[all]"`)
- Familiarity with [MMEngine's Registry system](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/registry.html)
- Read [Base AD Model](base_ad_model.md) for the class hierarchy

## Architecture Overview

Every BaoIAD detector inherits from one of the base classes in `baoiad/models/base_ad_model.py`:

| Base class | Use case | Examples |
|---|---|---|
| `BaseADModel` | General backbone → neck → head | Custom detectors |
| `MemoryBankADModel` | Feature collection + memory bank | PatchCore, PaDiM |
| `KnowledgeDistillationADModel` | Teacher-student | RD, EfficientAD |
| `FlowBasedADModel` | Normalizing flows | FastFlow, CFlow-AD |
| `ReconstructionADModel` | Autoencoder / reconstruction | DRAEM, MemSeg |
| `VisionLanguageADModel` | CLIP-based zero/few-shot | WinCLIP, AnomalyCLIP |
| `DiscriminatorADModel` | Feature discrimination + noise | SimpleNet, CFA |

All registries live in `baoiad/registry.py` and use the `baoiad` scope:

```python
from baoiad.registry import MODELS, DATASETS, TRANSFORMS, METRICS, HOOKS, VISUALIZERS
```

## Step 1: Create the Model File

Create `baoiad/models/detectors/my_detector.py`. The model must:

1. Inherit from the appropriate base class
2. Be decorated with `@MODELS.register_module()`
3. Implement `forward()` with three modes: `loss`, `predict`, `tensor`

Below is a minimal working example of a discriminator-style detector:

```python
"""MyDetector: a minimal discriminative anomaly detector."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from baoiad.models.base_ad_model import DiscriminatorADModel
from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS


class SimpleDiscriminator(nn.Module):
    """Two-layer MLP discriminator."""

    def __init__(self, in_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@MODELS.register_module()
class MyDetector(DiscriminatorADModel):
    """A minimal discriminative AD detector.

    Extracts backbone features, optionally projects them, adds Gaussian
    noise to synthesize anomalies, and trains a discriminator to separate
    real features from noisy ones.

    Args:
        backbone: Backbone config dict.
        projection_dim: Output dimension of the feature projection.
        noise_std: Standard deviation of the Gaussian noise.
        margin: Hinge loss margin.
    """

    def __init__(
        self,
        backbone: dict,
        projection_dim: int = 256,
        noise_std: float = 0.015,
        margin: float = 0.5,
        freeze_backbone: bool = True,
        data_preprocessor=None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(
            backbone=backbone,
            freeze_backbone=freeze_backbone,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
        )
        # Determine the concatenated feature dimension.
        # You can also read this from backbone.out_channels.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 256, 256)
            dummy_feats = self.extract_feat(dummy)
            feat_dim = sum(f.shape[1] for f in dummy_feats) * dummy_feats[0].shape[2] * dummy_feats[0].shape[3] // (dummy_feats[0].shape[2] * dummy_feats[0].shape[3])
            # Simpler: just flatten and check
            feat_dim = sum(f.shape[1] for f in dummy_feats)

        self.projection = nn.Linear(feat_dim, projection_dim)
        self.discriminator = SimpleDiscriminator(projection_dim)
        self.noise_std = noise_std
        self.margin = margin

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)

        feats = self.extract_feat(inputs)
        # Flatten and concatenate multi-scale features
        B = feats[0].shape[0]
        flat = torch.cat([f.mean(dim=[2, 3]) for f in feats], dim=1)  # (B, C)
        projected = self.projection(flat)

        if mode == 'loss':
            return self._compute_loss(projected)
        elif mode == 'predict':
            return self._predict(projected, B, inputs, data_samples)
        else:
            return projected

    def _compute_loss(self, feats: torch.Tensor) -> dict:
        noise = torch.randn_like(feats) * self.noise_std
        fake_feats = feats + noise
        scores = self.discriminator(torch.cat([feats, fake_feats]))
        real_scores = scores[:feats.shape[0]]
        fake_scores = scores[feats.shape[0]:]
        real_loss = torch.clamp(-real_scores + self.margin, min=0).mean()
        fake_loss = torch.clamp(fake_scores + self.margin, min=0).mean()
        return {'loss': real_loss + fake_loss}

    def _predict(self, feats, B, inputs, data_samples):
        patch_scores = -self.discriminator(feats).squeeze(-1)
        img_scores = patch_scores
        return build_predict_results(data_samples, img_scores)

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, 'backbone') and self.freeze_backbone:
            self.backbone.eval()
        return self
```

Key points:

- **`forward()` with three modes** is the core MMEngine contract. `loss` returns a dict of tensors for backprop. `predict` returns a list of `ADDataSample`. `tensor` returns raw tensors.
- **`build_predict_results()`** from `baoiad/models/predict_utils.py` is a helper that packs image-level scores and pixel-level anomaly maps into `ADDataSample` objects that the evaluator expects.
- **`extract_feat()`** is inherited from `BaseADModel` and runs backbone + neck with frozen-backbone handling.

## Step 2: Register the Model

The `@MODELS.register_module()` decorator adds your class to the `baoiad` scoped registry. The registry name (the class name string) is what you use in config files.

Make sure the module is importable by adding it to `baoiad/models/detectors/__init__.py`:

```python
from .my_detector import MyDetector  # noqa: F401
```

And ensure `baoiad/models/__init__.py` imports from `detectors`:

```python
from .detectors import *  # or add MyDetector explicitly
```

## Step 3: Create a Config File

Create `configs/mydetector/mydetector_wrn50_256_mvtec.py`:

```python
_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/mvtec_ad.py',
    '../_base_/schedules/schedule_100e.py',
]

model = dict(
    type='MyDetector',
    backbone=dict(
        type='TIMMBackbone',
        model_name='wide_resnet50_2',
        pretrained=True,
        features_only=True,
        out_indices=(2, 3),
        frozen=True,
    ),
    projection_dim=256,
    noise_std=0.015,
    margin=0.5,
    freeze_backbone=True,
)
```

The `type` field must match the registry name of your class (the class name by default).

## Step 4: Add to the Method Inventory

To make your method visible to the benchmark runner (`tools/benchmark.py`), add a `MethodEntry` to `baoiad/method_inventory.py`:

```python
MethodEntry(
    slug='mydetector',
    display='MyDetector',
    family='Discriminative',
    config_paths=(
        'configs/mydetector/mydetector_wrn50_256_mvtec.py',
        'configs/mydetector/mydetector_wrn50_256_visa.py',
    ),
    readme_path='configs/mydetector/README.md',
    alignment_path='docs/alignment/mydetector.md',
),
```

## Step 5: Create Config README and Alignment Evidence

**Config README** (`configs/mydetector/README.md`): describe the method, list config variants, note any reference implementation you align to, and record key hyperparameters.

**Alignment evidence** (`docs/alignment/mydetector.md`): document code-path parity checks and behavior probes that confirm your implementation matches the reference. See `docs/alignment/` for examples from existing methods.

## Step 6: Train and Test

```bash
# Train
python tools/train.py configs/mydetector/mydetector_wrn50_256_mvtec.py \
    --work-dir runs/mydetector/mvtec

# Test
python tools/test.py configs/mydetector/mydetector_wrn50_256_mvtec.py \
    runs/mydetector/mvtec/best.pth
```

## Choosing the Right Base Class

### MemoryBankADModel

For methods that collect features during training and build a memory bank (e.g., kNN-based). Override `build_memory_bank()`:

```python
from baoiad.models.base_ad_model import MemoryBankADModel

@MODELS.register_module()
class MyMemoryModel(MemoryBankADModel):
    def build_memory_bank(self):
        # Process self._memory_bank (list of collected tensors)
        all_feats = torch.cat(self._memory_bank, dim=0)
        self.bank = all_feats  # store for inference
        self._clear_memory_bank()
```

The `MemoryBankHook` (in `baoiad/engine/hooks/memory_bank_hook.py`) automatically calls `build_memory_bank()` after the last training epoch.

### KnowledgeDistillationADModel

For teacher-student architectures. Provides `extract_teacher_feats()` which runs the backbone with `torch.no_grad()`:

```python
from baoiad.models.base_ad_model import KnowledgeDistillationADModel

@MODELS.register_module()
class MyKDModel(KnowledgeDistillationADModel):
    def __init__(self, backbone, teacher_channels, student_channels, **kwargs):
        super().__init__(backbone=backbone, **kwargs)
        self.teacher = self.backbone  # frozen teacher
        self.student = self._build_student(student_channels)

    def forward(self, inputs, data_samples=None, mode='tensor'):
        inputs = self._stack_inputs(inputs)
        teacher_feats = self.extract_teacher_feats(inputs)
        student_feats = self.student(teacher_feats)
        # ...
```

### Using Heads Instead of Inline Logic

For backbone → neck → head architectures, you can delegate to a head module:

```python
@MODELS.register_module()
class MyHeadedModel(BaseADModel):
    def __init__(self, backbone, head, **kwargs):
        super().__init__(backbone=backbone, head=head, **kwargs)
        # forward() is already handled by BaseADModel:
        #   loss mode  → self.head.loss(feats, data_samples)
        #   predict mode → self.head.predict(feats, data_samples)
```

See `baoiad/models/heads/memory_bank_head.py` for a full head implementation example.
