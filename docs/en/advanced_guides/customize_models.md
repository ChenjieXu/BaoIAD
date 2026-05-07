# Customizing Models

BaoIAD models follow the **backbone → neck → head** decomposition inherited from `BaseADModel`. Each component can be customized independently via config dicts built by the `MODELS` registry.

## Model Architecture

All models are registered in the `MODELS` registry (from `baoiad/registry.py`):

```python
from baoiad.registry import MODELS
```

### Base Classes

| Base Class | Module | Purpose |
|-----------|--------|---------|
| `BaseADModel` | `baoiad.models.base_ad_model` | Backbone → neck → head, 3-mode forward |
| `MemoryBankADModel` | `baoiad.models.base_ad_model` | Feature collection + memory bank lifecycle |
| `KnowledgeDistillationADModel` | `baoiad.models.base_ad_model` | Teacher-student with frozen teacher |
| `FlowBasedADModel` | `baoiad.models.base_ad_model` | Normalizing flow + NLL loss |
| `ReconstructionADModel` | `baoiad.models.base_ad_model` | Autoencoder/reconstruction pattern |
| `VisionLanguageADModel` | `baoiad.models.base_ad_model` | CLIP-based zero/few-shot |
| `DiscriminatorADModel` | `baoiad.models.base_ad_model` | Feature discrimination with noise |

See [Base AD Model Guide](./base_ad_model.md) for the full class hierarchy.

## Customizing Model Components

### Custom Backbone

All available backbones are documented in the [Backbone Registry Guide](./backbone_registry.md). Here is a typical backbone config:

```python
model = dict(
    type='PatchCore',
    backbone=dict(
        type='TIMMBackbone',            # From baoiad.models.backbones.timm_backbone
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

To use a different backbone architecture, change the `type` and parameters:

```python
# DINOv2 backbone for a feature-based method
backbone=dict(
    type='DINOv2Backbone',               # From baoiad.models.backbones.dinov2_backbone
    model_name='dinov2_vitb14',
    frozen=True,
    pretrained=True,
)

# OpenCLIP backbone for vision-language methods
backbone=dict(
    type='OpenCLIPBackbone',             # From baoiad.models.backbones.clip_backbone
    model_name='ViT-L-14-336',
    pretrained='openai',
    frozen=True,
    force_quick_gelu=True,
)
```

### Custom Neck

Necks process multi-scale features before passing them to the head:

```python
neck=dict(
    type='MultiScalePooling',
    output_size=28,              # Spatial size of pooled features
)
```

Necks are optional — methods that handle features directly in the head can omit the neck config.

### Custom Head

Heads contain the method-specific anomaly scoring logic:

```python
head=dict(
    type='MemoryBankHead',          # From baoiad.models.heads
    coreset_ratio=0.1,              # Coreset subsampling ratio
    num_neighbors=9,                # kNN neighbors
    distance='euclidean',           # Distance metric
    blur_sigma=4.0,                 # Gaussian blur for anomaly map
)
```

Each head must implement:
- `loss(feats, data_samples)` → `dict[str, Tensor]` — called during `forward(mode='loss')`
- `predict(feats, data_samples)` → `list[ADDataSample]` — called during `forward(mode='predict')`

## Multi-Optimizer Setup

Some methods require different learning rates for different components. BaoIAD uses MMEngine's `optim_wrapper` system.

### Single Optimizer (default)

```python
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=0.001, weight_decay=1e-5),
)
```

### Multiple Optimizers

For methods with separate optimizers for different components (e.g., SimpleNet with different LR for projector and discriminator):

```python
optim_wrapper = dict(
    projector=dict(
        optimizer=dict(type='Adam', lr=0.001),
    ),
    discriminator=dict(
        optimizer=dict(type='Adam', lr=0.0001),
    ),
)
```

### Optimizer with ParamScheduler

```python
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.01),
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=True,
        begin=0,
        end=5,
    ),  # Warmup
    dict(
        type='CosineAnnealingLR',
        T_max=95,
        by_epoch=True,
        begin=5,
        end=100,
    ),  # Main schedule
]
```

## Custom Hooks

### MemoryBankHook

Always active via `configs/_base_/default_runtime.py`. Manages the memory bank lifecycle for methods like PatchCore, PaDiM, etc.

```python
custom_hooks = [dict(type='MemoryBankHook')]
```

See [Memory Bank Guide](./memory_bank.md) for the full lifecycle documentation.

### ADVisualizationHook

Disabled by default. Enable for visual anomaly map output:

```python
default_hooks = dict(
    visualization=dict(type='ADVisualizationHook', enable=True),
)
```

### Custom Hook Example

```python
from mmengine.hooks import Hook
from baoiad.registry import HOOKS


@HOOKS.register_module()
class MyCustomHook(Hook):

    def after_train_iter(self, runner, batch_idx, data_batch, outputs):
        # Custom logic after each training iteration
        pass

    def after_val_epoch(self, runner, metrics):
        # Custom logic after each validation epoch
        pass
```

Use in config:

```python
custom_hooks = [
    dict(type='MemoryBankHook'),
    dict(type='MyCustomHook'),
]
```

## Freeze Control

Control which components are frozen:

```python
model = dict(
    type='RD',
    backbone=dict(...),
    freeze_backbone=True,     # Freeze backbone (default for most bases)
)
```

`BaseADModel` freezes the backbone automatically when `freeze_backbone=True`:

```python
def _freeze_backbone(self):
    self.backbone.eval()
    for param in self.backbone.parameters():
        param.requires_grad = False
```

For fine-grained control, override `_freeze_module()` in a detector subclass:

```python
from baoiad.models.base_ad_model import BaseADModel


class MyDetector(BaseADModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Freeze specific layers
        self._freeze_module(self.backbone.layer4)
```

## Loss Configuration

All losses are registered in the `MODELS` registry. See [Customizing Losses](./customize_losses.md) for the full catalog.

```python
model = dict(
    type='MyDetector',
    backbone=dict(...),
    head=dict(
        type='MyHead',
        loss_recon=dict(type='MSELoss', loss_weight=1.0),
        loss_seg=dict(type='BinaryFocalLoss', alpha=0.25, gamma=2.0),
    ),
)
```

## Creating a New Model

### Step 1: Choose a Base Class

Pick the appropriate base from `baoiad.models.base_ad_model`:

- **Feature-memory methods** → `MemoryBankADModel`
- **Teacher-student methods** → `KnowledgeDistillationADModel`
- **Normalizing flow methods** → `FlowBasedADModel`
- **Reconstruction methods** → `ReconstructionADModel`
- **CLIP-based methods** → `VisionLanguageADModel`
- **Discriminator methods** → `DiscriminatorADModel`
- **Novel architecture** → `BaseADModel`

### Step 2: Implement the Detector

```python
# baoiad/models/detectors/my_detector.py
import torch
import torch.nn as nn
from baoiad.models.base_ad_model import BaseADModel
from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS


@MODELS.register_module()
class MyDetector(BaseADModel):
    def __init__(
        self,
        backbone: dict,
        head: dict,
        neck: dict | None = None,
        freeze_backbone: bool = True,
        **kwargs,
    ):
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            freeze_backbone=freeze_backbone,
            **kwargs,
        )
        # Additional initialization

    def forward(self, inputs, data_samples=None, mode='tensor'):
        feats = self.extract_feat(inputs)

        if mode == 'loss':
            return self.head.loss(feats, data_samples)
        elif mode == 'predict':
            return self.head.predict(feats, data_samples)
        return feats
```

### Step 3: Implement the Head

```python
# baoiad/models/heads/my_head.py
import torch
import torch.nn as nn
from mmengine.model import BaseModule
from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS
from baoiad.structures import ADDataSample


@MODELS.register_module()
class MyHead(BaseModule):
    def __init__(self, in_channels=256, loss=dict(type='MSELoss'), init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        from baoiad.registry import MODELS
        self.loss_fn = MODELS.build(loss)
        self.projector = nn.Linear(in_channels, in_channels)

    def loss(self, feats, data_samples):
        # Compute training loss
        feat = feats[0]  # Use first scale
        projected = self.projector(feat.mean(dim=[2, 3]))
        target = torch.zeros_like(projected)
        loss = self.loss_fn(projected, target)
        return dict(loss=loss)

    def predict(self, feats, data_samples):
        # Compute anomaly scores and maps
        feat = feats[0]
        scores = feat.mean(dim=[1, 2, 3])
        # Build anomaly map from features
        maps = feat.mean(dim=1, keepdim=True)

        return build_predict_results(
            data_samples=data_samples,
            img_scores=scores,
            score_maps=maps,
        )
```

### Step 4: Register Imports

Add imports in the appropriate `__init__.py` files:

```python
# baoiad/models/detectors/__init__.py
from .my_detector import MyDetector  # noqa: F401

# baoiad/models/heads/__init__.py
from .my_head import MyHead  # noqa: F401
```

### Step 5: Create a Config

```python
# configs/my_method/my_method_wrn50_256_mvtec.py
model = dict(
    type='MyDetector',
    backbone=dict(
        type='TIMMBackbone',
        model_name='wide_resnet50_2',
        pretrained=True,
        features_only=True,
        out_indices=(1, 2, 3),
        frozen=True,
    ),
    head=dict(
        type='MyHead',
        in_channels=256,
        loss=dict(type='MSELoss', loss_weight=1.0),
    ),
)
```

### Step 6: Train and Test

```bash
python tools/train.py configs/my_method/my_method_wrn50_256_mvtec.py --work-dir runs/my_method
python tools/test.py configs/my_method/my_method_wrn50_256_mvtec.py runs/my_method/best.pth
```

## Registry Reference

All registries are defined in `baoiad/registry.py`:

| Registry | Scope | Used for |
|----------|-------|----------|
| `MODELS` | `baoiad` | Models, backbones, necks, heads, losses |
| `DATASETS` | `baoiad` | Dataset classes |
| `TRANSFORMS` | `baoiad` | Data transforms |
| `METRICS` | `baoiad` | Evaluation metrics |
| `HOOKS` | `baoiad` | Training hooks |
| `LOOPS` | `baoiad` | Training/validation loops |
| `RUNNERS` | `baoiad` | Runner classes |
| `VISUALIZERS` | `baoiad` | Visualization tools |

Each registry is a child of the corresponding MMEngine registry with `scope='baoiad'`, meaning BaoIAD modules are looked up first, falling back to MMEngine's built-in modules.
