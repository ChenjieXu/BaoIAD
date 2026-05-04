# Customizing Models

## Customizing Model Components

BaoIAD models follow the backbone-neck-head decomposition. Each component can be customized independently.

### Custom Backbone

```python
model = dict(
    type='PatchCore',
    backbone=dict(
        type='TIMMBackbone',
        model_name='convnext_base',     # Change backbone architecture
        pretrained=True,
        features_only=True,
        out_indices=(2, 3),
        frozen=True,
    ),
    # ... neck and head configs
)
```

### Custom Neck

Necks process multi-scale features before passing to the head:

```python
model = dict(
    type='RD',
    backbone=dict(...),
    neck=dict(
        type='MultiScalePooling',
        output_size=28,              # Spatial size of pooled features
    ),
    head=dict(...),
)
```

### Custom Head

Heads contain the method-specific logic (memory bank, distillation, flow, etc.):

```python
model = dict(
    type='PatchCore',
    backbone=dict(...),
    neck=dict(...),
    head=dict(
        type='MemoryBankHead',
        coreset_ratio=0.1,          # Coreset subsampling ratio
        num_neighbors=9,             # kNN neighbors
        distance='euclidean',        # Distance metric
        blur_sigma=4.0,             # Gaussian blur for anomaly map
    ),
)
```

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

Always active via `configs/_base_/default_runtime.py`:

```python
custom_hooks = [dict(type='MemoryBankHook')]
```

### ADVisualizationHook

Disabled by default, enable for visual anomaly map output:

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

## Freeze Control

Control which components are frozen:

```python
model = dict(
    type='RD',
    backbone=dict(...),
    freeze_backbone=True,     # Freeze backbone (default)
    # Some methods also freeze the teacher:
    # freeze_teacher=True,
)
```

For fine-grained control, override `_freeze_module()` in the detector:

```python
class MyDetector(BaseADModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Freeze specific layers
        self._freeze_module(self.backbone.layer4)
```