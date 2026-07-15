# Config System

BaoIAD uses MMEngine's hierarchical config system with `_base_` inheritance. A typical method config is minimal — it only defines the `model` and inherits everything else from shared base configs.

## Config Inheritance

Use the `_base_` field to inherit from one or more base configs:

```python
# configs/patchcore/patchcore_wrn50_256_mvtec_strict.py
_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/mvtec_ad.py',
    '../_base_/schedules/schedule_100e.py',
]

model = dict(
    type='PatchCore',
    backbone=dict(
        type='TIMMBackbone',
        model_name='wide_resnet50_2',
        pretrained=True,
        features_only=True,
        out_indices=(2, 3),
        frozen=True,
    ),
    neck=dict(type='MultiScalePooling', output_size=28),
    head=dict(
        type='MemoryBankHead',
        coreset_ratio=0.1,
        num_neighbors=9,
        distance='euclidean',
        input_size=(256, 256),
        blur_sigma=4.0,
    ),
    freeze_backbone=True,
)
```

Fields defined in the current config override inherited values of the same name.

## Base Config Structure

```
configs/_base_/
├── backbones/
│   ├── wide_resnet50_unified.py    # WRN-50-2 (repository-standardized settings)
│   ├── wide_resnet50.py            # WRN-50-2 (default)
│   ├── wide_resnet50_raw.py        # WRN-50-2 (raw, no preprocessing)
│   ├── resnet18.py                 # ResNet-18
│   ├── resnet50.py                 # ResNet-50
│   ├── resnet18_raw.py             # ResNet-18 (raw)
│   ├── efficientnet_b4.py          # EfficientNet-B4
│   ├── efficientnet_b5.py          # EfficientNet-B5
│   ├── dinov2_vitb14.py            # DINOv2 ViT-B/14
│   └── dinov2reg_vit_base_14.py    # DINOv2-reg ViT-B/14
├── datasets/
│   ├── mvtec_ad.py                 # MVTec AD dataset + dataloader + evaluator
│   ├── visa.py                     # VisA
│   ├── btech.py                    # BTech
│   ├── mvtec_3d_ad.py              # MVTec 3D AD
│   ├── mvtec_loco_ad.py            # MVTec LOCO
│   ├── mpdd.py                     # MPDD
│   ├── mvtec_ad2.py                # MVTec AD 2
│   ├── kolektor.py                 # Kolektor
│   ├── vad.py                      # VAD
│   └── realiad.py                  # RealIAD
├── schedules/
│   └── schedule_100e.py            # 100 epochs, Adam, CosineAnnealing
└── default_runtime.py              # scope='baoiad', seed=42, MemoryBankHook
```

## Config Components

A fully resolved config contains these top-level fields:

| Field | Source | Description |
|-------|--------|-------------|
| `default_scope` | Runtime | Registry scope (`'baoiad'`) |
| `custom_imports` | Runtime | Auto-imports the `baoiad` package to trigger registration |
| `model` | Method config | Model definition (type, backbone, neck, head) |
| `train_dataloader` | Dataset config | Training dataloader (dataset, batch_size, sampler, pipeline) |
| `test_dataloader` | Dataset config | Test dataloader |
| `val_dataloader` | Dataset config | Validation dataloader (usually same as test) |
| `train_cfg` | Schedule config | Training loop config (`by_epoch`, `max_epochs`, `val_interval`) |
| `test_cfg` | Runtime | Test loop type (`ADTestLoop`) |
| `val_cfg` | Runtime | Validation loop type (`ADValLoop`) |
| `optim_wrapper` | Schedule config | Optimizer wrapper (e.g., Adam with lr=1e-3) |
| `param_scheduler` | Schedule config | Learning rate scheduler (e.g., CosineAnnealing) |
| `custom_hooks` | Runtime | Custom hooks (e.g., `MemoryBankHook`) |
| `default_hooks` | Runtime | Timer, logger, checkpoint, visualization hooks |
| `env_cfg` | Runtime | cuDNN, multiprocessing, distributed backend settings |
| `visualizer` | Runtime | `ADVisualizer` with `LocalVisBackend` |
| `randomness` | Runtime | Seed=42, deterministic=False |
| `test_evaluator` | Dataset config | `AnomalyDetectionMetric` |
| `val_evaluator` | Dataset config | `AnomalyDetectionMetric` |

## Runtime Config

[`configs/_base_/default_runtime.py`](../../../configs/_base_/default_runtime.py) sets the foundation for all method configs:

- **Scope**: `default_scope = 'baoiad'` ensures all component lookups use the BaoIAD registries.
- **Custom imports**: `custom_imports = dict(imports=['baoiad'])` triggers module registration.
- **Loops**: `ADTestLoop` and `ADValLoop` support deferred scoring (required by methods like MuSc that aggregate features across the test set before computing scores).
- **Hooks**: `MemoryBankHook` is included by default for methods that build a memory bank during training. `ADVisualizationHook` is disabled by default (enable via `--cfg-options`).
- **Reproducibility**: `seed=42` matches the anomalib default.

## Schedule Config

[`configs/_base_/schedules/schedule_100e.py`](../../../configs/_base_/schedules/schedule_100e.py) defines:

```python
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=1e-3, weight_decay=1e-5),
)
param_scheduler = [
    dict(type='CosineAnnealingLR', T_max=100, by_epoch=True, eta_min=1e-5),
]
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=10)
```

Methods that do not use gradient-based training (e.g., PatchCore, PaDiM) ignore the optimizer but still use `train_cfg.max_epochs` to control the number of passes over the data.

## Dataset Config

Each dataset base config defines `train_dataloader` and `test_dataloader` with the dataset class, data root, and pipeline. See [Dataset Zoo](../dataset_zoo.md) for details on all 10 datasets.

Key fields in a dataset config:

```python
# configs/_base_/datasets/mvtec_ad.py
data_root = 'data/mvtec_ad'
img_size = 256

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='MVTecADDataset',
        data_root=data_root,
        split='train',
        multi_class=True,    # iterate over all categories
        pipeline=train_pipeline,
    ),
)

test_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MVTecADDataset',
        data_root=data_root,
        split='test',
        multi_class=True,
        pipeline=test_pipeline,
    ),
)

val_dataloader = test_dataloader
test_evaluator = dict(type='AnomalyDetectionMetric')
val_evaluator = test_evaluator
```

## Runtime Overrides

Override any config field from the command line using `--cfg-options` with `key=value` pairs:

```bash
python tools/train.py <config> --work-dir runs/test \
    --cfg-options \
    train_dataloader.batch_size=16 \
    train_dataloader.dataset.cls_names="['bottle']" \
    train_dataloader.dataset.multi_class=False \
    model.head.coreset_ratio=0.01 \
    optim_wrapper.optimizer.lr=0.001
```

### Common Overrides

| Override | Effect |
|----------|--------|
| `train_dataloader.batch_size=N` | Change training batch size |
| `test_dataloader.batch_size=N` | Change test batch size |
| `train_dataloader.dataset.cls_names="['bottle']"` | Select specific categories |
| `train_dataloader.dataset.multi_class=False` | Single-category mode (must set with `cls_names`) |
| `train_dataloader.dataset.data_root=/path` | Change dataset root |
| `test_dataloader.dataset.data_root=/path` | Change test dataset root |
| `model.head.<param>=<value>` | Change model-specific parameters |
| `optim_wrapper.optimizer.lr=<value>` | Change learning rate |
| `default_hooks.visualization.enable=True` | Enable visualization outputs |

When selecting a single category, you must set both `cls_names` and `multi_class=False` for both `train_dataloader` and `test_dataloader`:

```bash
python tools/train.py <config> --work-dir runs/bottle \
    --cfg-options \
    train_dataloader.dataset.cls_names="['bottle']" \
    train_dataloader.dataset.multi_class=False \
    test_dataloader.dataset.cls_names="['bottle']" \
    test_dataloader.dataset.multi_class=False
```

## Strict vs Unified Configs

BaoIAD provides two config variants for most methods:

- **`*_strict.py`** (e.g., `patchcore_wrn50_256_mvtec_strict.py`): Reference configs that align with the original paper implementation, using whatever backbone, resolution, and hyperparameters the original authors specified. These are the configs used for alignment evidence in [`docs/alignment/`](../../../docs/alignment/).
- **`*_unified.py`** or configs without a `strict` suffix: Configs using repository-standardized settings such as `configs/_base_/backbones/wide_resnet50_unified.py`. Shared initialization removes one source of variation but does not by itself guarantee that different methods are directly or fairly comparable.

Use `strict` configs when reproducing a specific paper's results. Use unified configs when comparing methods head-to-head.

## Model Config Structure

All detectors follow a `backbone → neck → head` pattern:

```python
model = dict(
    type='<MethodName>',           # Registered in MODELS registry
    backbone=dict(
        type='<BackboneClass>',     # e.g., TIMMBackbone, DINOv2Backbone
        ...
    ),
    neck=dict(
        type='<NeckClass>',         # e.g., MultiScalePooling
        ...
    ),
    head=dict(
        type='<HeadClass>',         # e.g., MemoryBankHead, SimpleNetHead
        ...
    ),
    freeze_backbone=True,           # Common for feature-extraction methods
)
```

Not all methods use all three components. Some methods define only a subset (e.g., no `neck`), and some have additional method-specific fields.

## See Also

- [Get Started](../get_started.md) — Installation and first run
- [Dataset Zoo](../dataset_zoo.md) — Supported datasets and directory structures
- [Prepare Datasets](prepare_dataset.md) — Download and setup instructions
