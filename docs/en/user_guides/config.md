# Config System

BaoIAD uses MMEngine's config system with inheritance, allowing concise method definitions that reuse shared base configs.

## Config Inheritance

Configs can inherit from one or more base configs using the `_base_` field:

```python
# configs/patchcore/patchcore_wrn50_256_mvtec_strict.py
_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/mvtec_ad.py',
    '../_base_/schedules/schedule_100e.py',
]
```

The `_base_` list specifies which base configs to inherit. Fields defined in the current config override inherited values.

## Base Config Structure

```
configs/_base_/
├── backbones/
│   ├── wide_resnet50_unified.py    # WRN-50-2 (standardized for fair comparison)
│   ├── wide_resnet50.py            # WRN-50-2 (default)
│   ├── resnet18.py                 # ResNet-18
│   ├── resnet50.py                 # ResNet-50
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
│   └── schedule_100e.py            # 100 epochs, SGD, CosineAnnealing
└── default_runtime.py              # scope='baoiad', seed=42, MemoryBankHook
```

## Config Walkthrough: PatchCore

```python
# 1. Inherit base configs
_base_ = [
    '../_base_/default_runtime.py',    # Runtime: scope, hooks, seed
    '../_base_/datasets/mvtec_ad.py',  # Dataset: MVTec AD, batch_size=32
    '../_base_/schedules/schedule_100e.py',  # Schedule: 100 epochs
]

# 2. Define model
model = dict(
    type='PatchCore',
    backbone=dict(
        type='TIMMBackbone',
        model_name='wide_resnet50_2',
        pretrained=True,
        features_only=True,
        out_indices=(2, 3),  # layer2 + layer3 features
        frozen=True,
    ),
    neck=dict(
        type='MultiScalePooling',
        output_size=28,
    ),
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

## Config Components

A full config contains these top-level fields:

| Field | Description |
|-------|-------------|
| `model` | Model definition (type, backbone, neck, head) |
| `train_dataloader` | Training dataloader (dataset, batch_size, sampler) |
| `test_dataloader` | Test dataloader |
| `val_dataloader` | Validation dataloader |
| `train_cfg` | Training loop config |
| `val_cfg` | Validation loop config |
| `test_cfg` | Test loop config |
| `optim_wrapper` | Optimizer wrapper |
| `param_scheduler` | Learning rate scheduler |
| `custom_hooks` | Custom hooks (e.g., MemoryBankHook) |
| `default_hooks` | Default hooks (timer, logger, checkpoint) |
| `randomness` | Seed and deterministic settings |
| `default_scope` | Registry scope (`'baoiad'`) |

## Runtime Config Override

Override any config field from the command line using `--cfg-options`:

```bash
python tools/train.py <config> --work-dir runs/test \
    --cfg-options \
    train_dataloader.batch_size=16 \
    train_dataloader.dataset.cls_names="['bottle']" \
    train_dataloader.dataset.multi_class=False \
    model.head.coreset_ratio=0.01 \
    optim_wrapper.optimizer.lr=0.001
```

Key overrides:

| Override | Effect |
|----------|--------|
| `train_dataloader.batch_size=N` | Change batch size |
| `train_dataloader.dataset.cls_names="['bottle']"` | Select categories |
| `train_dataloader.dataset.multi_class=False` | Single-category mode |
| `train_dataloader.dataset.data_root=/path` | Change data root |
| `model.head.<param>=<value>` | Change model head parameters |
| `optim_wrapper.optimizer.lr=<value>` | Change learning rate |

## Strict vs Unified Configs

- **`*_strict.py`**: Reference configs aligned with the original paper implementation for each method
- **`*_unified.py`**: Configs using a standardized WRN-50-2 backbone for fair cross-method comparison

The unified configs inherit from `configs/_base_/backbones/wide_resnet50_unified.py` to ensure consistent backbone initialization across all methods.
