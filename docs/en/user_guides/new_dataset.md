# Adding a New Dataset

This guide explains how to add support for a new dataset in BaoIAD.

## Step 1: Create the Dataset Class

Create a new file in `baoiad/datasets/`, extending `BaseADDataset`:

```python
"""MyDataset for industrial anomaly detection."""

from typing import List, Optional

from baoiad.datasets.base_ad_dataset import BaseADDataset
from baoiad.registry import DATASETS


@DATASETS.register_module()
class MyDataset(BaseADDataset):
    """My custom dataset.

    Args:
        data_root: Path to dataset root directory.
        cls_names: List of category names to load.
        multi_class: Whether to load all categories together.
        split: Dataset split ('train' or 'test').
        ...existing BaseADDataset args...
    """

    # List of all categories in the dataset
    ALL_CATEGORIES = ['cat_a', 'cat_b', 'cat_c']

    def __init__(
        self,
        data_root: str,
        cls_names: Optional[List[str]] = None,
        multi_class: bool = True,
        split: str = 'train',
        **kwargs,
    ) -> None:
        cls_names = cls_names or self.ALL_CATEGORIES
        super().__init__(
            data_root=data_root,
            cls_names=cls_names,
            multi_class=multi_class,
            split=split,
            **kwargs,
        )
```

### BaseADDataset Interface

`BaseADDataset` handles the common boilerplate for AD datasets:

- Iterates over category directories
- Loads train/test splits
- Assigns `gt_label` (0=normal, 1=anomaly)
- Loads pixel-level `gt_mask` when available
- Populates `cls_name`, `img_path`, `defect_type` fields

You may need to override these methods if your dataset has a non-standard directory layout:

| Method | Purpose |
|--------|---------|
| `_load_data_list()` | Build the list of (image_path, gt_label, gt_mask, cls_name, defect_type) |
| `_get_defect_types()` | Return list of defect types for a category |

## Step 2: Register in `__init__.py`

Add the import to `baoiad/datasets/__init__.py`:

```python
from .my_dataset import MyDataset
```

## Step 3: Create Base Dataset Config

Create `configs/_base_/datasets/my_dataset.py`:

```python
data_root = 'data/my_dataset'

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='MyDataset',
        data_root=data_root,
        cls_names=None,  # All categories
        multi_class=True,
        split='train',
        pipeline=[
            dict(type='LoadImage'),
            dict(type='ResizeAD', size=256),
            dict(type='NormalizeAD'),
            dict(type='PackADInputs'),
        ],
    ),
)

test_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MyDataset',
        data_root=data_root,
        cls_names=None,
        multi_class=True,
        split='test',
        pipeline=[
            dict(type='LoadImage'),
            dict(type='ResizeAD', size=256),
            dict(type='NormalizeAD'),
            dict(type='PackADInputs'),
        ],
    ),
)

val_dataloader = test_dataloader

test_evaluator = dict(type='AnomalyDetectionMetric')
val_evaluator = dict(type='AnomalyDetectionMetric')
```

## Step 4: Create Method-Specific Configs

Create configs that inherit from the new dataset config:

```python
# configs/patchcore/patchcore_wrn50_256_my_dataset.py
_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/my_dataset.py',
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
    # ... rest of model config
)
```

## Step 5: Verify

```bash
# Test dataset loading
python -c "
import baoiad
from baoiad.registry import DATASETS
ds = DATASETS.build(dict(
    type='MyDataset',
    data_root='data/my_dataset',
    split='test',
    pipeline=[
        dict(type='LoadImage'),
        dict(type='ResizeAD', size=256),
        dict(type='NormalizeAD'),
        dict(type='PackADInputs'),
    ],
))
print(f'Loaded {len(ds)} samples')
print(ds[0])
"
```

## Expected Directory Structure

Most AD datasets follow this layout:

```
data/my_dataset/
├── cat_a/
│   ├── train/
│   │   └── good/
│   │       ├── 000.png
│   │       └── ...
│   └── test/
│       ├── good/
│       └── defect_type_1/
│           ├── 000.png
│           ├── 000_mask.png
│           └── ...
├── cat_b/
└── ...
```

If your dataset uses a different layout, override `_load_data_list()` to handle the custom structure.
