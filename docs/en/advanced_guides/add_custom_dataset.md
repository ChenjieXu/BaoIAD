# Add a Custom Dataset

BaoIAD datasets inherit from `BaseADDataset` and are registered in the `DATASETS` registry. This tutorial shows how to add support for a new dataset.

## Base Class

`BaseADDataset` (`baoiad/datasets/base_ad_dataset.py`) inherits from `mmengine.dataset.BaseDataset` and handles:

- Multi-class vs. single-class loading modes
- Automatic `cls_names` resolution from `ALL_CATEGORIES`
- Pipeline integration

The only method you must implement is `load_data_list()`.

## Dataset Interface Contract

Each dataset entry (dict) returned by `load_data_list()` must contain:

| Key | Type | Description |
|---|---|---|
| `img_path` | `str` | Path to the image file |
| `gt_label` | `int` | 0 = normal, 1 = anomaly |
| `gt_mask_path` | `str` | Path to ground truth mask (empty string if none) |
| `cls_name` | `str` | Category name (e.g., `'bottle'`) |
| `defect_type` | `str` | Defect type name (e.g., `'broken_large'`, `'good'`) |

The data pipeline transforms these raw dicts into the tensor format that models expect. See the [Transforms tutorial](add_custom_transform.md) for pipeline details.

## Example: Adding a Custom Dataset

Create `baoiad/datasets/my_dataset.py`:

```python
"""My custom anomaly detection dataset."""

import os.path as osp
from typing import Dict, List

from baoiad.datasets.base_ad_dataset import BaseADDataset
from baoiad.registry import DATASETS


@DATASETS.register_module()
class MyCustomDataset(BaseADDataset):
    """Custom dataset for anomaly detection.

    Expected directory structure::

        data_root/
        ├── category_a/
        │   ├── train/
        │   │   └── good/
        │   │       ├── 000.png
        │   │       └── ...
        │   ├── test/
        │   │   ├── good/
        │   │   │   └── ...
        │   │   ├── defect_type_1/
        │   │   │   └── ...
        │   │   └── ...
        │   └── ground_truth/
        │       └── defect_type_1/
        │           ├── 000_mask.png
        │           └── ...
        ├── category_b/
        └── ...

    Args:
        data_root: Root directory of the dataset.
        split: 'train' or 'test'.
        cls_names: Categories to include. None for all.
        multi_class: Whether to load multiple categories together.
        pipeline: Data transform pipeline configs.
    """

    ALL_CATEGORIES = ('category_a', 'category_b', 'category_c')

    def load_data_list(self) -> List[Dict]:
        """Load annotation list.

        Returns:
            List of dicts with keys: img_path, gt_label, gt_mask_path,
            cls_name, defect_type.
        """
        data_list: List[Dict] = []

        for cls_name in self.cls_names:
            cls_dir = osp.join(self.data_root, cls_name, self.split)
            if not osp.isdir(cls_dir):
                continue

            gt_dir = osp.join(self.data_root, cls_name, 'ground_truth')

            for defect_type in sorted(os.listdir(cls_dir)):
                defect_dir = osp.join(cls_dir, defect_type)
                if not osp.isdir(defect_dir):
                    continue

                is_normal = (defect_type == 'good')
                gt_label = 0 if is_normal else 1

                for img_name in sorted(os.listdir(defect_dir)):
                    if not img_name.lower().endswith(('.png', '.jpg', '.bmp')):
                        continue

                    img_path = osp.join(defect_dir, img_name)

                    # Resolve mask path for defective test images
                    gt_mask_path = ''
                    if self.split == 'test' and not is_normal:
                        stem = osp.splitext(img_name)[0]
                        # Try _mask convention first
                        mask_name = f'{stem}_mask.png'
                        mask_path = osp.join(gt_dir, defect_type, mask_name)
                        if osp.exists(mask_path):
                            gt_mask_path = mask_path
                        else:
                            # Fallback: same stem
                            mask_name = f'{stem}.png'
                            mask_path = osp.join(gt_dir, defect_type, mask_name)
                            if osp.exists(mask_path):
                                gt_mask_path = mask_path

                    data_list.append(dict(
                        img_path=img_path,
                        gt_label=gt_label,
                        gt_mask_path=gt_mask_path,
                        cls_name=cls_name,
                        defect_type=defect_type,
                    ))

        return data_list
```

## Register the Dataset

Add to `baoiad/datasets/__init__.py`:

```python
from baoiad.datasets.my_dataset import MyCustomDataset  # noqa: F401
```

And add `'MyCustomDataset'` to the `__all__` list.

## Create a Dataset Config

Create `configs/_base_/datasets/my_dataset.py`:

```python
data_root = 'data/my_dataset'

train_pipeline = [
    dict(type='LoadImage'),
    dict(type='ResizeAD', size=256),
    dict(type='NormalizeAD'),
    dict(type='PackADInputs'),
]

test_pipeline = [
    dict(type='LoadImage'),
    dict(type='LoadMask'),
    dict(type='ResizeAD', size=256),
    dict(type='NormalizeAD'),
    dict(type='PackADInputs'),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='MyCustomDataset',
        data_root=data_root,
        split='train',
        pipeline=train_pipeline,
    ),
)

test_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MyCustomDataset',
        data_root=data_root,
        split='test',
        pipeline=test_pipeline,
    ),
)

val_dataloader = test_dataloader
```

## Data Flow

The pipeline processes each data dict through sequential transforms:

```
load_data_list()  →  [LoadImage, LoadMask, ResizeAD, NormalizeAD, PackADInputs]
                         |          |          |           |              |
                     img (HWC)  gt_mask    resized     normalized    ADDataSample
```

- **`LoadImage`**: Reads `img_path` → adds `img` (numpy HWC), `img_shape`
- **`LoadMask`**: Reads `gt_mask_path` → adds `gt_mask` (numpy HW)
- **`ResizeAD`**: Resizes `img` and `gt_mask` to target size
- **`NormalizeAD`**: Normalizes pixel values (mean/std)
- **`PackADInputs`**: Converts to `ADDataSample` with tensor inputs

See the [Transforms tutorial](add_custom_transform.md) for how to add custom pipeline stages.

## Multi-Class vs. Single-Class Mode

**Multi-class** (default): loads all categories (or specified `cls_names`) into one dataset instance. Used for benchmark-wide evaluation.

**Single-class** (`multi_class=False`): loads exactly one category. You must specify `cls_names`:

```python
dataset=dict(
    type='MyCustomDataset',
    data_root=data_root,
    split='train',
    cls_names=['category_a'],   # required when multi_class=False
    multi_class=False,
    pipeline=train_pipeline,
)
```

## Environment Variable

Set the dataset root via environment variable:

```bash
export BAOIAD_DATA_ROOT=/path/to/datasets
```

Or source the helper script:

```bash
source tools/env.sh
```

Then reference `$BAOIAD_DATA_ROOT/my_dataset` as `data_root` in configs.

## Existing Dataset Implementations

For reference, see these dataset implementations in `baoiad/datasets/`:

- `mvtec_ad.py` — MVTec AD (15 categories, good/defect splits, `_mask.png` convention)
- `visa.py` — VisA dataset
- `btech.py` — BTech dataset
- `draem_dataset.py` — DRAEM training dataset (augmented pairs with anomaly masks)
- `realiad.py` — Real-IAD dataset (multi-shot with category hierarchy)
