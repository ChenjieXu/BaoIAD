# Data Flow

This page traces a data sample from disk through the full BaoIAD pipeline: dataset → transforms → dataloader → model → predictions. Understanding this flow is essential for adding custom datasets or transforms.

## Pipeline Overview

```
Disk (images + annotations)
  ↓
BaseADDataset.load_data_list()     → list[dict]  (raw annotation records)
  ↓
Transform pipeline                 → dict          (processed data sample)
  ↓
DataLoader                         → dict          (batched inputs + data_samples)
  ↓
BaseADModel.forward(mode='predict') → list[ADDataSample]
  ↓
AnomalyDetectionMetric             → dict[str, float]
```

## Stage 1: Dataset — load_data_list()

All datasets inherit from `BaseADDataset` (in `baoiad/datasets/base_ad_dataset.py`), which extends `mmengine.dataset.BaseDataset`.

```python
from baoiad.datasets.base_ad_dataset import BaseADDataset
```

The dataset's `load_data_list()` method returns a list of dicts, where each dict represents one sample:

```python
def load_data_list(self) -> list[dict]:
    return [
        {
            'img_path': '/data/visa/pcb/good/001.JPG',
            'gt_mask_path': '',                     # Empty for normal images
            'gt_label': 0,                          # 0=normal, 1=anomaly
            'cls_name': 'pcb',
            'defect_type': 'good',
        },
        {
            'img_path': '/data/visa/pcb/bad/002.JPG',
            'gt_mask_path': '/data/visa/pcb/bad/002_mask.JPG',
            'gt_label': 1,
            'cls_name': 'pcb',
            'defect_type': 'crack',
        },
        # ...
    ]
```

### Dataset Construction

```python
class BaseADDataset(BaseDataset):
    def __init__(
        self,
        data_root: str,
        split: str = 'train',        # 'train' or 'test'
        cls_names: list[str] | None = None,
        multi_class: bool = True,
        pipeline: list[dict] | None = None,
        **kwargs,
    )
```

- `multi_class=True`: Loads all categories from `ALL_CATEGORIES` into one dataset.
- `multi_class=False`: Requires `cls_names` to be specified explicitly.

## Stage 2: Transform Pipeline

Each data dict passes through a sequence of transforms. Transforms are registered in the `TRANSFORMS` registry and configured as a list of dicts.

### Standard Pipeline (Feature-Based Methods)

```python
train_pipeline = [
    dict(type='LoadImage', to_float32=False, to_rgb=True),
    dict(type='LoadMask'),
    dict(type='ResizeAD', size=256),
    dict(type='NormalizeAD'),
    dict(type='PackADInputs'),
]

test_pipeline = [
    dict(type='LoadImage', to_float32=False, to_rgb=True),
    dict(type='LoadMask'),
    dict(type='ResizeAD', size=256),
    dict(type='NormalizeAD'),
    dict(type='PackADInputs'),
]
```

### Transform Dict Conventions

Data flows through transforms as a single `dict` with these keys:

#### Loading Transforms

**`LoadImage`** (`baoiad/datasets/transforms/loading.py`)

| Key | Direction | Type | Description |
|-----|-----------|------|-------------|
| `img_path` | Input | `str` | Path to image file |
| `img` | Output | `ndarray (H, W, 3)` | Image in uint8 RGB |
| `img_shape` | Output | `tuple(int, int)` | (H, W) |
| `ori_shape` | Output | `tuple(int, int)` | Original (H, W) |

Args: `color_type='color'`, `to_float32=False`, `to_rgb=True`, `keep_bgr_copy=False`, `backend='cv2'`.

**`LoadMask`** (`baoiad/datasets/transforms/loading.py`)

| Key | Direction | Type | Description |
|-----|-----------|------|-------------|
| `gt_mask_path` | Input | `str` | Path to mask file (empty for normal) |
| `gt_mask` | Output | `ndarray (H, W)` | Binary float32 mask (0.0 or 1.0) |

Args: `backend='cv2'`, `to_binary=True`.

#### Geometric Transforms

**`ResizeAD`** (`baoiad/datasets/transforms/augmentation.py`)

Resizes `img` and `gt_mask` to the target size. Supports `cv2` and `pillow` backends.

```python
dict(type='ResizeAD', size=256, backend='pillow')
```

**`CenterCrop`** (`baoiad/datasets/transforms/augmentation.py`)

Crops `img` and `gt_mask` from center.

```python
dict(type='CenterCrop', size=256)
```

**`RandomRotation`** (`baoiad/datasets/transforms/augmentation.py`)

Random rotation augmentation. Applied during training only.

```python
dict(type='RandomRotation', degrees=180.0)
```

#### Normalization Transforms

**`NormalizeAD`** (`baoiad/datasets/transforms/augmentation.py`)

Subtracts ImageNet mean and divides by ImageNet std (in uint8 scale):

```python
dict(type='NormalizeAD')  # default: ImageNet stats (123.675, 116.28, 103.53) / (58.395, 57.12, 57.375)
```

**`ScaleNormalizeAD`** (`baoiad/datasets/transforms/augmentation.py`)

Scales images to [0, 1] float32 (used by DRAEM and reconstruction methods that don't use ImageNet normalization):

```python
dict(type='ScaleNormalizeAD')
```

**`OpenCLIPPreprocessAD`** (`baoiad/datasets/transforms/augmentation.py`)

Full CLIP preprocessing: Resize → CenterCrop → ToTensor → Normalize with OpenAI stats.

```python
dict(type='OpenCLIPPreprocessAD', size=336)
```

#### Method-Specific Transforms

**`CFlowOfficialTransform`** (`baoiad/datasets/transforms/cflow.py`) — CFlow-AD's combined resize + normalize.

**`DeSTSegAugment`** / **`PackDeSTSegInputs`** (`baoiad/datasets/transforms/destseg.py`) — DeSTSeg-specific augmentation and packing.

#### Formatting Transforms

**`PackADInputs`** (`baoiad/datasets/transforms/formatting.py`)

The final transform that converts the dict into mmengine's expected format:

| Input key | Output |
|-----------|--------|
| `img` (HWC ndarray) | `inputs` (CHW Tensor, float32) |
| `gt_label` | `data_samples.gt_label` (int) |
| `gt_mask` (HW ndarray) | `data_samples.gt_mask` (Tensor) |
| `cls_name`, `img_path`, `defect_type` | `data_samples.set_metainfo({...})` |
| `support_imgs` (optional) | `data_samples.support_imgs` (Tensor) |

Output format:

```python
{
    'inputs': torch.Tensor,          # (C, H, W) float32
    'data_samples': ADDataSample,    # Contains gt_label, gt_mask, metainfo
}
```

**`PackDRAEMInputs`** (`baoiad/datasets/transforms/formatting.py`)

For DRAEM-style methods that need original + augmented image pairs:

```python
{
    'inputs': torch.Tensor,                    # Original image (C, H, W)
    'data_samples': ADDataSample,              # gt_label + metainfo with augmented_img and anomaly_mask
}
```

## Stage 3: DataLoader

MMEngine's DataLoader collates the list of dicts into batches:

```python
# Batch format (what the model receives):
{
    'inputs': torch.Tensor,                # (B, C, H, W)
    'data_samples': list[ADDataSample],    # Length B
}
```

## Stage 4: ADDataSample

`ADDataSample` (from `baoiad/structures/ad_data_sample.py`) extends `mmengine.structures.BaseDataElement`:

```python
from baoiad.structures import ADDataSample

sample = ADDataSample()
# Meta fields (set via set_metainfo):
sample.set_metainfo({
    'cls_name': 'bottle',
    'img_path': '/data/bottle/001.png',
    'defect_type': 'broken',
})

# Data fields (set as attributes):
sample.gt_label = 1                          # 0=normal, 1=anomaly
sample.gt_mask = torch.Tensor(H, W)          # Ground truth mask
sample.pred_score = 0.85                     # Predicted anomaly score
sample.pred_anomaly_map = torch.Tensor(1, H, W)  # Predicted anomaly map
```

### All Fields

| Field | Type | Set by | Description |
|-------|------|--------|-------------|
| `gt_label` | `int` | `PackADInputs` | Ground truth (0=normal, 1=anomaly) |
| `gt_mask` | `Tensor` | `PackADInputs` | Ground truth mask (H, W) |
| `pred_score` | `float` | model head | Predicted anomaly score |
| `pred_score_mean` | `float` | model head | Mean-pooled score (optional) |
| `pred_score_max` | `float` | model head | Max-pooled score (optional) |
| `pred_anomaly_map` | `Tensor` | model head | Predicted anomaly map (1, H, W) |
| `cls_name` | `str` | metainfo | Category name |
| `img_path` | `str` | metainfo | Image file path |
| `defect_type` | `str` | metainfo | Defect type name |

## Stage 5: Model → Predictions

During inference, the model head calls `build_predict_results()` to populate `ADDataSample` predictions:

```python
from baoiad.models.predict_utils import build_predict_results

results = build_predict_results(
    data_samples=data_samples,   # list[ADDataSample] with ground truth
    img_scores=scores,           # Tensor (B,) of anomaly scores
    score_maps=maps,             # Optional Tensor (B, 1, H, W)
)
# Each data_sample now has pred_score and pred_anomaly_map set
```

## Stage 6: Evaluation

The `AnomalyDetectionMetric` (see [Evaluation Guide](./evaluation.md)) iterates over the returned `ADDataSample` list, extracting `pred_score`, `pred_anomaly_map`, `gt_label`, and `gt_mask` for metric computation.

## Complete Example Config

```python
# A complete train/test pipeline for a feature-based method
train_dataloader = dict(
    batch_size=32,
    dataset=dict(
        type='VisADataset',
        data_root='data/visa',
        split='train',
        cls_names=['pcb'],
        pipeline=[
            dict(type='LoadImage', to_rgb=True),
            dict(type='LoadMask'),
            dict(type='ResizeAD', size=256),
            dict(type='NormalizeAD'),
            dict(type='PackADInputs'),
        ],
    ),
)
```
