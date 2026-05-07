# Add a Custom Transform

Transforms in BaoIAD process data dicts through a pipeline. Each transform reads some keys from the input dict, performs an operation, and writes results back. This tutorial shows how to create and register a custom transform.

## Transform Interface

All transforms must:

1. Inherit from `mmcv.transforms.BaseTransform`
2. Be registered with `@TRANSFORMS.register_module()`
3. Implement `transform(self, results: Dict) -> Dict`

The `results` dict carries all intermediate data through the pipeline. Transforms read keys they need and write new keys.

## Available Transforms

BaoIAD ships with these transforms in `baoiad/datasets/transforms/`:

| Class | Module | Description |
|---|---|---|
| `LoadImage` | `loading.py` | Load image from `img_path` |
| `LoadMask` | `loading.py` | Load GT mask from `gt_mask_path` |
| `ResizeAD` | `augmentation.py` | Resize image (and mask if present) |
| `NormalizeAD` | `augmentation.py` | Normalize pixel values |
| `ScaleNormalizeAD` | `augmentation.py` | Scale-based normalization |
| `OpenCLIPPreprocessAD` | `augmentation.py` | OpenCLIP-specific preprocessing |
| `CenterCrop` | `augmentation.py` | Center crop |
| `RandomRotation` | `augmentation.py` | Random rotation augmentation |
| `ThresholdMask` | `augmentation.py` | Binarize mask by threshold |
| `PackADInputs` | `formatting.py` | Pack into `ADDataSample` tensors |
| `PackDRAEMInputs` | `formatting.py` | Pack DRAEM-specific inputs |
| `CFlowOfficialTransform` | `cflow.py` | CFlow-specific augmentation |
| `DeSTSegAugment` | `destseg.py` | DeSTSeg augmentation |
| `PackDeSTSegInputs` | `destseg.py` | Pack DeSTSeg-specific inputs |

## Data Dict Convention

Keys in the `results` dict follow a consistent naming convention:

| Key | Type | Set by | Description |
|---|---|---|---|
| `img_path` | `str` | dataset | Path to the image file |
| `gt_mask_path` | `str` | dataset | Path to the GT mask |
| `img` | `ndarray (H, W, C)` | `LoadImage` | Image pixels |
| `img_shape` | `tuple (H, W)` | `LoadImage` | Image spatial shape |
| `ori_shape` | `tuple (H, W)` | `LoadImage` | Original shape before resize |
| `gt_mask` | `ndarray (H, W)` | `LoadMask` | Ground truth mask |
| `gt_label` | `int` | dataset | 0=normal, 1=anomaly |
| `cls_name` | `str` | dataset | Category name |
| `defect_type` | `str` | dataset | Defect type |

After `PackADInputs`, the dict becomes:

```python
{
    'inputs': Tensor (C, H, W),          # image tensor
    'data_samples': ADDataSample,         # with gt_label, gt_mask, metainfo
}
```

## Example: Random Erasing Transform

Create `baoiad/datasets/transforms/random_erase.py`:

```python
"""Random erasing transform for anomaly detection augmentation."""

import random
from typing import Dict

import numpy as np
from mmcv.transforms import BaseTransform

from baoiad.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RandomEraseAD(BaseTransform):
    """Randomly erase a rectangular region of the image.

    Useful for simulating occlusions during training of reconstruction-based
    methods.

    Args:
        erase_prob: Probability of applying the transform.
        erase_scale_range: (min, max) fraction of image area to erase.
        erase_ratio_range: (min, max) aspect ratio of the erased region.
        fill_value: Pixel value to fill the erased region with.
    """

    def __init__(
        self,
        erase_prob: float = 0.5,
        erase_scale_range: tuple = (0.02, 0.15),
        erase_ratio_range: tuple = (0.3, 3.3),
        fill_value: int = 0,
    ):
        self.erase_prob = erase_prob
        self.erase_scale_range = erase_scale_range
        self.erase_ratio_range = erase_ratio_range
        self.fill_value = fill_value

    def transform(self, results: Dict) -> Dict:
        """Apply random erasing.

        Args:
            results: Dict containing at least 'img' (H, W, C) ndarray.

        Returns:
            Updated results dict.
        """
        if random.random() > self.erase_prob:
            return results

        img = results['img']
        h, w = img.shape[:2]
        area = h * w

        # Sample erase region
        target_area = random.uniform(*self.erase_scale_range) * area
        aspect = random.uniform(*self.erase_ratio_range)

        erase_w = int(round(np.sqrt(target_area * aspect)))
        erase_h = int(round(np.sqrt(target_area / aspect)))

        erase_w = min(erase_w, w)
        erase_h = min(erase_h, h)

        if erase_w < 1 or erase_h < 1:
            return results

        # Random top-left corner
        top = random.randint(0, h - erase_h)
        left = random.randint(0, w - erase_w)

        img = img.copy()
        img[top:top + erase_h, left:left + erase_w] = self.fill_value
        results['img'] = img

        # Also erase the corresponding mask region if present
        if 'gt_mask' in results and results['gt_mask'] is not None:
            mask = results['gt_mask'].copy()
            mask[top:top + erase_h, left:left + erase_w] = 0
            results['gt_mask'] = mask

        return results
```

## Register the Transform

Add to `baoiad/datasets/transforms/__init__.py`:

```python
from .random_erase import RandomEraseAD  # noqa: F401
```

And add `'RandomEraseAD'` to the `__all__` list.

## Using the Transform in a Pipeline

Insert the transform into the pipeline config between the appropriate stages:

```python
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='ResizeAD', size=256),
    dict(type='RandomEraseAD', erase_prob=0.5, fill_value=0),
    dict(type='NormalizeAD'),
    dict(type='PackADInputs'),
]
```

The transform will be applied to every training sample. It is skipped for test pipelines unless you explicitly include it.

## Example: Custom Formatting Transform

Some methods require a specialized input format. You can create a custom pack transform similar to `PackDRAEMInputs`:

```python
@TRANSFORMS.register_module()
class PackMyMethodInputs(BaseTransform):
    """Pack inputs with additional custom fields."""

    def transform(self, results: Dict) -> Dict:
        import torch
        from baoiad.structures import ADDataSample

        img = results['img']
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img.transpose(2, 0, 1)).contiguous().float()

        data_sample = ADDataSample()
        data_sample.set_metainfo({
            'cls_name': results.get('cls_name', ''),
            'img_path': results.get('img_path', ''),
            'defect_type': results.get('defect_type', ''),
        })
        data_sample.gt_label = results.get('gt_label', 0)

        # Add method-specific fields
        if 'augmented_img' in results:
            aug = results['augmented_img']
            if isinstance(aug, np.ndarray):
                aug = torch.from_numpy(aug.transpose(2, 0, 1)).contiguous().float()
            data_sample.set_metainfo({'augmented_img': aug})

        return dict(inputs=img, data_samples=data_sample)
```

## Key Points

- **Read before write**: Always check that required keys exist in `results` before accessing them. Use `results.get(key, default)` for optional keys.
- **Copy before mutation**: If you modify arrays in-place (e.g., erasing pixels), copy them first to avoid side effects on other transforms.
- **Pipeline ordering**: Place transforms in the correct order — loading before resizing before normalization before packing. The pack transform must always come last.
- **Scope**: Register with `TRANSFORMS` from `baoiad.registry` (not `mmengine.registry` directly) to use the `baoiad` scope.
