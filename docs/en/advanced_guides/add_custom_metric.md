# Add a Custom Metric

BaoIAD evaluates anomaly detection methods using `AnomalyDetectionMetric`, registered in the `METRICS` registry. This tutorial shows how to add a custom evaluation metric.

## Built-in Metrics

`AnomalyDetectionMetric` (`baoiad/evaluation/ad_metric.py`) computes these metrics:

| Metric name | Level | Description |
|---|---|---|
| `image_auroc` | Image | Area under ROC curve |
| `pixel_auroc` | Pixel | Area under ROC curve |
| `image_f1max` | Image | Maximum F1 score |
| `pixel_f1max` | Pixel | Maximum F1 score |
| `image_ap` | Image | Average precision |
| `pixel_ap` | Pixel | Average precision |
| `aupro` | Pixel | Area under per-region overlap |
| `aupimo` | Pixel | Area under per-instance mean overlap |
| `image_ece` | Image | Expected calibration error |
| `pixel_ece` | Pixel | Expected calibration error |
| `image_fpr@95tpr` | Image | FPR at 95% TPR |

## Metric Interface

Custom metrics must:

1. Inherit from `mmengine.evaluator.BaseMetric`
2. Be registered with `@METRICS.register_module()`
3. Implement `process(self, data_batch, data_samples)` — collect per-batch results
4. Implement `compute_metrics(self, results)` — aggregate and return final values

## Example: Top-K Accuracy Metric

Create `baoiad/evaluation/topk_metric.py`:

```python
"""Top-K image-level accuracy metric for anomaly detection."""

from typing import Dict, List, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric

from baoiad.registry import METRICS


@METRICS.register_module()
class TopKAccuracyMetric(BaseMetric):
    """Compute top-K accuracy: fraction of the K most anomalous images
    that are truly anomalous.

    Useful for scenarios where only the top-K suspicious images are
    reviewed by a human inspector.

    Args:
        k: Number of top-scored images to check.
        image_score_field: Which score field to use from the data sample.
    """

    default_prefix = 'ad'

    def __init__(
        self,
        k: int = 10,
        image_score_field: str = 'pred_score',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.k = k
        self.image_score_field = image_score_field

    def process(self, data_batch: dict, data_samples: Sequence) -> None:
        """Collect predictions from each batch.

        Args:
            data_batch: Raw data batch from the dataloader.
            data_samples: List of data samples with predictions.
        """
        for sample in data_samples:
            pred_score = float(
                sample[self.image_score_field]
                if isinstance(sample, dict)
                else getattr(sample, self.image_score_field)
            )
            gt_label = int(
                sample['gt_label']
                if isinstance(sample, dict)
                else getattr(sample, 'gt_label')
            )
            cls_name = str(
                sample.get('cls_name', '')
                if isinstance(sample, dict)
                else getattr(sample, 'cls_name', '')
            )
            self.results.append(dict(
                pred_score=pred_score,
                gt_label=gt_label,
                cls_name=cls_name,
            ))

    def compute_metrics(self, results: List[Dict]) -> Dict:
        """Compute top-K accuracy, per-class and averaged.

        Args:
            results: List of collected result dicts.

        Returns:
            Dict mapping metric names to values.
        """
        from collections import defaultdict
        grouped = defaultdict(list)
        for r in results:
            grouped[r['cls_name']].append(r)

        out = {}
        precisions = []
        for cls_name, samples in grouped.items():
            sorted_samples = sorted(
                samples, key=lambda x: x['pred_score'], reverse=True)
            top_k = sorted_samples[:self.k]
            hits = sum(1 for s in top_k if s['gt_label'] == 1)
            precision = hits / min(self.k, len(top_k))
            out[f'{cls_name}/top{self.k}_precision'] = precision
            precisions.append(precision)

        out[f'top{self.k}_precision'] = (
            float(np.mean(precisions)) if precisions else 0.0)
        return out
```

## Register the Metric

Add to `baoiad/evaluation/__init__.py`:

```python
from .topk_metric import TopKAccuracyMetric  # noqa: F401
```

## Configure in a Config

Use the metric in your test/val evaluator:

```python
test_evaluator = dict(
    type='TopKAccuracyMetric',
    k=10,
    image_score_field='pred_score',
)
val_evaluator = test_evaluator
```

You can also combine multiple metrics by using a list:

```python
test_evaluator = [
    dict(type='AnomalyDetectionMetric', metrics=['image_auroc', 'pixel_auroc']),
    dict(type='TopKAccuracyMetric', k=10),
]
```

## How AnomalyDetectionMetric Works

The built-in metric follows the same `process` → `compute_metrics` pattern:

1. **`process()`**: Called after each test batch. Extracts `pred_score`, `gt_label`, `pred_anomaly_map`, `gt_mask`, and `cls_name` from each `ADDataSample`. Handles tensor → numpy conversion, mask resizing, and score alignment. Appends flattened results to `self.results`.

2. **`compute_metrics()`**: Groups results by `cls_name`. For each class, computes all requested metrics. Returns both per-class values (`{cls_name}/{metric}`) and averaged values (`{metric}`).

The metric supports several configuration options:

```python
test_evaluator = dict(
    type='AnomalyDetectionMetric',
    metrics=['image_auroc', 'pixel_auroc', 'aupro'],  # subset of metrics
    resize_mask=256,                                    # resize masks to common size
    resize_gt_mask_mode='nearest',                      # or 'bilinear'
    normalize_image_scores=True,                        # min-max normalize scores
    normalize_pred_maps='per_image',                    # or 'batch_broadcast' or None
    flip_auroc_if_below_half=True,                      # invert AUROC < 0.5
    image_score_field='pred_score',                     # or 'pred_score_mean', 'pred_score_max'
)
```

## Accessing Data Sample Fields

In `process()`, data samples may be `ADDataSample` objects (attribute access) or dicts (key access). Handle both:

```python
def _get(sample, key):
    return sample[key] if isinstance(sample, dict) else getattr(sample, key)

def _has(sample, key):
    return key in sample if isinstance(sample, dict) else hasattr(sample, key)
```

This pattern is used throughout `AnomalyDetectionMetric.process()`.

## Key Points

- **`default_prefix`**: Set this on your metric class to namespace results in logs (e.g., `ad/image_auroc`).
- **Per-class averaging**: Follow the convention of computing per-class metrics first, then averaging. Store both in the output dict.
- **Robustness**: Handle edge cases like single-class batches (no AUROC possible) and empty results gracefully.
