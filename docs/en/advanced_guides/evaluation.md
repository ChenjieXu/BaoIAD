# Evaluation

BaoIAD uses `AnomalyDetectionMetric` to compute all evaluation metrics in a unified manner.

## AnomalyDetectionMetric

`AnomalyDetectionMetric` (`baoiad/evaluation/ad_metric.py`) computes both image-level and pixel-level metrics.

### Image-Level Metrics

| Metric | Key | Description |
|--------|-----|-------------|
| AUROC | `image_auroc` | Area Under the ROC Curve |
| F1-max | `image_f1max` | Maximum F1 score across all thresholds |
| Average Precision | `image_ap` | Area Under the Precision-Recall Curve |
| ECE | `image_ece` | Expected Calibration Error |
| FPR@95TPR | `image_fpr@95tpr` | False Positive Rate at 95% True Positive Rate |

### Pixel-Level Metrics

| Metric | Key | Description |
|--------|-----|-------------|
| AUROC | `pixel_auroc` | Area Under the ROC Curve (per-pixel) |
| F1-max | `pixel_f1max` | Maximum F1 score across all thresholds |
| Average Precision | `pixel_ap` | Area Under the Precision-Recall Curve |
| AUPRO | `aupro` | Area Under the Per-Region Overlap curve |
| AUPIMO | `aupimo` | Area Under the Per-Image Mean Overlap curve |
| ECE | `pixel_ece` | Expected Calibration Error (per-pixel) |

### Metric Logging

Metrics are logged in two formats:

- **Averaged**: `ad/<metric>: <value>` (mean across categories)
- **Per-category**: `ad/<category>/<metric>: <value>`

The `benchmark.py` tool parses the averaged format.

## AUPRO Details

AUPRO (Area Under the Per-Region Overlap curve) evaluates pixel-level detection quality by measuring overlap between predicted and ground-truth anomaly regions on a per-region basis. This is more informative than pixel AUROC for images with varying anomaly sizes.

- Computed by averaging the per-region overlap across all connected anomaly regions
- Normalized by the number of regions to avoid bias from large anomalous regions
- Range: [0, 1], higher is better

## AUPIMO Details

AUPIMO (Area Under the Per-Image Mean Overlap curve) extends AUPRO by computing overlap per-image rather than per-region:

- Measures the mean overlap between predicted and ground-truth masks per image
- Aggregates across images for the final score
- More robust to images with many small anomalous regions
- Range: [0, 1], higher is better

## Configuring Metrics

The default metric config is defined in the base dataset configs:

```python
test_evaluator = dict(type='AnomalyDetectionMetric')
val_evaluator = dict(type='AnomalyDetectionMetric')
```

### Selecting Specific Metrics

```python
test_evaluator = dict(
    type='AnomalyDetectionMetric',
    metrics=['image_auroc', 'pixel_auroc'],  # Only compute these
)
```

## Adding Custom Metrics

To add a new metric:

1. Implement the metric computation in `AnomalyDetectionMetric`
2. Or create a new `BaseMetric` subclass in `baoiad/evaluation/`
3. Register with `METRICS` registry

```python
from baoiad.registry import METRICS
from mmengine.evaluator import BaseMetric


@METRICS.register_module()
class MyMetric(BaseMetric):

    def process(self, data_samples, data_batch):
        # Store predictions and ground truths
        pass

    def compute_metrics(self, results):
        # Compute the metric
        return {'my_metric': value}
```