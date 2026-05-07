# Evaluation

BaoIAD uses `AnomalyDetectionMetric` (registered as `'AnomalyDetectionMetric'` in the `METRICS` registry) to compute all evaluation metrics in a unified manner.

## AnomalyDetectionMetric

**Module**: `baoiad/evaluation/ad_metric.py`
**Registry**: `METRICS`
**Default prefix**: `ad`

### Supported Metrics

#### Image-Level Metrics

| Metric Key | Description | Range |
|-----------|-------------|-------|
| `image_auroc` | Area Under the ROC Curve | [0, 1], higher is better |
| `image_auroc_mean` | AUROC using `pred_score_mean` field | [0, 1], higher is better |
| `image_auroc_max` | AUROC using `pred_score_max` field | [0, 1], higher is better |
| `image_f1max` | Maximum F1 score across all thresholds | [0, 1], higher is better |
| `image_ap` | Average Precision (area under PR curve) | [0, 1], higher is better |
| `image_ece` | Expected Calibration Error (requires calibrated probabilities) | [0, 1], lower is better |
| `image_fpr@95tpr` | False Positive Rate at 95% True Positive Rate | [0, 1], lower is better |

#### Pixel-Level Metrics

| Metric Key | Description | Range |
|-----------|-------------|-------|
| `pixel_auroc` | Area Under the ROC Curve (per-pixel) | [0, 1], higher is better |
| `pixel_f1max` | Maximum F1 score across all thresholds (per-pixel) | [0, 1], higher is better |
| `pixel_ap` | Average Precision (per-pixel) | [0, 1], higher is better |
| `aupro` | Area Under the Per-Region Overlap curve | [0, 1], higher is better |
| `aupimo` | Area Under the Per-Image Missed Overlap curve | [0, 1], higher is better |
| `pixel_ece` | Expected Calibration Error (per-pixel probability maps) | [0, 1], lower is better |

### Constructor Arguments

```python
test_evaluator = dict(
    type='AnomalyDetectionMetric',
    metrics=None,                    # None = all supported metrics
    resize_mask=None,                # Optional int or (H, W) to resize masks/maps
    resize_gt_mask_mode='nearest',   # 'nearest' or 'bilinear'
    resize_gt_mask_threshold=None,   # Binarization threshold after resize
    normalize_image_scores=False,    # Min-max normalize image scores
    normalize_pred_maps=False,       # None, 'per_image', or 'batch_broadcast'
    flip_auroc_if_below_half=False,  # Flip AUROC if < 0.5
    image_score_field='pred_score',  # 'pred_score', 'pred_score_mean', or 'pred_score_max'
)
```

### Metric Logging

Metrics are logged in two formats:

- **Averaged**: `ad/image_auroc: 0.956` (mean across all categories)
- **Per-category**: `ad/bottle/image_auroc: 0.982`, `ad/cable/image_auroc: 0.930`

The `benchmark.py` tool parses the averaged format for benchmark tables.

## Metric Implementations

### AUROC

Computed via `sklearn.metrics.roc_auc_score`. Image-level uses `gt_label` vs `pred_score`; pixel-level flattens all `gt_mask` and `pred_anomaly_map` values across the dataset.

When `flip_auroc_if_below_half=True`, scores below 0.5 are inverted (useful when anomaly direction is unknown).

### F1-max

Computes the precision-recall curve via `sklearn.metrics.precision_recall_curve`, then finds the threshold that maximizes `F1 = 2·P·R / (P+R)`.

### AP (Average Precision)

Computed via `sklearn.metrics.average_precision_score`. Equal to the area under the precision-recall curve.

### ECE (Expected Calibration Error)

**Module**: `baoiad/evaluation/ece.py`

ECE measures how well predicted probabilities match actual accuracy. It bins predictions into `n_bins=15` equal-width bins and computes:

```
ECE = Σ (n_b / N) · |acc_b - conf_b|
```

where `n_b` is the count in bin `b`, `acc_b` is the average accuracy, and `conf_b` is the average predicted probability.

**Important**: ECE expects `pred_score` values already in [0, 1] (calibrated probabilities). Raw anomaly scores must be normalized before ECE.

Pixel-level ECE (`pixel_ece`) flattens all masks/maps and computes the same metric.

### FPR@95TPR

**Module**: `baoiad/evaluation/fpr_at_tpr.py`

Computes the false positive rate at a target true positive rate (default 95%). Finds the first point on the ROC curve where TPR ≥ target and returns the corresponding FPR.

```python
fpr, tpr, _ = roc_curve(gt_labels, pred_scores)
idx = np.where(tpr >= 0.95)[0]
return fpr[idx[0]]
```

### AUPRO (Area Under Per-Region Overlap)

**Module**: `baoiad/evaluation/aupro.py`

AUPRO evaluates pixel-level detection quality on a per-region basis:

1. Run connected component analysis on ground truth masks to identify individual anomaly regions.
2. Offset region labels across the batch so they are globally unique.
3. For each pixel sorted by descending prediction score, accumulate per-region overlap (PRO) and false positive rate.
4. Integrate the PRO-vs-FPR curve up to `max_fpr=0.3` and normalize.

This metric is more informative than pixel AUROC for images with varying anomaly sizes because it weights each region equally regardless of size.

### AUPIMO (Area Under Per-Image Missed Overlap)

**Module**: `baoiad/evaluation/aupimo.py`

AUPIMO follows the official BMVC 2024 paper definition:

1. Identify normal images (all-zero gt_mask) and anomalous images.
2. Build a threshold grid from normal-image pixel scores only.
3. Compute shared FPR (F_sh) — the average FPR across normal images at each threshold.
4. For each anomalous image, compute per-image TPR (T_i) — the fraction of anomalous pixels detected.
5. Integrate each per-image PIMO curve in log(F_sh) space over `fpr_bounds=(1e-5, 1e-4)`.
6. Return the average of per-image AUPIMO values.

Default parameters: `fpr_bounds=(1e-5, 1e-4)`, `num_thresholds=300`.

A `compute_pimo_per_image()` function is also available for per-image analysis.

## Configuring Metrics

### Default (All Metrics)

```python
test_evaluator = dict(type='AnomalyDetectionMetric')
val_evaluator = dict(type='AnomalyDetectionMetric')
```

### Selecting Specific Metrics

```python
test_evaluator = dict(
    type='AnomalyDetectionMetric',
    metrics=['image_auroc', 'pixel_auroc'],
)
```

### Score Normalization

For methods that produce uncalibrated scores, enable normalization:

```python
test_evaluator = dict(
    type='AnomalyDetectionMetric',
    normalize_image_scores=True,          # Min-max normalize image scores per class
    normalize_pred_maps='per_image',      # Normalize anomaly maps per image
)
```

`normalize_pred_maps` options:
- `None` / `False` — No normalization
- `'per_image'` — Min-max normalize each image's map independently
- `'batch_broadcast'` — Average normalized maps across the batch

### Selecting Image Score Field

Some methods produce multiple score variants. Select which one to use for image-level metrics:

```python
test_evaluator = dict(
    type='AnomalyDetectionMetric',
    image_score_field='pred_score_max',  # Use max-pooled score instead of default
)
```

Available fields: `'pred_score'` (default), `'pred_score_mean'`, `'pred_score_max'`.

## Per-Category Breakdown

`AnomalyDetectionMetric.compute_metrics()` groups results by `cls_name` and computes metrics for each category independently, then averages:

```python
# Internal flow:
grouped = defaultdict(list)  # cls_name → list of results
for r in results:
    grouped[r['cls_name']].append(r)

per_class = {cls: compute_single_class(samples) for cls, samples in grouped.items()}
averaged = {metric: mean(per_class[cls][metric] for cls in per_class) for metric in metrics}
```

Both averaged and per-category results appear in the logged output.

## Adding a Custom Metric

To create a new metric:

1. Subclass `mmengine.evaluator.BaseMetric`.
2. Register with the `METRICS` registry.
3. Implement `process()` (collect per-batch results) and `compute_metrics()` (compute final scores).

```python
from mmengine.evaluator import BaseMetric
from baoiad.registry import METRICS


@METRICS.register_module()
class MyCustomMetric(BaseMetric):
    default_prefix = 'custom'

    def process(self, data_batch, data_samples):
        for sample in data_samples:
            pred_score = float(sample.pred_score)
            gt_label = int(sample.gt_label)
            self.results.append(dict(pred_score=pred_score, gt_label=gt_label))

    def compute_metrics(self, results):
        # Compute your metric from accumulated results
        return {'my_metric': value}
```

Use in config:

```python
test_evaluator = dict(type='MyCustomMetric')
```
