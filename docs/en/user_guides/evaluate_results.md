# Evaluate Results

## Metrics

BaoIAD computes metrics through [`AnomalyDetectionMetric`](../../../baoiad/evaluation/ad_metric.py), which evaluates predictions at both image and pixel levels. Metrics are computed per category and then averaged.

### Image-Level Metrics

| Key | Name | Range | Direction | Description |
|---|---|---|---|---|
| `ad/image_auroc` | Image AUROC | [0, 1] | Higher is better | Area Under the ROC Curve for image-level anomaly classification. |
| `ad/image_auroc_mean` | Image AUROC (mean score) | [0, 1] | Higher is better | AUROC using the mean of per-pixel scores as the image score. |
| `ad/image_auroc_max` | Image AUROC (max score) | [0, 1] | Higher is better | AUROC using the max of per-pixel scores as the image score. |
| `ad/image_f1max` | Image F1-max | [0, 1] | Higher is better | Maximum F1 score over all possible thresholds on the precision-recall curve. |
| `ad/image_ap` | Image AP | [0, 1] | Higher is better | Average Precision (area under the precision-recall curve). |
| `ad/image_ece` | Image ECE | [0, 1] | Lower is better | Expected Calibration Error. Measures how well predicted probabilities match actual accuracy. Requires scores in [0, 1]. |
| `ad/image_fpr@95tpr` | Image FPR@95TPR | [0, 1] | Lower is better | False Positive Rate when True Positive Rate is at least 95%. |

### Pixel-Level Metrics

| Key | Name | Range | Direction | Description |
|---|---|---|---|---|
| `ad/pixel_auroc` | Pixel AUROC | [0, 1] | Higher is better | AUROC for pixel-level anomaly segmentation. |
| `ad/pixel_f1max` | Pixel F1-max | [0, 1] | Higher is better | Maximum F1 score for pixel-level predictions. |
| `ad/pixel_ap` | Pixel AP | [0, 1] | Higher is better | Average Precision for pixel-level predictions. |
| `ad/aupro` | AUPRO | [0, 1] | Higher is better | Area Under the Per-Region Overlap curve, computed up to 30% FPR. Measures detection quality at the connected-component level. |
| `ad/aupimo` | AUPIMO | [0, 1] | Higher is better | Area Under the Per-Image Missed Overlap curve. Integrates per-image TPR over log-scaled FPR in the [1e-5, 1e-4] range. More sensitive to low-FPR performance. |
| `ad/pixel_ece` | Pixel ECE | [0, 1] | Lower is better | Expected Calibration Error for pixel-level probability maps. |

### Metric Directionality

For all **higher-is-better** metrics (AUROC, F1-max, AP, AUPRO, AUPIMO), a value of 1.0 indicates perfect performance and 0.5 (for AUROC) indicates random guessing.

For **lower-is-better** metrics (ECE, FPR@95TPR), a value of 0.0 indicates perfect performance. Lower values mean better calibration (ECE) or fewer false positives at high recall (FPR@95TPR).

## Metric Computation Details

### AUPRO

Computed in [`baoiad/evaluation/aupro.py`](../../../baoiad/evaluation/aupro.py). Uses a global-sorting method aligned with anomalib:
1. Connected-component analysis on ground truth masks to identify anomaly regions.
2. Global sorting of pixels by anomaly score (descending).
3. Cumulative FPR and per-region overlap (PRO) computation.
4. AUC integration up to `max_fpr=0.3`, normalized to [0, 1].

### AUPIMO

Computed in [`baoiad/evaluation/aupimo.py`](../../../baoiad/evaluation/aupimo.py). Follows the BMVC 2024 paper definition:
1. FPR thresholds computed from **normal images only** (not all normal pixels).
2. Per-image TPR curves for each anomalous image.
3. Integration in log-FPR space over the range [1e-5, 1e-4] (default), using 300 thresholds.
4. Returns the average of per-image AUPIMO values.

### ECE

Computed in [`baoiad/evaluation/ece.py`](../../../baoiad/evaluation/ece.py). Uses 15 equal-width bins:
- **Important**: ECE assumes `pred_scores` are already calibrated probabilities in [0, 1]. If your model outputs uncalibrated scores, ECE will raise an error or produce misleading results.
- The metric does **not** rescale arbitrary anomaly scores, because per-batch min-max normalization changes the probability semantics.

### FPR@95TPR

Computed in [`baoiad/evaluation/fpr_at_tpr.py`](../../../baoiad/evaluation/fpr_at_tpr.py). Finds the minimum FPR at which TPR >= 95% from the ROC curve. Returns 1.0 if 95% TPR is unachievable.

## Where to Find Results

### In Training/Testing Logs

Metrics are logged with the prefix `ad/`. Look for lines like:

```
Epoch [10][50/100]  ad/image_auroc: 0.9523  ad/pixel_auroc: 0.9712  ad/aupro: 0.8934
```

In the JSON log file (`<work_dir>/<timestamp>.log.json`), each entry includes metric key-value pairs.

### Per-Class Breakdown

`AnomalyDetectionMetric` also computes per-class metrics, logged as:

```
ad/bottle/image_auroc: 0.9812
ad/bottle/pixel_auroc: 0.9934
ad/hazelnut/image_auroc: 0.9234
...
```

The averaged (cross-category) metrics are logged without a class prefix.

### Benchmark Output JSON

When running benchmarks via `tools/benchmark.py`, results are saved to a JSON file (default: `results/benchmark.json`) with the structure:

```json
{
  "patchcore": {
    "bottle": {
      "image_auroc": 0.9812,
      "pixel_auroc": 0.9934,
      "image_f1max": 0.9678,
      "image_ap": 0.9845,
      "aupro": 0.9234,
      "aupimo": 0.8156,
      "image_ece": 0.0312,
      "pixel_ece": 0.0456,
      "image_fpr@95tpr": 0.0612,
      "time": 12.3
    },
    "_average": {
      "image_auroc": 0.9654,
      "pixel_auroc": 0.9789,
      "image_f1max": 0.9543,
      "pixel_f1max": 0.6412,
      "num_categories": 15
    }
  }
}
```

Each method has per-category results and an `_average` entry with cross-category means. The `num_categories` field indicates how many categories contributed to the average.

## Customizing Metrics

To select a subset of metrics, configure the `test_evaluator` in your config:

```python
test_evaluator = dict(
    type='AnomalyDetectionMetric',
    metrics=['image_auroc', 'pixel_auroc', 'aupro'],
)
```

Available metric names match the keys listed above (without the `ad/` prefix): `image_auroc`, `pixel_auroc`, `image_auroc_mean`, `image_auroc_max`, `image_f1max`, `pixel_f1max`, `image_ap`, `pixel_ap`, `aupro`, `aupimo`, `image_ece`, `pixel_ece`, `image_fpr@95tpr`.
