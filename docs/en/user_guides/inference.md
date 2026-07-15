# Inference

## Running Inference

BaoIAD does not provide a dedicated single-image inference CLI. Inference is done through `tools/test.py`, which evaluates a trained model on a full test set defined by the config's `test_dataloader`.

### Evaluate on a Test Set

```bash
python tools/test.py <config> <checkpoint> --work-dir runs/<name>
```

The test script loads the model from the checkpoint, runs forward passes on all images in the test dataloader, and computes metrics via [`AnomalyDetectionMetric`](../../../baoiad/evaluation/ad_metric.py).

### Single-Image Inference (Python API)

For single-image or batch inference outside of the test runner, use the Python API directly:

```python
import torch
from baoiad import register_all_modules
from mmengine.config import Config
from baoiad.registry import MODELS
from mmengine.runner import Runner

# Explicit registration is recommended for application entry points. BaoIAD's
# registries also have lazy import locations, so Registry.get/build remains
# compatible with older programmatic callers that did not call this function.
register_all_modules()

# Load config and build model
cfg = Config.fromfile('configs/patchcore/patchcore_wrn50_256_mvtec_strict.py')
model = MODELS.build(cfg.model).cuda().eval()

# For memory-bank methods, build the memory bank first
train_loader = Runner.build_dataloader(cfg.train_dataloader)
model.train()
with torch.no_grad():
    for batch in train_loader:
        inputs = batch['inputs'].cuda()
        model(inputs, batch.get('data_samples', []), mode='loss')
model.build_memory_bank()
model.eval()

# Run inference on a single batch
test_loader = Runner.build_dataloader(cfg.test_dataloader)
with torch.no_grad():
    for batch in test_loader:
        inputs = batch['inputs'].cuda()
        results = model(inputs, batch.get('data_samples', []), mode='predict')
        for result in results:
            score = float(result.pred_score)
            anomaly_map = result.pred_anomaly_map.cpu().numpy()
            print(f"Score: {score:.4f}")
        break  # first batch only
```

## Understanding Output Predictions

Each prediction result (a `DataSample` object) contains:

| Field | Type | Description |
|---|---|---|
| `pred_score` | `float` | Image-level anomaly score. Higher values indicate greater anomaly likelihood. |
| `pred_score_mean` | `float` | Mean of per-pixel anomaly scores (may differ from `pred_score` for some methods). |
| `pred_score_max` | `float` | Max of per-pixel anomaly scores. |
| `pred_anomaly_map` | `ndarray` (H, W) | Pixel-level anomaly score map. Higher values indicate greater anomaly likelihood at that location. |
| `pred_label` | `int` | Predicted label (0 = normal, 1 = anomaly), derived by thresholding `pred_score`. |
| `gt_label` | `int` | Ground truth label. |
| `gt_mask` | `ndarray` (H, W) | Ground truth pixel-level mask (0 = normal, >0 = anomalous). |

### Interpreting Anomaly Scores

Anomaly scores are **not calibrated probabilities** by default. Their scale varies by method:

- **Memory-bank methods** (PatchCore, PaDiM): scores are distances in feature space.
- **Normalizing flow methods** (CFlow, FastFlow): scores are negative log-likelihoods.
- **Reconstruction methods** (DRAEM, MemSeg): scores are reconstruction errors.
- **Knowledge distillation methods** (RD, EfficientAD): scores are feature divergence measures.

To compare scores across methods, use the AUROC metric (which is rank-based and scale-invariant) rather than raw score values.

### Anomaly Maps

The `pred_anomaly_map` is a 2D array with the same spatial dimensions as the input image (or the config-specified output size). Values are raw anomaly scores — to visualize them as a heatmap, normalize to [0, 1] and apply a colormap. See [Visualization](visualization.md) for built-in visualization tools.
