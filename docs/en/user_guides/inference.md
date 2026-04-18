# Inference

## Running Inference

BaoIAD's inference is performed through the `predict()` mode of each detector, which is automatically invoked during testing.

### Using tools/test.py

The standard way to run inference on a test set:

```bash
python tools/test.py configs/patchcore/patchcore_wrn50_256_mvtec_strict.py \
    runs/patchcore/best.pth
```

This loads the model checkpoint, runs `predict()` on all test images, and computes evaluation metrics.

### Programmatic Inference

For custom inference pipelines, you can load a model and run prediction directly:

```python
import baoiad  # Must import first to trigger registry
from mmengine.config import Config
from mmengine.runner import Runner

# Load config and build model
cfg = Config.fromfile('configs/patchcore/patchcore_wrn50_256_mvtec_strict.py')
runner = Runner.from_cfg(cfg)

# Load checkpoint
runner.load_checkpoint('runs/patchcore/best.pth')

# Run test (predict on all test data)
runner.test()
```

### Loading a Model for Single-Image Inference

```python
import baoiad
import torch
from mmengine.config import Config

# Load config
cfg = Config.fromfile('configs/patchcore/patchcore_wrn50_256_mvtec_strict.py')

# Build model from config
from baoiad.registry import MODELS
model = MODELS.build(cfg.model)

# Load checkpoint weights
checkpoint = torch.load('runs/patchcore/best.pth', map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Run prediction (requires proper data preprocessing)
with torch.no_grad():
    # Preprocess image (resize, normalize, etc.)
    # processed = preprocess(raw_image)
    # results = model.predict(processed)
    pass
```

:::{note}
For memory bank methods (PatchCore, SPADE, PaDiM, etc.), you must call `model.build_memory_bank()` before `predict()` works correctly. This is handled automatically by `MemoryBankHook` during normal training/testing.
:::

## Interpreting Results

### ADDataSample Fields

Each prediction produces an `ADDataSample` with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `gt_label` | int | Ground-truth label (0=normal, 1=anomaly) |
| `gt_mask` | Tensor (H, W) | Ground-truth pixel mask |
| `cls_name` | str | Product category name |
| `img_path` | str | Path to the input image |
| `defect_type` | str | Defect type name |
| `pred_score` | float | Predicted anomaly score (higher = more anomalous) |
| `pred_anomaly_map` | Tensor (1, H, W) | Predicted pixel-level anomaly map |

### Anomaly Scores

- **Image-level**: `pred_score` is a single float per image. Threshold this to classify images as normal/anomalous.
- **Pixel-level**: `pred_anomaly_map` is a per-pixel score map. Threshold this to obtain a binary anomaly mask.

### Threshold Selection

The optimal threshold can be found from the F1-max metric, which computes the best F1 score across all possible thresholds on the test set.

## Visualization

Enable visualization hook in the config:

```python
default_hooks = dict(
    visualization=dict(type='ADVisualizationHook', enable=True),
)
```

Or override from command line:

```bash
python tools/test.py <config> <checkpoint> \
    --cfg-options default_hooks.visualization.enable=True
```
