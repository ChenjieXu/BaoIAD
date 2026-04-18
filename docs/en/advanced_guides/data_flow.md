# Data Flow

This page documents how data flows through BaoIAD, from raw images to model predictions.

## ADDataSample

`ADDataSample` (in `baoiad/structures/`) extends `mmengine.structures.BaseDataElement` and carries all per-image data through the pipeline.

### Fields

| Field | Type | Set By | Description |
|-------|------|--------|-------------|
| `gt_label` | int | Dataset | Ground-truth label (0=normal, 1=anomaly) |
| `gt_mask` | Tensor (H, W) | Dataset | Ground-truth pixel mask (binary) |
| `cls_name` | str | Dataset | Product category name |
| `img_path` | str | Dataset | Path to input image |
| `defect_type` | str | Dataset | Defect type name (e.g., 'broken_large') |
| `pred_score` | float | Detector `predict()` | Predicted image-level anomaly score |
| `pred_anomaly_map` | Tensor (1, H, W) | Detector `predict()` | Predicted pixel-level anomaly map |

### Usage

```python
from baoiad.structures import ADDataSample

sample = ADDataSample()
sample.gt_label = 1
sample.gt_mask = torch.zeros(256, 256)
sample.cls_name = 'bottle'
sample.img_path = '/path/to/image.png'
sample.defect_type = 'broken_large'

# After prediction
sample.pred_score = 0.8723
sample.pred_anomaly_map = torch.rand(1, 256, 256)
```

## build_predict_results

`build_predict_results()` (in `baoiad.models.predict_utils`) is the standard way to construct prediction output from a detector's `predict()` method.

### Signature

```python
def build_predict_results(
    data_samples: List[ADDataSample],
    anomaly_map: Optional[Tensor] = None,
    image_scores: Optional[Tensor] = None,
) -> List[ADDataSample]:
```

### How It Works

1. If `image_scores` is provided, assigns them to `data_samples[i].pred_score`
2. If `image_scores` is not provided, derives image scores from `anomaly_map` (typically `max` of the anomaly map)
3. Assigns anomaly maps to `data_samples[i].pred_anomaly_map`
4. Returns the updated data samples

### Usage

```python
from baoiad.models.predict_utils import build_predict_results

def predict(self, batch_inputs, data_samples):
    feats = self.extract_feat(batch_inputs)
    anomaly_map = self.compute_anomaly_map(feats)

    return build_predict_results(
        data_samples=data_samples,
        anomaly_map=anomaly_map,
    )
```

## Data Pipeline

The data pipeline transforms raw images into model inputs through a sequence of transform operations.

### Standard Pipeline

```
LoadImage → ResizeAD → NormalizeAD → PackADInputs
```

### Transform Descriptions

#### LoadImage

Loads an image from disk and converts to RGB.

```python
dict(type='LoadImage')
```

#### ResizeAD

Resizes images and optionally ground-truth masks.

```python
dict(type='ResizeAD', size=256)               # Resize to 256x256
dict(type='ResizeAD', size=(256, 256))        # Explicit height x width
dict(type='ResizeAD', size=336)               # Used by CLIP-based methods
```

#### NormalizeAD

Normalizes images with ImageNet mean and standard deviation.

```python
dict(type='NormalizeAD')                      # Default: ImageNet normalization
dict(type='NormalizeAD', mean=[...], std=[...])  # Custom normalization
```

#### PackADInputs

Packs processed data into the format expected by MMEngine's `BaseModel`.

```python
dict(type='PackADInputs')
```

### Augmentation Pipeline

Some methods (DRAEM, DeSTSeg, CutPaste, NSA) use augmented training pipelines with anomaly synthesis:

```python
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='ResizeAD', size=256),
    dict(type='DRAEMAugmentation', ...),    # Anomaly synthesis for DRAEM
    dict(type='NormalizeAD'),
    dict(type='PackADInputs'),
]
```

### Method-Specific Pipelines

| Method | Pipeline Variant | Key Difference |
|--------|-----------------|----------------|
| DRAEM | `DRAEMAugmentation` | Perlin noise + DTD texture anomaly synthesis |
| DeSTSeg | `DeSTSegAugmentation` | Similar to DRAEM with segmentation guidance |
| CutPaste | `CutPasteAugmentation` | Cut-and-paste augmentation |
| NSA | `NSAAugmentation` | Poisson blending anomaly synthesis |
| MemAE | `MemAEVideoPipeline` | Video frame loading for temporal reconstruction |
| GLASS | `GLASSAugmentation` | LAS-based anomaly synthesis |
| RegAD | `RegADPipeline` | Few-shot support set loading |
| AdaCLIP | `AdaCLIPAuxPipeline` | Auxiliary text prompt loading |

## Data Batch Flow

```
Dataset.__getitem__(idx)
    │
    ├── Apply transforms (LoadImage → Resize → Normalize → Pack)
    │
    ▼
DataLoader (batching)
    │
    ▼
Model.forward(batch_inputs, data_samples, mode)
    │
    ├── mode='loss'  → model.loss()
    ├── mode='predict' → model.predict()
    └── mode='tensor' → model._forward()
    │
    ▼
AnomalyDetectionMetric.process(data_samples, predictions)
    │
    ▼
AnomalyDetectionMetric.compute_metrics(results)
```