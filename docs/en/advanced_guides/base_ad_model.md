# Base AD Model Architecture

All anomaly detection models in BaoIAD inherit from `BaseADModel` (in `baoiad/models/base_ad_model.py`), which itself extends `mmengine.model.BaseModel`. The class hierarchy follows the **backbone → neck → head** decomposition pattern adapted from mmdetection.

## Class Hierarchy

```
BaseADModel (backbone → neck → head, 3-mode forward)
  ├── MemoryBankADModel        — PatchCore, PaDiM, DFM, DFKDE
  ├── KnowledgeDistillationADModel — RD, EfficientAD, RD++, Dinomaly
  ├── FlowBasedADModel         — FastFlow, CFlow-AD, UFlow, DifferNet
  ├── ReconstructionADModel    — DRAEM, MemSeg, DeSTSeg
  ├── VisionLanguageADModel    — WinCLIP, AnomalyCLIP, MuSc, AdaCLIP
  └── DiscriminatorADModel     — SimpleNet, SuperSimpleNet, CFA
```

## BaseADModel

```python
from baoiad.models.base_ad_model import BaseADModel
```

`BaseADModel` implements the backbone → neck → head pipeline:

| Component | Config key | Registry | Description |
|-----------|-----------|----------|-------------|
| backbone  | `backbone` | `MODELS` | Feature extractor (frozen by default) |
| neck      | `neck`     | `MODELS` | Optional feature processing (e.g., `MultiScalePooling`) |
| head      | `head`     | `MODELS` | Method-specific anomaly scoring logic |

### Constructor

```python
class BaseADModel(BaseModel):
    def __init__(
        self,
        backbone: Optional[dict] = None,
        neck: Optional[dict] = None,
        head: Optional[dict] = None,
        freeze_backbone: bool = True,
        data_preprocessor: Optional[dict] = None,
        init_cfg: Optional[dict] = None,
    )
```

Each component is built via the `MODELS` registry:

```python
if backbone is not None:
    self.backbone = MODELS.build(backbone)
self.neck = MODELS.build(neck) if neck else None
self.head = MODELS.build(head) if head else None
```

When `freeze_backbone=True` (the default), the backbone is set to eval mode and all its parameters have `requires_grad=False`.

### Three-Mode Forward

`BaseADModel.forward()` supports three modes controlled by MMEngine's Runner:

```python
def forward(self, inputs, data_samples=None, mode='tensor'):
    feats = self.extract_feat(inputs)

    if mode == 'loss':
        return self.head.loss(feats, data_samples)    # Training
    elif mode == 'predict':
        return self.head.predict(feats, data_samples)  # Eval/Test
    elif mode == 'tensor':
        return feats                                    # Feature extraction
```

| Mode | When called | Return type |
|------|------------|-------------|
| `'loss'` | During `train_step` | `dict[str, Tensor]` — loss dict |
| `'predict'` | During `val_step` / `test_step` | `list[ADDataSample]` — predictions |
| `'tensor'` | Manual feature extraction | `tuple[Tensor, ...]` — feature maps |

### Feature Extraction

```python
def extract_feat(self, batch_inputs):
    ctx = torch.no_grad() if self.freeze_backbone else torch.enable_grad()
    with ctx:
        feats = self.backbone(batch_inputs)
    if isinstance(feats, Tensor):
        feats = (feats,)
    if self.neck is not None:
        feats = self.neck(feats)
    return feats
```

Backbone features are extracted under `no_grad` when the backbone is frozen. The neck (if present) processes multi-scale features before passing them to the head.

### train_step and test_step

MMEngine's `BaseModel` provides `train_step` and `test_step` which call `forward()` with the appropriate mode:

- **`train_step`**: Calls `forward(mode='loss')`, returns the loss dict for the optimizer.
- **`test_step`** / **`val_step`**: Calls `forward(mode='predict')`, returns `list[ADDataSample]`.

### Config Example

```python
model = dict(
    type='PatchCore',
    backbone=dict(
        type='TIMMBackbone',
        model_name='wide_resnet50_2',
        pretrained=True,
        features_only=True,
        out_indices=(1, 2, 3),
        frozen=True,
    ),
    neck=dict(type='MultiScalePooling', output_size=1),
    head=dict(
        type='MemoryBankHead',
        coreset_ratio=0.1,
        num_neighbors=9,
    ),
)
```

## MemoryBankADModel

```python
from baoiad.models.base_ad_model import MemoryBankADModel
```

Base class for methods that collect normal features during training and build a memory bank before evaluation.

**Methods using this base**: PatchCore, PaDiM, DFM, DFKDE, MemoryBank baseline.

Key features:
- Maintains an internal `_memory_bank: list[Tensor]` for feature collection.
- `_collect_features(feats)` appends CPU tensors during training.
- `build_memory_bank()` is called by `MemoryBankHook` after the final training epoch.
- `fit()` is a backward-compatible alias for `build_memory_bank()`.

```python
model = dict(
    type='PatchCore',          # inherits MemoryBankADModel
    backbone=dict(...),
    head=dict(type='MemoryBankHead', ...),
)
```

See [Memory Bank Guide](./memory_bank.md) for the full lifecycle.

## KnowledgeDistillationADModel

```python
from baoiad.models.base_ad_model import KnowledgeDistillationADModel
```

Base for teacher-student methods where a frozen teacher network guides a trainable student.

**Methods using this base**: RD, EfficientAD, RD++, Dinomaly.

Key features:
- `extract_teacher_feats(batch_inputs)` — extracts features from the frozen teacher backbone with `no_grad`.
- Overrides `train()` to keep both backbone and teacher in eval mode during training.

```python
model = dict(
    type='RD',                 # inherits KnowledgeDistillationADModel
    backbone=dict(...),
    head=dict(type='RDHead', ...),
)
```

## FlowBasedADModel

```python
from baoiad.models.base_ad_model import FlowBasedADModel
```

Base for normalizing-flow methods that model the distribution of normal features.

**Methods using this base**: FastFlow, CFlow-AD, UFlow, DifferNet.

Key features:
- `compute_flow_loss(outputs)` — computes negative log-likelihood from flow outputs.
- Frozen backbone + trainable normalizing flows.

## ReconstructionADModel

```python
from baoiad.models.base_ad_model import ReconstructionADModel
```

Base for methods that reconstruct normal images and detect anomalies via reconstruction error.

**Methods using this base**: DRAEM, MemSeg, DeSTSeg.

Key features:
- `freeze_backbone` defaults to `False` (unlike other bases).
- Supports anomaly generation during training.

## VisionLanguageADModel

```python
from baoiad.models.base_ad_model import VisionLanguageADModel
```

Base for CLIP-based methods that leverage vision-language features.

**Methods using this base**: WinCLIP, AnomalyCLIP, MuSc, AdaCLIP, AnoVL.

Key features:
- Provides CLIP normalization constants (`CLIP_MEAN`, `CLIP_STD`).
- Overrides `train()` to keep the backbone (CLIP visual encoder) frozen.

## DiscriminatorADModel

```python
from baoiad.models.base_ad_model import DiscriminatorADModel
```

Base for methods that discriminate between normal features and synthesized noise.

**Methods using this base**: SimpleNet, SuperSimpleNet, CFA.

Key features:
- Frozen backbone + trainable discriminator head.
- Overrides `train()` to keep backbone in eval mode.

## ADDataSample

All model predictions are packaged into `ADDataSample` instances (from `baoiad/structures/ad_data_sample.py`):

```python
from baoiad.structures import ADDataSample

sample = ADDataSample()
sample.set_metainfo({'cls_name': 'bottle', 'img_path': '/data/001.png'})
sample.gt_label = 1
sample.pred_score = 0.85
sample.pred_anomaly_map = score_map   # Tensor (1, H, W)
```

See [Data Flow Guide](./data_flow.md) for the full pipeline.

## build_predict_results Utility

The `build_predict_results` function (from `baoiad/models/predict_utils.py`) is the standard way heads package predictions:

```python
from baoiad.models.predict_utils import build_predict_results

results = build_predict_results(
    data_samples=data_samples,   # list[ADDataSample]
    img_scores=scores,           # Tensor or array of image-level scores
    score_maps=maps,             # Optional Tensor of pixel-level anomaly maps
    extra_scores={'pred_score_mean': mean_scores},
)
```

It normalizes scores, extracts per-sample anomaly maps, and sets `pred_score` and `pred_anomaly_map` on each `ADDataSample`.
