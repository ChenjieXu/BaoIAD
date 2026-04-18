# BaseADModel Architecture

`BaseADModel` is the core base class for all anomaly detectors in BaoIAD. It follows the MMEngine convention of backbone-neck-head decomposition and implements a 3-mode forward dispatch.

## Class Hierarchy

```
BaseADModel (base_ad_model.py)
├── MemoryBankADModel
│   └── PatchCore, SPADE, PaDiM, DFM, DFKDE, RegAD, GraphCore
├── KnowledgeDistillationADModel
│   └── RD, RD++, STFPM, EfficientAD, Dinomaly
├── FlowBasedADModel
│   └── CSFlow, FastFlow, CFlow, UFlow, DifferNet, PyramidFlow
├── ReconstructionADModel
│   └── DRAEM, MemSeg, DeSTSeg, MemAE, FRE, GANomaly, DSR
├── VisionLanguageADModel
│   └── WinCLIP, AnomalyCLIP, AnoVL, MuSc, AdaCLIP, AACLIP, AnomalyDINO
├── DiscriminatorADModel
│   └── SimpleNet, SuperSimpleNet, CFA
└── BaseADModel (direct)
    └── InvAD, ViTAD, UniAD, MambaAD, NSA, ResAD, CutPaste, GLASS, AST, PNI, ...
```

## When to Use Each Sub-class

| Sub-class | Pattern | Key Hook |
|-----------|---------|----------|
| `MemoryBankADModel` | Build a bank of normal features; detect anomalies by distance | `build_memory_bank()` called by MemoryBankHook |
| `KnowledgeDistillationADModel` | Teacher-student feature matching | `loss()` computes teacher-student discrepancy |
| `FlowBasedADModel` | Normalizing flow on feature distributions | `loss()` computes negative log-likelihood |
| `ReconstructionADModel` | Input reconstruction with anomaly synthesis | `loss()` computes reconstruction + synthesis loss |
| `VisionLanguageADModel` | CLIP-based zero/few-shot | `loss()` aligns text-image features |
| `DiscriminatorADModel` | Discriminate normal vs perturbed features | `loss()` computes adversarial loss |

## Three-Mode Forward Dispatch

`BaseADModel.forward()` dispatches to one of three methods based on the `mode` argument:

```python
def forward(self, inputs, data_samples, mode='tensor'):
    if mode == 'loss':
        return self.loss(inputs, data_samples)
    elif mode == 'predict':
        return self.predict(inputs, data_samples)
    elif mode == 'tensor':
        return self._forward(inputs)
    else:
        raise RuntimeError(f'Invalid mode: {mode}')
```

| Mode | Called By | Required Implementation |
|------|-----------|------------------------|
| `loss` | Training loop | `loss()` returns `Dict[str, Tensor]` |
| `predict` | Test/val loop | `predict()` returns `List[ADDataSample]` |
| `tensor` | Feature extraction | `_forward()` returns `Tensor` or `Tuple[Tensor]` |

## Feature Extraction

```python
def extract_feat(self, batch_inputs):
    """Extract features from backbone (frozen) and optional neck."""
    feats = self.backbone(batch_inputs)
    if self.with_neck:
        feats = self.neck(feats)
    return feats
```

The backbone is **always frozen** by default. The `train()` method is overridden to keep the backbone in eval mode even when the model is in training mode.

## Building a Detector

The minimal interface for any detector is:

```python
class MyDetector(BaseADModel):

    def loss(self, batch_inputs, data_samples):
        """Compute training losses. Must return dict of scalar tensors."""
        feats = self.extract_feat(batch_inputs)
        # ... compute loss ...
        return {'loss': loss_value}

    def predict(self, batch_inputs, data_samples):
        """Predict anomaly scores. Must return list of ADDataSample."""
        feats = self.extract_feat(batch_inputs)
        # ... compute anomaly map and score ...
        return build_predict_results(data_samples=data_samples, anomaly_map=anomaly_map)
```

## Freeze Behavior

`BaseADModel.train()` is overridden to keep the backbone in eval mode:

```python
def train(self, mode=True):
    super().train(mode)
    # Keep backbone frozen
    if self.freeze_backbone and self.backbone is not None:
        self.backbone.eval()
```

This ensures batch normalization and dropout layers in the backbone always use inference statistics, which is critical for consistent feature extraction.