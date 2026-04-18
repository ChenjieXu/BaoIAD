# Memory Bank Lifecycle

Several anomaly detection methods in BaoIAD (PatchCore, SPADE, PaDiM, DFM, DFKDE, RegAD, GraphCore) use a memory bank of normal features. This page documents the memory bank lifecycle managed by `MemoryBankHook`.

## MemoryBankHook

`MemoryBankHook` is declared in `configs/_base_/default_runtime.py` and is **always active**. It automatically calls the appropriate method after training completes and before validation.

### Hook Lifecycle

```
Training Loop
    │
    ├── epoch 1 ... epoch N (loss mode)
    │
    ▼
After Training (before first validation)
    │
    ├── model.build_memory_bank()    ← MemoryBankHook calls this
    │       or
    ├── model.fit()                  ← Fallback if build_memory_bank() not defined
    │
    ▼
Validation / Test (predict mode)
```

### How It Works

```python
class MemoryBankHook(Hook):
    """Hook to build memory bank after training."""

    def after_train(self, runner) -> None:
        model = runner.model
        if hasattr(model, 'build_memory_bank'):
            model.build_memory_bank()
        elif hasattr(model, 'fit'):
            model.fit()
```

Since `BaseADModel` provides a no-op `build_memory_bank()`, only methods that need a memory bank override this method.

## Methods Using Memory Banks

| Method | Override | Description |
|--------|----------|-------------|
| PatchCore | `build_memory_bank()` | Extracts features, applies coreset subsampling, builds FAISS index |
| SPADE | `build_memory_bank()` | Stores all normal features with their pixel positions |
| PaDiM | `build_memory_bank()` | Fits multivariate Gaussian to features per position |
| DFM | `build_memory_bank()` | Applies PCA transformation to features |
| DFKDE | `build_memory_bank()` | Fits kernel density estimate to features |
| RegAD | `fit()` | Registers support set features for few-shot matching |
| GraphCore | `build_memory_bank()` | Builds graph-structured memory with nearest neighbors |

## Implementing a Memory Bank Method

To add a method that uses a memory bank:

```python
from baoiad.models.base_ad_model import MemoryBankADModel


class MyMemoryMethod(MemoryBankADModel):

    def loss(self, batch_inputs, data_samples):
        """During training, just extract features (no real loss for memory bank methods)."""
        feats = self.extract_feat(batch_inputs)
        # Some memory bank methods don't have a training loss
        # Return a dummy loss or zero loss
        return {'loss': feats[0].sum() * 0}

    def build_memory_bank(self):
        """Called by MemoryBankHook after training."""
        # Collect features from training set
        self.memory_bank = self._collect_features()
        # Build index (e.g., FAISS, scikit-learn)
        self._build_index()

    def predict(self, batch_inputs, data_samples):
        """Compute anomaly scores using the memory bank."""
        feats = self.extract_feat(batch_inputs)
        # Compute distance to memory bank
        anomaly_map = self._compute_distance(feats)
        return build_predict_results(data_samples=data_samples, anomaly_map=anomaly_map)
```

## Training Details

For memory bank methods, the "training" phase may consist of:

1. **Feature extraction only** (PatchCore, SPADE): No gradient computation; features are collected and stored
2. **Distribution fitting** (PaDiM, DFKDE, DFM): After feature extraction, statistical parameters are estimated
3. **Few-shot registration** (RegAD): Support set features are registered for matching

The `MemoryBankHook` ensures the memory bank is built before any validation or test evaluation occurs, regardless of the training schedule.