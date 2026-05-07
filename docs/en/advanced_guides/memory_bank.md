# Memory Bank Lifecycle

Several anomaly detection methods in BaoIAD use a **memory bank** — a collection of normal features gathered during training and processed into a searchable index or statistical model before evaluation. This page documents how memory banks are built, saved, and loaded.

## Methods Using Memory Banks

| Method | Base Class | Memory bank content |
|--------|-----------|-------------------|
| PatchCore | `MemoryBankADModel` | CoreSet-subsampled patch features |
| PaDiM | `MemoryBankADModel` | Per-position Gaussian distributions |
| DFM | `MemoryBankADModel` | PCA-transformed feature statistics |
| DFKDE | `MemoryBankADModel` | Kernel density estimate of features |
| MemoryBank baseline | `MemoryBankADModel` | Raw averaged normal features |
| EfficientAD | `KnowledgeDistillationADModel` | Teacher stats + quantile normalization |
| PyramidFlow | `BaseADModel` | Latent template from training data |
| CutPaste | `BaseADModel` | Gaussian Density Estimate (GDE) |

## Architecture

The memory bank lifecycle is managed by two cooperating components:

1. **`MemoryBankADModel`** (`baoiad/models/base_ad_model.py`) — Provides `_memory_bank` list, `_collect_features()`, `_clear_memory_bank()`, and `build_memory_bank()`.
2. **`MemoryBankHook`** (`baoiad/engine/hooks/memory_bank_hook.py`) — Registered as `MemoryBankHook` in the `HOOKS` registry. Triggers memory bank construction at the right point in the training loop.

## Lifecycle

### Phase 1: Feature Collection (Training)

During training, methods collect features from the normal training set. The exact mechanism varies:

- **MemoryBankADModel subclasses**: The head's `forward()` / `loss()` method calls `_collect_features(feats)` which appends CPU tensors to `_memory_bank`.
- **Other methods** (EfficientAD, CutPaste): Collect statistics incrementally during training iterations.

### Phase 2: Memory Bank Construction

After the final training epoch, `MemoryBankHook.after_train_epoch()` triggers `build_memory_bank()`:

```
Training loop:
  Epoch 0 ... N-1  ← Features collected during forward passes
  ─────────────────
  after_train_epoch (epoch == max_epochs - 1)
    → MemoryBankHook._ensure_memory_bank()
      → model.build_memory_bank()   or   model.fit()
        → Processes collected features into final index/stats
```

The hook also fires in `after_train()` for iteration-based training methods (where `after_train_epoch` may not be reached).

### Phase 3: Before Validation / Test

`MemoryBankHook.before_val_epoch()` ensures the memory bank is built before evaluation starts. For methods with `always_refit=True` (e.g., CutPaste), the bank is rebuilt at every validation checkpoint.

`MemoryBankHook.before_test_epoch()` performs the same check for test-only evaluation.

## MemoryBankHook API

```python
from baoiad.registry import HOOKS

# Registered as:
@HOOKS.register_module(force=True)
class MemoryBankHook(Hook):
    priority = 'LOW'
```

### Hook Points

| Hook point | Behavior |
|-----------|----------|
| `before_train` | Calls `model.pre_train_setup(dataloader)` if available (e.g., EfficientAD teacher stats) |
| `before_train_epoch` | Calls `model.set_epoch_info(epoch, max_epochs)` for phase-switching methods (DSR, DeSTSeg) |
| `before_train_iter` | Calls `model.set_iter_info(iter, max_iters)` for iteration-based methods (CutPaste) |
| `after_train_epoch` | Triggers `build_memory_bank()` on the last epoch |
| `after_train` | Ensures memory bank is built for iteration-based methods; optional refit via `refit_after_train=True` |
| `before_val_epoch` | Rebuilds if `always_refit=True`; calls `compute_normalization_stats()` if available |
| `before_test_epoch` | Same as `before_val_epoch` for test-only runs |

### Model Opt-In Attributes

Models can opt into specific memory bank behaviors by setting these attributes:

| Attribute | Type | Default | Effect |
|-----------|------|---------|--------|
| `always_refit` | `bool` | `False` | Rebuild memory bank before every val/test (e.g., CutPaste GDE) |
| `refit_after_train` | `bool` | `False` | Force rebuild in `after_train` even if already built |
| `pre_train_setup_builds_memory_bank` | `bool` | `False` | Mark bank as built after `pre_train_setup` (e.g., EfficientAD) |
| `template_dataloader_split` | `str` | `'val'` | Which dataloader to use for template building (`'train'` or `'val'`) |

### Configuring the Hook

`MemoryBankHook` is included in the default runtime config:

```python
# configs/_base_/default_runtime.py
custom_hooks = [dict(type='MemoryBankHook')]
```

No additional configuration is needed — the hook auto-detects whether the model uses memory banks and acts accordingly.

## build_memory_bank Dataloader Parameter

The hook inspects the method signature of `build_memory_bank()` to decide whether to pass the training dataloader:

```python
sig = inspect.signature(method)
params = list(sig.parameters.keys())
if len(params) > 0:
    method(runner.train_dataloader)   # Passes dataloader
else:
    method()                          # No arguments
```

Some methods (e.g., PyramidFlow via `build_template_from_dataloader`) need the dataloader for template building.

## MemoryBankADModel API

```python
class MemoryBankADModel(BaseADModel):
    def __init__(self, backbone=None, freeze_backbone=True, **kwargs):
        self._memory_bank: list[Tensor] = []

    def _collect_features(self, feats: tuple[Tensor, ...]) -> None:
        """Append features to the memory bank (CPU tensors)."""
        for f in feats:
            self._memory_bank.append(f.cpu())

    def _clear_memory_bank(self) -> None:
        """Clear collected features."""
        self._memory_bank.clear()

    def build_memory_bank(self) -> None:
        """Override in subclass to process collected features."""
        pass

    def fit(self) -> None:
        """Backward-compatible alias for build_memory_bank()."""
        self.build_memory_bank()
```

### Example: Custom Memory Bank Method

```python
from baoiad.models.base_ad_model import MemoryBankADModel
from baoiad.registry import MODELS

@MODELS.register_module()
class MyMemoryBankDetector(MemoryBankADModel):
    def __init__(self, backbone=None, **kwargs):
        super().__init__(backbone=backbone, **kwargs)
        self.bank = None

    def build_memory_bank(self):
        """Process collected features into final bank."""
        all_feats = torch.cat(self._memory_bank, dim=0)
        # Custom processing (e.g., clustering, PCA, KDE)
        self.bank = self._process(all_feats)
        self._clear_memory_bank()
```

## Saving and Loading Memory Banks

Memory bank state is saved as part of the model checkpoint. When loading a checkpoint for testing, the `MemoryBankHook.before_test_epoch()` will attempt to build the memory bank. If the model's `state_dict` already contains the bank parameters, the bank is restored from the checkpoint directly.
