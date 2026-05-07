# Method Caveats

This page documents per-method requirements, optional dependencies, special training scripts, and known quirks. If a method fails to import or train, check here first.

## Optional Dependencies

The core BaoIAD package requires only `mmengine`, `torch`, and common scientific Python packages. Several method families need additional packages that are not installed by default.

### Normalizing Flow Family

Requires the [FrEIA](https://github.com/VLL-HD/FrEIA) library for invertible neural network layers.

```bash
pip install FrEIA
```

| Method | Slug | Affected file |
|--------|------|---------------|
| CFlow | `cflow` | `baoiad/models/detectors/cflow.py` |
| DifferNet | `differnet` | `baoiad/models/detectors/differnet.py` |
| FastFlow | `fastflow` | `baoiad/models/detectors/fastflow.py` |
| U-Flow | `uflow` | `baoiad/models/detectors/uflow.py` |

**Note**: DifferNet implements its own normalizing flow blocks but references FrEIA design patterns. CFlow and FastFlow import `FrEIA.framework` and `FrEIA.modules` directly. PyramidFlow (`pyramidflow`) is a normalizing flow method but does **not** depend on FrEIA — it ships its own flow implementation.

### Vision-Language Family

Requires the [open_clip](https://github.com/mlfoundations/open_clip) library for CLIP model loading.

```bash
pip install open_clip_torch
```

| Method | Slug |
|--------|------|
| WinCLIP | `winclip` |
| AnomalyCLIP | `anomalyclip` |
| AACLIP | `aaclip` |
| AdaCLIP | `adaclip` |
| AnoVL | `anovl` |
| SAA+ | `saaplus` |
| MuSc | `musc` |

All VL methods route through `baoiad/models/backbones/clip_backbone.py`, which wraps `open_clip.create_model_and_transforms`. The backbone tries a local reference fallback before importing `open_clip`, but the full pipeline requires the package.

**Note**: AnomalyDINO (`anomalydino`) is a few-shot/registration method that uses DINOv2 features rather than CLIP — it does **not** require `open_clip`.

### Self-Supervised Synthesis Family

Some synthesis methods require additional augmentation libraries:

```bash
pip install imgaug
```

| Method | Slug | Dependency |
|--------|------|------------|
| GLASS | `glass` | `imgaug` (for Perlin noise rotation augmentations via `baoiad/utils/glass_utils.py`) |
| DRAEM | `draem` | Ships its own augmentation in `baoiad/datasets/transforms/augmentation.py` — no extra package needed |
| DSR | `dsr` | Ships its own augmentation — no extra package needed |
| CutPaste | `cutpaste` | No extra package needed |
| NSA | `nsa` | No extra package needed |

### Feature-Memory / Density Family

| Method | Slug | Dependency |
|--------|------|------------|
| DFKDE | `dfkde` | `faiss` (for KDE scoring in `baoiad/models/heads/scoring_heads.py` and `baoiad/models/heads/memory_bank_head.py`) |
| PatchCore | `patchcore` | No extra package needed (uses numpy nearest-neighbor) |
| PaDiM | `padim` | No extra package needed |
| DFM | `dfm` | No extra package needed |

Install faiss for DFKDE:

```bash
pip install faiss-cpu   # or faiss-gpu for CUDA acceleration
```

### GeomLoss

| Method | Slug | Dependency |
|--------|------|------------|
| RD++ | `rdpp` | `geomloss` (for optimal-transport loss in `baoiad/models/detectors/rdpp.py`) |

```bash
pip install geomloss
```

## Memory Bank Lifecycle

Several methods pre-compute feature representations from the training set and store them as a "memory bank" or equivalent structure. These methods have a distinct training phase that differs from gradient-based optimization:

| Method | Slug | Bank mechanism |
|--------|------|----------------|
| PatchCore | `patchcore` | Core-set sampled feature vectors stored in `model.head.memory_bank` |
| PaDiM | `padim` | Per-layer Gaussian statistics (`mean_list`, `cov_list`) fitted on training features |
| DFM | `dfm` | PCA model fitted on training features |
| DFKDE | `dfkde` | Feature bank + KDE scoring via faiss |
| RegAD | `regad` | Support bank of feature vectors for registration-based comparison |
| AnomalyDINO | `anomalydino` | Memory bank of DINOv2 patch features |
| EfficientAD | `efficientad` | Pre-computed feature maps for teacher-student normalization |
| CFA | `cfa` | Feature bank with coupled attention |
| MemSeg | `memseg` | Memory bank of normal feature maps |
| PyramidFlow | `pyramidflow` | Normalizing flow fitted on training features |

For benchmark runs, `tools/benchmark.py` handles the memory bank lifecycle automatically — it calls `build_memory_bank()` or runs the forward-on-training pass as needed. When using `tools/train.py` directly, the MMEngine `MemoryBankHook` (registered in `configs/_base_/default_runtime.py`) manages this lifecycle.

## Special Training Scripts

Three methods require dedicated training scripts instead of the standard `tools/train.py`:

### AST — `tools/train_ast.py`

AST uses a two-stage training procedure (teacher pre-training then student distillation) that cannot be expressed in a single MMEngine training loop.

```bash
python tools/train_ast.py configs/ast/ast_effnet_b5_768_mvtec_strict.py --work-dir runs/ast/mvtec
```

The benchmark runner invokes this script automatically when the AST config declares `benchmark_train_script='tools/train_ast.py'`.

### RegAD — `tools/train_regad_strict.py`

RegAD uses cross-category training: the registration network trains on all categories *except* the target one, then evaluates on the target. This requires a custom training loop that iterates over category combinations.

```bash
python tools/train_regad_strict.py configs/regad/regad_wrn50_256_mvtec_strict.py --work-dir runs/regad/mvtec
```

The benchmark runner passes category information via cfg-options (`model.target_cls`, `model.data_root`) when invoking the standard training script, but for standalone training use `tools/train_regad_strict.py`.

### ViTAD — `tools/train_vitad_exact_order.py`

ViTAD requires a specific training sample order that matches the official ADer implementation for strict alignment. This script replays the official training order.

```bash
python tools/train_vitad_exact_order.py configs/vitad/vitad_256_mvtec_strict.py --work-dir runs/vitad/mvtec
```

## Known Quirks

### NSA Category-Specific Epochs

NSA uses different epoch counts depending on the MVTec AD category: `hazelnut`, `metal_nut`, and `screw` require 560 epochs while all others use 320. The benchmark runner (`tools/benchmark.py`) applies this automatically, but when training manually, adjust `train_cfg.max_epochs` and `param_scheduler.0.T_max` accordingly.

### CutPaste RepeatDataset Wrapper

CutPaste configs wrap the training dataset in `RepeatDataset` to accumulate enough training samples from the self-supervised augmentation pipeline. When overriding config options (e.g. via `--cfg-options`), dataset-level keys must target the inner dataset: `train_dataloader.dataset.dataset.data_root` instead of `train_dataloader.dataset.data_root`.

### Multi-Class Methods

Some methods operate on all categories simultaneously rather than per-category. These are flagged with `benchmark_multi_class=True` in their config. The benchmark runner executes them once with category set to `'all'` instead of looping.

### Iteration-Based Training

Methods like EfficientAD use iteration-based training (`by_epoch=False`, `max_iters=...`) rather than epoch-based. The `--epochs` flag in `tools/benchmark.py` is silently ignored for these methods since the training budget is defined in iterations.

### Deterministic Behavior

BaoIAD sets `randomness = dict(seed=42, deterministic=False)` in the default runtime config. Setting `deterministic=True` may reduce training speed. Some methods (particularly those with stochastic augmentation like DRAEM, GLASS, CutPaste) may show small run-to-run variation even with a fixed seed due to GPU nondeterminism in certain operations.

### GPU Memory

Methods with large backbones or high-resolution inputs may require significant GPU memory:
- **AST**: EfficientNet-B5 at 768×768 — requires ~10 GB VRAM.
- **Dinomaly**: 392×392 ViT-based — requires ~8 GB VRAM.
- **VL methods** (WinCLIP, AnomalyCLIP, AACLIP, AdaCLIP, MuSc): CLIP ViT-L/14 at 336 or 518 resolution — require ~6–8 GB VRAM.
- **AnomalyDINO**: ViT-B/14 at 448 resolution — requires ~6 GB VRAM.

If you encounter OOM errors, try reducing `batch_size` via the `--batch_size` flag in `tools/benchmark.py`.
