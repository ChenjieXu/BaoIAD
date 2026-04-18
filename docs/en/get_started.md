# Get Started

## Installation

### Prerequisites

- Python >= 3.9
- PyTorch >= 2.0 (with CUDA support recommended)
- [MMEngine](https://github.com/open-mmlab/mmengine) >= 0.10
- [MMCV](https://github.com/open-mmlab/mmcv) >= 2.0

### Install BaoIAD

```bash
git clone https://github.com/xxx/BaoIAD.git
cd BaoIAD
pip install -e .
```

### Optional Dependencies

BaoIAD uses optional dependencies for specific method families:

```bash
pip install -e ".[flow]"        # Normalizing flow methods (CSFlow, FastFlow, CFlow, etc.)
pip install -e ".[vl]"          # Vision-language methods (WinCLIP, AnomalyCLIP, etc.)
pip install -e ".[saa]"         # SAA/SAA+ (requires groundingdino + segment_anything)
pip install -e ".[geomloss]"    # MuSc, RD++ (optimal transport losses)
pip install -e ".[imgaug]"      # DRAEM, DeSTSeg (anomaly synthesis)
pip install -e ".[mmpretrain]"  # UniNet, UniVAD
pip install -e ".[faiss-cpu]"   # PatchCore, PaDiM (fast nearest neighbor)
pip install -e ".[all]"         # Install all optional dependencies
```

For development:

```bash
pip install -e ".[dev]"         # pytest, ruff, pre-commit
```

### Environment Setup

Set up environment variables for data caching and model downloads:

```bash
source tools/env.sh
```

Or set manually:

```bash
export BAOIAD_DATA_ROOT=/path/to/data   # Default: ./data
export HF_HOME=/path/to/hf_cache        # HuggingFace cache
export TORCH_HOME=/path/to/torch_cache  # PyTorch model cache
```

## Verification

Verify your installation:

```python
import baoiad
print(baoiad.__version__)  # 0.1.0
```

Quick functional test:

```bash
pytest tests/ -k "test_patchcore" -x
```

## Quick Start

### Training

Train PatchCore on MVTec AD (all 15 categories):

```bash
python tools/train.py configs/patchcore/patchcore_wrn50_256_mvtec_strict.py \
    --work-dir runs/patchcore
```

Train on a single category:

```bash
python tools/train.py configs/patchcore/patchcore_wrn50_256_mvtec_strict.py \
    --work-dir runs/patchcore_bottle \
    --cfg-options train_dataloader.dataset.cls_names="['bottle']" train_dataloader.dataset.multi_class=False
```

### Testing

Test with a trained checkpoint:

```bash
python tools/test.py configs/patchcore/patchcore_wrn50_256_mvtec_strict.py \
    runs/patchcore/best.pth
```

### Benchmarking

Run a benchmark across methods and categories:

```bash
python tools/benchmark.py \
    --data_root data/mvtec_ad \
    --methods patchcore rd \
    --categories all \
    --output runs/benchmark_results.json
```

## Common Issues

### MMEngine/MMCV Installation

MMCV requires compilation from source for custom CUDA ops. If you encounter build errors:

```bash
pip install mmcv -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
```

Adjust the CUDA/PyTorch version URL to match your environment.

### FrEIA Not Found

Normalizing flow methods (CSFlow, FastFlow, CFlow, UFlow, DifferNet, PyramidFlow, AST) require FrEIA:

```bash
pip install -e ".[flow]"
```

### open_clip Not Found

Vision-language methods (WinCLIP, AnomalyCLIP, AnoVL, MuSc, AdaCLIP, AACLIP) require open_clip:

```bash
pip install -e ".[vl]"
```

### CUDA Out of Memory

For large models or high-resolution images:

- Reduce batch size: `--cfg-options train_dataloader.batch_size=8`
- Use gradient accumulation: `--cfg-options optim_wrapper.accumulative_counts=4`
- Force CPU: `--cpu` (not recommended for training)

### faiss Not Available

PatchCore and PaDiM benefit from faiss for fast nearest-neighbor search. Without faiss, they fall back to a slower brute-force implementation:

```bash
pip install -e ".[faiss-cpu]"
```
