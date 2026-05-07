# Get Started

## Prerequisites

- Python >= 3.9
- PyTorch >= 2.0 (with CUDA toolkit for GPU support)
- mmengine >= 0.10
- mmcv >= 2.0

## Installation

Clone the repository and install BaoIAD in editable mode:

```bash
git clone https://github.com/ChenjieXu/BaoIAD.git
cd BaoIAD
pip install -e .
```

This installs the core dependencies listed in [`pyproject.toml`](../../pyproject.toml): `mmengine`, `mmcv`, `torch`, `torchvision`, `timm`, `scipy`, `scikit-learn`, `einops`, `opencv-python-headless`, and `Pillow`.

### Optional Extras

Some methods require additional packages. Install them individually or all at once:

```bash
# Install everything
pip install -e ".[all]"

# Or install only what you need:
pip install -e ".[flow]"        # FrEIA (normalizing flow methods: FastFlow, CFlow, ...)
pip install -e ".[vl]"          # open_clip_torch (vision-language methods: WinCLIP, ...)
pip install -e ".[saa]"         # groundingdino, segment-anything (SAA+)
pip install -e ".[mmpretrain]"  # mmpretrain (backbone variants)
pip install -e ".[faiss-cpu]"   # faiss-cpu (fast nearest-neighbor search)
pip install -e ".[geomloss]"    # geomloss (optimal transport losses)
pip install -e ".[imgaug]"      # imgaug, openpyxl (augmentation-based methods)
pip install -e ".[mamba]"       # mamba-ssm (Mamba-based methods)
pip install -e ".[dev]"         # pytest, ruff, pre-commit (development)
```

The `[all]` extra includes `flow`, `vl`, `saa`, `geomloss`, `imgaug`, `mmpretrain`, and `faiss-cpu`. The `mamba` and `faiss-gpu` extras are excluded from `[all]` because they require specific CUDA builds.

## Dataset Setup

Set the `BAOIAD_DATA_ROOT` environment variable to point to the directory containing your datasets:

```bash
# Option 1: Source the provided env script (defaults to <repo>/data/)
source tools/env.sh

# Option 2: Set the variable manually
export BAOIAD_DATA_ROOT=/path/to/your/datasets
```

The env script ([`tools/env.sh`](../../tools/env.sh)) also configures `HF_HOME` and `TORCH_HOME` for offline lab environments, and sets `HF_HUB_OFFLINE=1` by default during training/testing to avoid network calls.

Place each dataset under `$BAOIAD_DATA_ROOT/<dataset_name>/` (e.g., `data/mvtec_ad/`, `data/visa/`). See [Prepare Datasets](user_guides/prepare_dataset.md) for download links and expected directory layouts.

## Your First Run

Train PatchCore on MVTec AD (all categories):

```bash
python tools/train.py configs/patchcore/patchcore_wrn50_256_mvtec_strict.py \
    --work-dir runs/patchcore_mvtec
```

This will:

1. Load the config from [`configs/patchcore/patchcore_wrn50_256_mvtec_strict.py`](../../configs/patchcore/patchcore_wrn50_256_mvtec_strict.py), which inherits base configs for runtime, MVTec AD dataset, and a 100-epoch schedule.
2. Train PatchCore with a WideResNet-50-2 backbone on all 15 MVTec AD categories.
3. Save checkpoints and logs to `runs/patchcore_mvtec/`.

### Test a Checkpoint

After training completes, evaluate the model:

```bash
python tools/test.py configs/patchcore/patchcore_wrn50_256_mvtec_strict.py \
    runs/patchcore_mvtec/best.pth \
    --work-dir runs/patchcore_mvtec
```

### Train a Single Category

Override the dataset category from the command line:

```bash
python tools/train.py configs/patchcore/patchcore_wrn50_256_mvtec_strict.py \
    --work-dir runs/patchcore_bottle \
    --cfg-options \
    train_dataloader.dataset.cls_names="['bottle']" \
    train_dataloader.dataset.multi_class=False \
    test_dataloader.dataset.cls_names="['bottle']" \
    test_dataloader.dataset.multi_class=False
```

### Force CPU Mode

Both `train.py` and `test.py` support a `--cpu` flag:

```bash
python tools/train.py <config> --cpu --work-dir runs/cpu_test
```

## Output Directory Structure

After training, the work directory contains:

```
runs/patchcore_mvtec/
├── <timestamp>/
│   ├── vis_data/          # Visualization outputs (when enabled)
│   └── <timestamp>.log    # Training log
├── best.pth               # Best checkpoint
├── last_checkpoint        # Path to latest checkpoint (for resume)
└── <epoch>.pth            # Periodic checkpoints
```

## CLI Reference

### `tools/train.py`

```
python tools/train.py <config> [options]

positional:
  config                Path to training config file

options:
  --work-dir DIR        Working directory for logs and checkpoints
  --resume              Resume from the latest checkpoint in work_dir
  --cpu                 Force CPU device (disables CUDA and MPS)
  --cfg-options K=V ... Override config fields (key=value format)
```

### `tools/test.py`

```
python tools/test.py <config> [checkpoint] [options]

positional:
  config                Path to test config file
  checkpoint            (optional) Path to checkpoint file

options:
  --work-dir DIR        Working directory for results
  --cpu                 Force CPU device
  --cfg-options K=V ... Override config fields
```

Source: [`tools/train.py`](../../tools/train.py), [`tools/test.py`](../../tools/test.py).

## Docs Build vs. Runtime Environment

This documentation is written in Markdown for direct browsing and can also be built with [Sphinx](https://www.sphinx-doc.org/) (see [`docs/`](../) for RST index files). Building the docs uses `docs/requirements.txt` — these packages are **not** listed in `pyproject.toml` because they are not needed to run the benchmark. If you want to build the docs locally:

```bash
pip install -r docs/requirements.txt
python -m sphinx -b html docs/en /tmp/baoiad-docs-html
```

The runtime environment (training, testing, benchmarking) only needs the dependencies in `pyproject.toml`.
