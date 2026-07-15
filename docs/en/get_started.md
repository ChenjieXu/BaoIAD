# Get Started

## Prerequisites

- Python 3.10 or 3.12 (the release-verified versions)
- PyTorch >= 2.0; validate the intended CUDA build separately before claiming GPU support
- mmengine >= 0.10
- MMCV >= 2.0 (`mmcv-lite` in the core installation)

## Installation

Clone the repository and install BaoIAD in editable mode:

```bash
git clone https://github.com/Baosight-xVue/BaoIAD.git
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
pip install -e ".[faiss-cpu]"   # faiss-cpu (fast nearest-neighbor search)
pip install -e ".[geomloss]"    # geomloss (optimal transport losses)
pip install -e ".[glass]"       # pandas, openpyxl (GLASS metadata/assets)
pip install -e ".[visualization]"  # matplotlib (optional plotting)
pip install -e ".[dev]"         # pytest, ruff, pre-commit (development)
```

The `[all]` extra includes `flow`, `vl`, `saa`, `geomloss`, `glass`, `visualization`, and `faiss-cpu`. Development tools are installed separately with `[dev]`.

## Installation verification

Run the lightweight local checks after installation:

```bash
python tools/check_install.py
python tools/check_method_inventory.py
```

`check_install.py` reports the BaoIAD, Python, PyTorch, MMEngine, and MMCV versions; resolves the configured data root and cache directories; and reports which optional dependency groups are available. Missing optional groups are informational and do not fail a valid core installation.

This is a CPU-only local release gate. It does not download models, inspect dataset contents, load checkpoints, start training, query CUDA/MPS, or perform GPU computation. A successful result therefore does **not** constitute GPU validation. Run an independent GPU smoke gate on the intended CUDA/PyTorch/MMCV stack with the selected method, assets, and dataset before claiming GPU support.

Use `python tools/check_install.py --offline` to enable BaoIAD and supported model-hub offline environment variables for the self-check process. The self-check is network-free even without that flag. The same `--offline` option on `train.py`, `test.py`, and `benchmark.py` blocks supported download paths and requires all datasets and artifacts to be available locally.

`--trusted-checkpoint` is not needed or accepted by `check_install.py` because it never loads a checkpoint. On `train.py`, `test.py`, and `benchmark.py`, use it only for a legacy pickle checkpoint obtained from and verified against a trusted source: deserialization may execute arbitrary code. Prefer safely loadable checkpoint formats and the default restricted policy whenever possible.

Tensor-only `.pth` files and `.safetensors` use the restricted loader on every supported PyTorch version. PyTorch 2.6+ can also reconstruct BaoIAD's known MMEngine `HistoryBuffer` metadata through a narrow safe-globals allowlist. On PyTorch 2.0–2.5, a standard MMEngine resume/evaluation checkpoint containing that metadata is rejected by default; after independently verifying its origin and integrity, either move the run to PyTorch 2.6+ or use `--trusted-checkpoint` for that file.

## Dataset Setup

Set the `BAOIAD_DATA_ROOT` environment variable to point to the directory containing your datasets:

```bash
# Option 1: Source the provided env script (defaults to <repo>/data/)
source tools/env.sh

# Option 2: Set the variable manually
export BAOIAD_DATA_ROOT=/path/to/your/datasets
```

The env script ([`tools/env.sh`](../../tools/env.sh)) configures `HF_HOME` and `TORCH_HOME`, but does not silently disable network access. Add `--offline` to `train.py`, `test.py`, or `benchmark.py` when all required datasets and weights are already available locally. BaoIAD then enables the supported model-hub offline variables and rejects its own download attempts before opening a connection. Existing offline environment variables are also respected.

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
