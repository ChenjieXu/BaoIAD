# BaoIAD

A unified benchmarking framework for industrial anomaly detection, built on [MMEngine](https://github.com/open-mmlab/mmengine) (OpenMMLab style).

BaoIAD integrates 50+ anomaly detection methods under a single config-driven interface for fair comparison across datasets including MVTec AD, VisA, and more.

## Features

- **50+ methods**: PatchCore, RD, Dinomaly, AnomalyCLIP, SimpleNet, DRAEM, and many more
- **Unified interface**: Config-driven, MMEngine-style training and evaluation
- **Fair comparison**: Standardized backbones, consistent evaluation metrics
- **OpenMMLab ecosystem**: Registry, config inheritance, modular components

## Installation

```bash
pip install -e .

# With optional dependencies
pip install -e ".[flow]"    # Normalizing flow methods (CSFlow, FastFlow, etc.)
pip install -e ".[vl]"      # Vision-language methods (WinCLIP, AnomalyCLIP, etc.)
pip install -e ".[all]"     # Everything
```

Set up environment variables:

```bash
source tools/env.sh
# Or set manually:
export BAOIAD_DATA_ROOT=/path/to/data
```

## Quick Start

### Training

```bash
python tools/train.py configs/patchcore/patchcore_wrn50_256_mvtec_strict.py --work-dir runs/patchcore

# Single category
python tools/train.py configs/patchcore/patchcore_wrn50_256_mvtec_strict.py \
    --work-dir runs/patchcore_bottle \
    --cfg-options train_dataloader.dataset.cls_names="['bottle']" train_dataloader.dataset.multi_class=False
```

### Testing

```bash
python tools/test.py configs/patchcore/patchcore_wrn50_256_mvtec_strict.py runs/patchcore/best.pth
```

### Benchmarking

```bash
python tools/benchmark.py --data_root data/mvtec_ad --methods patchcore rd --categories all \
    --output runs/benchmark_results.json
```

## Project Structure

```
BaoIAD/
├── baoiad/
│   ├── models/
│   │   ├── base_ad_model.py      # Base classes (BaseADModel, MemoryBankADModel, etc.)
│   │   ├── detectors/             # 50+ detector implementations
│   │   ├── backbones/             # Backbone wrappers (TIMM, CLIP, DINOv2, etc.)
│   │   └── losses/                # Registered loss modules
│   ├── datasets/                  # Dataset classes and transforms
│   ├── evaluation/                # AnomalyDetectionMetric
│   ├── registry.py                # 13 registries with scope='baoiad'
│   └── utils/                     # Shared utilities
├── configs/
│   ├── _base_/                    # Shared base configs
│   │   ├── backbones/             # Backbone definitions
│   │   ├── datasets/              # Dataset + dataloader configs
│   │   └── default_runtime.py     # Scope, seed, hooks
│   ├── patchcore/                 # Per-method configs
│   └── ...                        # 40+ method directories
├── tools/
│   ├── train.py                   # Training entry point
│   ├── test.py                    # Testing entry point
│   ├── benchmark.py               # Benchmark runner
│   └── env.sh                     # Environment variables
└── tests/                         # Test suite
```

## Supported Methods

| Category | Methods |
|----------|---------|
| Memory Bank | PatchCore, SPADE, PaDiM, DFM, DFKDE, RegAD, GraphCore |
| Knowledge Distillation | RD, RD++, STFPM, EfficientAD, Dinomaly |
| Normalizing Flow | CSFlow, FastFlow, CFlow, UFlow, DifferNet, PyramidFlow |
| Reconstruction | DRAEM, MemSeg, DeSTSeg, MemAE, FRE, GANomaly, DSR |
| Vision-Language | WinCLIP, AnomalyCLIP, AnoVL, MuSc, AdaCLIP, AACLIP, AnomalyDINO |
| Discriminator | SimpleNet, SuperSimpleNet, CFA |
| Other | InvAD, ViTAD, UniAD, MambaAD, NSA, ResAD, CutPaste, GLASS, AST, UniNet, UniVAD |

## Supported Datasets

MVTec AD, VisA, BTech, MVTec 3D AD, MVTec LOCO, MPDD, MVTec AD 2, Kolektor, VAD, RealIAD

## Evaluation Metrics

- **Image-level**: AUROC, F1-max, AP, ECE, FPR@95TPR
- **Pixel-level**: AUROC, F1-max, AP, AUPRO, AUPIMO, ECE

## License

This project is licensed under the Apache License 2.0.
