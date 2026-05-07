# Overview

**BaoIAD** is a unified benchmark for industrial anomaly detection (IAD). It provides 37 methods across 9 families, evaluated on 10 datasets under a single, consistent MMEngine-based framework. BaoIAD is designed to make cross-method comparison fair, reproducible, and transparent.

## What BaoIAD Is

- A **benchmark-first toolbox**: every method ships with reference-aligned configs, frozen reference outputs, and code-path parity evidence documented in [`docs/alignment/`](../alignment/).
- A **single-framework reimplementation** of 37 IAD methods — from feature-memory (PatchCore, PaDiM) to vision-language (WinCLIP, AnomalyCLIP) — sharing a common data pipeline, training loop, and evaluation metric set.
- A **config-driven** system built on MMEngine: inherit base configs for runtime, datasets, schedules, and backbones; override any field from the command line.

## What BaoIAD Is Not

- Not a general-purpose training library. Configs and code are tuned for the benchmark setting (specific backbones, resolutions, hyperparameters chosen for alignment with original papers).
- Not a deployment or inference server. The focus is on reproducible training, testing, and evaluation.

## Key Features

### Benchmark Mode

Run `python tools/benchmark.py --data_root data/mvtec_ad --methods all --categories all` to train and evaluate any subset of methods across all categories. Results are collected into a unified table with image-level and pixel-level metrics.

### Alignment Evidence

Every method has an alignment document under [`docs/alignment/`](../alignment/) that records:

- The original paper / code reference used
- Config-level hyperparameter provenance
- Code-path parity checks (forward pass, loss computation, scoring)
- Behavior probes comparing BaoIAD outputs to frozen reference outputs

### Config System

BaoIAD uses MMEngine's hierarchical config system with `_base_` inheritance. A typical method config is only a `model` definition — everything else (dataset, schedule, runtime) comes from base configs. See [Config System](user_guides/config.md) for details.

## Method Families

| Family | Methods |
|--------|---------|
| Feature-memory / density | PatchCore, PaDiM, DFM, DFKDE |
| Knowledge distillation | RD, RD++, EfficientAD, AST, DeSTSeg |
| Normalizing flow | FastFlow, CFlow, DifferNet, PyramidFlow, U-Flow |
| Reconstruction / ViT | UniAD, Dinomaly, ViTAD, MemSeg, GANomaly |
| Self-supervised synthesis | DRAEM, GLASS, DSR, CutPaste, NSA |
| Discriminative | SimpleNet, SuperSimpleNet, CFA |
| Vision-language / foundation | WinCLIP, AnomalyCLIP, AnoVL, MuSc, SAA+, AdaCLIP, AACLIP |
| Few-shot / registration | RegAD, AnomalyDINO |
| Hybrid / unified | UniNet |

The full method inventory is in [`baoiad/method_inventory.py`](../../baoiad/method_inventory.py).

## Architecture Overview

BaoIAD follows the MMEngine paradigm with registries scoped to `baoiad`:

```
baoiad/
├── registry.py          # MODELS, DATASETS, TRANSFORMS, METRICS, HOOKS, ...
├── method_inventory.py  # 37 method entries with config paths and family tags
├── models/
│   └── detectors/       # One module per method (patchcore.py, rd.py, ...)
├── datasets/
│   ├── base_ad_dataset.py
│   ├── mvtec_ad.py      # MVTecADDataset
│   ├── visa.py           # VisADataset
│   └── ...
├── evaluation/
│   └── anomaly_metric.py # AnomalyDetectionMetric
└── visualization/
    └── ad_visualizer.py  # ADVisualizer, ADVisualizationHook
```

### Core Concepts

- **Registry**: All components (models, datasets, transforms, metrics, hooks) are registered via [`baoiad/registry.py`](../../baoiad/registry.py) using MMEngine's `Registry` with `scope='baoiad'`.
- **Base AD Model**: All detectors follow a `backbone → neck → head` pattern. The base class handles freezing, forward dispatch (train vs. test), and score aggregation.
- **Data Pipeline**: A standard `LoadImage → LoadMask → ResizeAD → NormalizeAD → PackADInputs` pipeline shared across all datasets. Some methods (e.g., DRAEM, GLASS) add augmentation transforms.
- **Training Loop**: MMEngine `Runner` drives training. Custom hooks like `MemoryBankHook` handle method-specific logic (e.g., PatchCore memory bank construction).
- **Evaluation**: `AnomalyDetectionMetric` computes image-level metrics (AUROC, F1-max, AP, ECE, FPR@95TPR) and pixel-level metrics (AUROC, F1-max, AP, AUPRO, AUPIMO, ECE).

## Supported Datasets

See [Dataset Zoo](dataset_zoo.md) for the full list of 10 datasets and [Prepare Datasets](user_guides/prepare_dataset.md) for download and setup instructions.

## Quick Start

See [Get Started](get_started.md) for installation and your first training run.
