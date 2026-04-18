# Overview

BaoIAD is a unified benchmarking framework for **industrial anomaly detection (IAD)**, built on top of [MMEngine](https://github.com/open-mmlab/mmengine) following the OpenMMLab style. It integrates **50+ anomaly detection methods** under a single config-driven interface, enabling fair comparison across datasets including MVTec AD, VisA, BTech, and more.

## Key Features

- **50+ methods**: PatchCore, RD, Dinomaly, AnomalyCLIP, SimpleNet, DRAEM, and many more
- **Unified interface**: Config-driven, MMEngine-style training and evaluation
- **Fair comparison**: Standardized backbones, consistent evaluation metrics, controlled seeds
- **OpenMMLab ecosystem**: Registry, config inheritance, modular components
- **Comprehensive metrics**: Image-level (AUROC, F1-max, AP, ECE, FPR@95TPR) and pixel-level (AUROC, F1-max, AP, AUPRO, AUPIMO, ECE)
- **Multiple datasets**: MVTec AD, VisA, BTech, MVTec 3D AD, MVTec LOCO, MPDD, MVTec AD 2, Kolektor, VAD, RealIAD

## Architecture

BaoIAD follows the **backbone -> neck -> head** pipeline pattern from the OpenMMLab ecosystem:

```
Input Image
    |
    v
+----------+     +-------+     +------+
| Backbone | --> | Neck  | --> | Head |
+----------+     +-------+     +------+
    |                              |
    | (frozen)                     |
    v                              v
  Features                    Loss / Predict
```

### Core Components

- **BaseADModel**: Base class with 3-mode forward dispatch (`loss`, `predict`, `tensor`)
- **6 specialized sub-classes**: MemoryBankADModel, KnowledgeDistillationADModel, FlowBasedADModel, ReconstructionADModel, VisionLanguageADModel, DiscriminatorADModel
- **ADDataSample**: Data structure carrying ground-truth and prediction fields
- **AnomalyDetectionMetric**: Unified metric computing 12 image/pixel-level metrics
- **MemoryBankHook**: Lifecycle hook for memory bank construction after training

### Config System

BaoIAD uses MMEngine's config inheritance system. Each method config inherits from shared base configs:

```
configs/
├── _base_/
│   ├── backbones/        # Shared backbone definitions
│   ├── datasets/         # Dataset + dataloader + evaluator configs
│   ├── schedules/        # Optimizer + LR scheduler configs
│   └── default_runtime.py  # Scope, seed, hooks
└── <method>/
    ├── <method>_mvtec_strict.py   # MVTec reference config
    └── <method>_visa.py           # VisA config
```

### Method Categories

| Category | Methods | Base Class |
|----------|---------|------------|
| Memory Bank | PatchCore, SPADE, PaDiM, DFM, DFKDE, RegAD, GraphCore | `MemoryBankADModel` |
| Knowledge Distillation | RD, RD++, STFPM, EfficientAD, Dinomaly | `KnowledgeDistillationADModel` |
| Normalizing Flow | CSFlow, FastFlow, CFlow, UFlow, DifferNet, PyramidFlow | `FlowBasedADModel` |
| Reconstruction | DRAEM, MemSeg, DeSTSeg, MemAE, FRE, GANomaly, DSR | `ReconstructionADModel` |
| Vision-Language | WinCLIP, AnomalyCLIP, AnoVL, MuSc, AdaCLIP, AACLIP, AnomalyDINO | `VisionLanguageADModel` |
| Discriminator | SimpleNet, SuperSimpleNet, CFA | `DiscriminatorADModel` |
| Other | InvAD, ViTAD, UniAD, MambaAD, NSA, ResAD, CutPaste, GLASS, AST, UniNet, UniVAD | `BaseADModel` |
