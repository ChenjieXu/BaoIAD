# Changelog

## v1.1.0 — Unreleased

This release candidate prepares BaoIAD for its public organization release.
The release date, source commit, and version-specific Zenodo DOI will be added
only after the final v1.1.0 archive is published. Until then, citation metadata
uses the verified Zenodo concept DOI for all BaoIAD versions.

The historical Git tag `v1.0.0` peels to commit
`697fc4304cc76876d397067e2706ed771f62e708`, whose package metadata reports
version `0.1.0`. Version `1.1.0` therefore records the first organization
release without rewriting or reusing that historical package identity. The
machine-readable [v1.0.0 compatibility contract](../../alignment/v1_0_0_compatibility.json)
locks the retained paths and CLI surface and documents intentional migrations.

### Compatibility and migration

- Python 3.10 or newer is required; release checks target Python 3.10 and 3.12.
- The core environment uses `mmcv-lite>=2.0` and does not require compiled MMCV
  operators.
- RegAD's deterministic fallback remains available by default. Set
  `strict_require_official_support_set=True` to require the official support
  set and fail instead of falling back.
- ViTAD exact-order evaluation requires a user-provided, verified
  `--order-file`; BaoIAD does not generate or redistribute that upstream file.
- Checkpoint loading is restricted by default. Use `--trusted-checkpoint` only
  for a verified legacy pickle checkpoint, because loading it may execute code.
- Unsupported public extras from older metadata (`mamba`, `mmpretrain`, and
  `faiss-gpu`) are no longer declared. Use the supported CPU FAISS extra or
  install method-specific dependencies explicitly when following upstream code.
- The legacy `imgaug` extra is no longer declared because release-tested
  packaged paths use local SciPy/torchvision augmentation alternatives. Legacy
  callers should migrate to those paths instead of relying on the old extra.

## v1.0.0 — Initial Release

Release of the BaoIAD unified industrial anomaly detection benchmark.

### Highlights

- **37 methods** across 9 families: feature-memory, knowledge distillation, normalizing flow, reconstruction, self-supervised synthesis, discriminative, vision-language, few-shot, and hybrid/unified.
- **10 supported datasets**: MVTec AD, VisA, BTech, MVTec 3D AD, MVTec LOCO, MPDD, MVTec AD 2, Kolektor, VAD, Real-IAD.
- **Unified evaluation pipeline** with image-level metrics (AUROC, F1-max, AP, ECE, FPR@95TPR) and pixel-level metrics (AUROC, F1-max, AP, AUPRO, AUPIMO, ECE).
- **Implementation provenance and reproducibility records** for the 37-method inventory, with method-specific validation states and known limitations.
- **Benchmark runner** (`tools/benchmark.py`) for large-scale evaluation across methods and categories.
- **Speed benchmark** support for inference latency and throughput measurement.
- **MMEngine-based architecture** with scoped registries for models, datasets, transforms, metrics, hooks, and visualizers.
- **Per-method config READMEs** and alignment records in `docs/alignment/`.

### Included Methods

| Family | Methods |
|--------|---------|
| Feature-memory / density | PatchCore, PaDiM, DFM, DFKDE |
| Knowledge distillation | RD, RD++, AST, EfficientAD, DeSTSeg |
| Normalizing flow | FastFlow, CFlow, DifferNet, U-Flow, PyramidFlow |
| Reconstruction / ViT | Dinomaly, ViTAD, MemSeg, UniAD, GANomaly |
| Self-supervised synthesis | DRAEM, GLASS, DSR, CutPaste, NSA |
| Discriminative | SimpleNet, SuperSimpleNet, CFA |
| Vision-language / foundation | WinCLIP, AnomalyCLIP, MuSc, AACLIP, AnoVL, AdaCLIP, SAA+ |
| Few-shot / registration | AnomalyDINO, RegAD |
| Hybrid / unified | UniNet |
