# Changelog

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
