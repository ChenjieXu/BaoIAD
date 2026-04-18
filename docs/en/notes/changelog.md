# Changelog

## v0.1.0 (2025)

Initial release of BaoIAD.

### Methods (50+)

- **Memory Bank**: PatchCore, SPADE, PaDiM, DFM, DFKDE, RegAD, GraphCore
- **Knowledge Distillation**: RD, RD++, STFPM, EfficientAD, Dinomaly
- **Normalizing Flow**: CSFlow, FastFlow, CFlow, UFlow, DifferNet, PyramidFlow
- **Reconstruction**: DRAEM, MemSeg, DeSTSeg, MemAE, FRE, GANomaly, DSR
- **Vision-Language**: WinCLIP, AnomalyCLIP, AnoVL, MuSc, AdaCLIP, AACLIP, AnomalyDINO
- **Discriminator**: SimpleNet, SuperSimpleNet, CFA
- **Other**: InvAD, ViTAD, UniAD, MambaAD, NSA, ResAD, CutPaste, GLASS, AST, PNI, RealNet, ComposeAD, UniNet, UniVAD, SAA+

### Datasets

- MVTec AD, VisA, BTech, MVTec 3D AD, MVTec LOCO, MPDD, MVTec AD 2, Kolektor, VAD, RealIAD

### Features

- Unified MMEngine-based framework with config inheritance
- 6 specialized BaseADModel sub-classes for different method paradigms
- Comprehensive evaluation: 12 image/pixel-level metrics including AUPRO and AUPIMO
- MemoryBankHook for automatic memory bank lifecycle management
- Benchmark tool with per-category subprocess execution and JSON output
- Multi-GPU benchmark support
