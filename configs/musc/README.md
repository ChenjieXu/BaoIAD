## MuSc

> MuSc: Mutual Scoring of Pseudo Pairs for Zero-Shot Anomaly Classification and Segmentation

- **Paper**: [publication](https://arxiv.org/abs/2401.16753)
- **Implementation source**: [upstream repository](https://github.com/xrli-U/MuSc); revision: `72d58ad56c0cafa2b056bd0aa7676f9c21fccbc4`
- **Category**: Vision-Language
- **Backbone**: OpenCLIP-ViT/L-14

MuSc performs zero-shot anomaly detection by mutual scoring of pseudo-pairs between test and reference images using CLIP features. The key idea is to score each test patch against all reference patches without any training, using CLIP's pre-aligned visual features. No training is required — the method computes pairwise similarity scores between test and reference CLIP features at inference time. Anomaly scores are derived from the minimum similarity to reference patches, with both image-level and pixel-level scores available.

### Configs

| Config | Description |
|--------|-------------|
| [`musc_vitl14_336_518_mvtec_strict.py`](musc_vitl14_336_518_mvtec_strict.py) | MVTec AD reference configuration |
| [`musc_vitl14_336_518_visa.py`](musc_vitl14_336_518_visa.py) | VisA |

<!-- BaoIAD repo-local evidence: start -->

### MVTec AD result summary

| #cat | img mean±std | pxl mean±std | AUPRO mean±std | AUPIMO mean±std | iECE mean±std | pECE mean±std |
|---|---|---|---|---|---|---|
| 15 | 0.9780±0.0000 | 0.9711±0.0000 | 0.9402±0.0000 | 0.5661±0.0000 | 0.3466±0.0000 | 0.3250±0.0000 |

### VisA result summary

| img_AUROC | pxl_AUROC | img_F1max | pxl_F1max | img_AP | pxl_AP | AUPRO | cats |
|---|---|---|---|---|---|---|---|
| 0.9103 | 0.9888 | 0.8744 | 0.4593 | 0.9187 | 0.4166 | 0.9295 | 12/12 |

### Speed summary

| avg_ms_per_img | fps | img_size | forward_mode |
|---|---|---|---|
| 93.19 | 10.7 | 518 | predict |

### Alignment note

This method is part of the repo-local BaoIAD inventory under the **Vision-language / foundation** family. The [implementation provenance and reproducibility record](../../docs/alignment/musc.md) is indexed by the [public method-status manifest](../../docs/alignment/method_status.json). Its manifest validation state is **partially verified**. Referenced raw evidence is not distributed with this repository. The result tables above are historical repository summaries, not a unified public ranking, and should be interpreted according to that manifest state.

<!-- BaoIAD repo-local evidence: end -->
