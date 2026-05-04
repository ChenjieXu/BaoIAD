## ViTAD

> ViTAD: Vision Transformer for Anomaly Detection

- **Paper**: [ViTAD: Vision Transformer for Anomaly Detection](https://arxiv.org/abs/2404.09163)
- **Category**: Other
- **Backbone**: ViT-AD

ViTAD uses a Vision Transformer as the backbone for anomaly detection with a masked knowledge distillation scheme. The key innovation is using masked patches during distillation to force the student to learn robust normal representations, preventing it from simply copying the teacher's behavior. During training, the student ViT learns to reconstruct the teacher's features from partially masked input patches on normal images. At inference, the feature discrepancy between student and teacher serves as the anomaly score, with both image-level and pixel-level localization.

### Configs

| Config | Description |
|--------|-------------|
| [`vitad_256_mvtec_strict.py`](vitad_256_mvtec_strict.py) | MVTec AD strict alignment |
| [`vitad_256_visa.py`](vitad_256_visa.py) | VisA |

<!-- BaoIAD repo-local evidence: start -->

### MVTec AD result summary

| #cat | img mean±std | pxl mean±std | AUPRO mean±std | AUPIMO mean±std | iECE mean±std | pECE mean±std |
|---|---|---|---|---|---|---|
| 15 | 0.9820±0.0012 | 0.9765±0.0002 | 0.9136±0.0006 | 0.6330±0.0074 | 0.6829±0.0002 | 0.0224±0.0000 |

### VisA result summary

| img_AUROC | pxl_AUROC | img_F1max | pxl_F1max | img_AP | pxl_AP | AUPRO | cats |
|---|---|---|---|---|---|---|---|
| 0.9092 | 0.9812 | 0.8707 | 0.4174 | 0.9230 | 0.3692 | 0.8395 | agg |

### Speed summary

| avg_ms_per_img | fps | img_size | forward_mode |
|---|---|---|---|
| 8.59 | 116.4 | 256 | predict |

### Alignment note

This method is part of the repo-local BaoIAD inventory under the **Reconstruction / ViT** family. The alignment record is [`docs/alignment/vitad.md`](../../docs/alignment/vitad.md); it preserves the detailed strict-alignment evidence, including reference freeze notes, code-path checks, probes, and archived benchmark stop-lines.

<!-- BaoIAD repo-local evidence: end -->
