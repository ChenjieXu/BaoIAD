## Dinomaly

> Dinomaly: Less Is More in Anomaly Detection

- **Paper**: [Dinomaly: Less Is More in Anomaly Detection](https://arxiv.org/abs/2405.14525)
- **Category**: Knowledge Distillation
- **Backbone**: DINOv2-ViT-B/14

Dinomaly uses DINOv2 as a frozen encoder with a lightweight decoder for anomaly detection. The key innovation is using DINOv2's powerful self-supervised features with minimal architectural overhead — only a simple reconstruction decoder is trained. During training, the decoder learns to reconstruct DINOv2 features of normal images. At inference, the reconstruction error in feature space serves as the anomaly score, with both image-level and pixel-level localization.

### Configs

| Config | Description |
|--------|-------------|
| [`dinomaly_392_mvtec_strict.py`](dinomaly_392_mvtec_strict.py) | MVTec AD strict alignment |
| [`dinomaly_392_visa.py`](dinomaly_392_visa.py) | VisA |

<!-- BaoIAD repo-local evidence: start -->

### MVTec AD result summary

| #cat | img mean±std | pxl mean±std | AUPRO mean±std | AUPIMO mean±std | iECE mean±std | pECE mean±std |
|---|---|---|---|---|---|---|
| 15 | 0.9960±0.0000 | 0.9831±0.0000 | 0.9482±0.0000 | 0.8182±0.0000 | 0.5932±0.0000 | 0.0635±0.0000 |

### VisA result summary

| img_AUROC | pxl_AUROC | img_F1max | pxl_F1max | img_AP | pxl_AP | AUPRO | cats |
|---|---|---|---|---|---|---|---|
| 0.4184 | 0.6756 | 0.7203 | 0.0409 | 0.5390 | 0.0216 | 0.2700 | agg |

### Speed summary

| avg_ms_per_img | fps | img_size | forward_mode |
|---|---|---|---|
| 27.2 | 36.8 | 392 | predict |

### Alignment note

This method is part of the repo-local BaoIAD inventory under the **Reconstruction / ViT** family. The alignment record is [`docs/alignment/dinomaly.md`](../../docs/alignment/dinomaly.md); it preserves the detailed strict-alignment evidence, including reference freeze notes, code-path checks, probes, and archived benchmark stop-lines.

<!-- BaoIAD repo-local evidence: end -->
