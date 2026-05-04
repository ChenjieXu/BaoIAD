## CutPaste

> CutPaste: Self-Supervised Learning for Anomaly Detection and Segmentation

- **Paper**: [CutPaste: Self-Supervised Learning for Anomaly Detection and Segmentation](https://arxiv.org/abs/2104.02515)
- **Category**: Other
- **Backbone**: ResNet-18

CutPaste creates synthetic anomalies by cut-and-paste augmentation of training patches, then trains a binary classifier to distinguish normal from augmented samples. The key idea is that self-supervised classification on synthetic anomalies provides a good proxy for real anomaly detection. During training, patches are either cut out and pasted at random locations (CutPaste) or swapped between images (CutSwap), and a classifier learns to detect these perturbations. At inference, the classifier's anomaly probability serves as the detection score.

### Configs

| Config | Description |
|--------|-------------|
| [`cutpaste_rn18_256_mvtec_strict.py`](cutpaste_rn18_256_mvtec_strict.py) | MVTec AD strict alignment |
| [`cutpaste_rn18_256_visa.py`](cutpaste_rn18_256_visa.py) | VisA |

<!-- BaoIAD repo-local evidence: start -->

### MVTec AD result summary

| #cat | img mean±std | pxl mean±std | AUPRO mean±std | AUPIMO mean±std | iECE mean±std | pECE mean±std |
|---|---|---|---|---|---|---|
| 15 | 0.8658±0.0252 | 0.6391±0.0088 | 0.2230±0.0119 | 0.0000±0.0000 | 0.1805±0.0037 | 0.5653±0.0052 |

### VisA result summary

| img_AUROC | pxl_AUROC | img_F1max | pxl_F1max | img_AP | pxl_AP | AUPRO | cats |
|---|---|---|---|---|---|---|---|
| 0.8061 | 0.7365 | 0.8048 | 0.0915 | 0.8480 | 0.0461 | 0.2457 | 12/12 |

### Speed summary

| avg_ms_per_img | fps | img_size | forward_mode |
|---|---|---|---|
| 2.13 | 469.9 | 256 | predict |

### Alignment note

This method is part of the repo-local BaoIAD inventory under the **Self-supervised synthesis** family. The alignment record is [`docs/alignment/cutpaste.md`](../../docs/alignment/cutpaste.md); it preserves the detailed strict-alignment evidence, including reference freeze notes, code-path checks, probes, and archived benchmark stop-lines.

<!-- BaoIAD repo-local evidence: end -->
