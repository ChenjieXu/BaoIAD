## DeSTSeg

> DeSTSeg: Segmentation Guided Denoising for Anomaly Detection

- **Paper**: [DeSTSeg: Segmentation Guided Denoising for Anomaly Detection](https://arxiv.org/abs/2304.08401)
- **Category**: Reconstruction
- **Backbone**: ResNet-18

DeSTSeg combines a denoising student-teacher network with a segmentation module for anomaly detection. The student network learns to denoise the teacher's features, and the denoising error highlights anomalous regions where the student fails to reconstruct normal patterns. During training, both the denoising network and segmentation head are trained jointly, with the denoising error providing a training signal for segmentation. At inference, the segmentation network directly outputs pixel-level anomaly predictions, guided by the denoising error.

### Configs

| Config | Description |
|--------|-------------|
| [`destseg_rn18_256_mvtec_strict.py`](destseg_rn18_256_mvtec_strict.py) | MVTec AD strict alignment |
| [`destseg_rn18_256_visa.py`](destseg_rn18_256_visa.py) | VisA |

<!-- BaoIAD repo-local evidence: start -->

### MVTec AD result summary

| #cat | img mean±std | pxl mean±std | AUPRO mean±std | AUPIMO mean±std | iECE mean±std | pECE mean±std |
|---|---|---|---|---|---|---|
| 15 | 0.9792±0.0059 | 0.9795±0.0019 | 0.9281±0.0092 | 0.5480±0.0499 | 0.1390±0.0172 | 0.0140±0.0024 |

### VisA result summary

| img_AUROC | pxl_AUROC | img_F1max | pxl_F1max | img_AP | pxl_AP | AUPRO | cats |
|---|---|---|---|---|---|---|---|
| 0.9084 | 0.9690 | 0.8786 | 0.4601 | 0.9269 | 0.4070 | 0.8507 | 12/12 |

### Speed summary

| avg_ms_per_img | fps | img_size | forward_mode |
|---|---|---|---|
| 7.97 | 125.4 | 256 | predict |

### Alignment note

This method is part of the repo-local BaoIAD inventory under the **Knowledge distillation** family. The alignment record is [`docs/alignment/destseg.md`](../../docs/alignment/destseg.md); it preserves the detailed strict-alignment evidence, including reference freeze notes, code-path checks, probes, and archived benchmark stop-lines.

<!-- BaoIAD repo-local evidence: end -->
