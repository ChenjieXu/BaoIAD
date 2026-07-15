## DeSTSeg

> DeSTSeg: Segmentation Guided Denoising for Anomaly Detection

- **Paper**: [publication](https://arxiv.org/abs/2211.11317)
- **Implementation source**: [upstream repository](https://github.com/apple/ml-destseg); revision: `f6ea31fb5b097698b195f85b1d5e3efaedce9eb6`
- **Category**: Reconstruction
- **Backbone**: ResNet-18

DeSTSeg combines a denoising student-teacher network with a segmentation module for anomaly detection. The student network learns to denoise the teacher's features, and the denoising error highlights anomalous regions where the student fails to reconstruct normal patterns. During training, both the denoising network and segmentation head are trained jointly, with the denoising error providing a training signal for segmentation. At inference, the segmentation network directly outputs pixel-level anomaly predictions, guided by the denoising error.

### Configs

| Config | Description |
|--------|-------------|
| [`destseg_rn18_256_mvtec_strict.py`](destseg_rn18_256_mvtec_strict.py) | MVTec AD reference configuration |
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

This method is part of the repo-local BaoIAD inventory under the **Knowledge distillation** family. The [implementation provenance and reproducibility record](../../docs/alignment/destseg.md) is indexed by the [public method-status manifest](../../docs/alignment/method_status.json). Its manifest validation state is **partially verified**. Referenced raw evidence is not distributed with this repository. The result tables above are historical repository summaries, not a unified public ranking, and should be interpreted according to that manifest state.

<!-- BaoIAD repo-local evidence: end -->
