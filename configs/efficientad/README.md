## EfficientAD

> EfficientAD: Accurate Visual Anomaly Detection at Millisecond Level

- **Paper**: [publication](https://arxiv.org/abs/2303.14535)
- **Implementation source**: [upstream repository](https://github.com/open-edge-platform/anomalib); revision: `4f6af1acb0ee7b81f54cda036dd9f1c27f63b69a`
- **Category**: Knowledge Distillation
- **Backbone**: PDN + WRN-50-2

EfficientAD combines a lightweight patch description network (PDN) with a student-teacher distillation branch for anomaly detection. The PDN provides fast feature extraction, while the student-teacher branch captures more subtle anomalies through knowledge distillation. During training, the student network learns to match the teacher's output on normal images, and both branches are trained jointly. At inference, anomaly scores combine the PDN's distance to the normal feature distribution and the student-teacher discrepancy, achieving millisecond-level inference.

### Configs

| Config | Description |
|--------|-------------|
| [`efficientad_256_mvtec_strict.py`](efficientad_256_mvtec_strict.py) | MVTec AD reference configuration |
| [`efficientad_256_visa.py`](efficientad_256_visa.py) | VisA |

<!-- BaoIAD repo-local evidence: start -->

### MVTec AD result summary

| #cat | img mean±std | pxl mean±std | AUPRO mean±std | AUPIMO mean±std | iECE mean±std | pECE mean±std |
|---|---|---|---|---|---|---|
| 15 | 0.9795±0.0000 | 0.9585±0.0000 | 0.8933±0.0000 | 0.5812±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 |

### VisA result summary

| img_AUROC | pxl_AUROC | img_F1max | pxl_F1max | img_AP | pxl_AP | AUPRO | cats |
|---|---|---|---|---|---|---|---|
| 0.9193 | 0.9786 | 0.8891 | 0.4121 | 0.9407 | 0.3432 | 0.8598 | 12/12 |

### Speed summary

| avg_ms_per_img | fps | img_size | forward_mode |
|---|---|---|---|
| 3.36 | 297.4 | 256 | predict |

### Alignment note

This method is part of the repo-local BaoIAD inventory under the **Knowledge distillation** family. The [implementation provenance and reproducibility record](../../docs/alignment/efficientad.md) is indexed by the [public method-status manifest](../../docs/alignment/method_status.json). Its manifest validation state is **partially verified**. Referenced raw evidence is not distributed with this repository. The result tables above are historical repository summaries, not a unified public ranking, and should be interpreted according to that manifest state.

<!-- BaoIAD repo-local evidence: end -->
