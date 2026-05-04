## DifferNet

> Same Same But DifferNet: Unsupervised Anomaly Detection

- **Paper**: [Same Same But DifferNet: Unsupervised Anomaly Detection](https://arxiv.org/abs/2008.12577)
- **Category**: Normalizing Flow
- **Backbone**: ResNet-18

DifferNet uses a pre-trained teacher and a student network with different architectures for anomaly detection. The architectural difference creates an asymmetric feature mapping where the student can only learn normal patterns. During training, the student network is trained to match the teacher's output on normal images. At inference, the discrepancy between student and teacher features serves as the anomaly score, with larger differences indicating anomalous regions.

### Configs

| Config | Description |
|--------|-------------|
| [`differnet_alexnet_256_mvtec_strict.py`](differnet_alexnet_256_mvtec_strict.py) | MVTec AD strict alignment |
| [`differnet_alexnet_256_visa.py`](differnet_alexnet_256_visa.py) | VisA |

<!-- BaoIAD repo-local evidence: start -->

### MVTec AD result summary

| #cat | img mean±std | pxl mean±std | AUPRO mean±std | AUPIMO mean±std | iECE mean±std | pECE mean±std |
|---|---|---|---|---|---|---|
| 15 | 0.9461±0.0014 | 0.7366±0.0005 | 0.2333±0.0028 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 |

### VisA result summary

| img_AUROC | pxl_AUROC | img_F1max | pxl_F1max | img_AP | pxl_AP | AUPRO | cats |
|---|---|---|---|---|---|---|---|
| 0.8947 | 0.8038 | 0.8698 | 0.1077 | 0.9026 | 0.0620 | 0.2325 | 12/12 |

### Speed summary

| avg_ms_per_img | fps | img_size | forward_mode |
|---|---|---|---|
| 627.49 | 1.6 | 256 | predict |

### Alignment note

This method is part of the repo-local BaoIAD inventory under the **Normalizing flow** family. The alignment record is [`docs/alignment/differnet.md`](../../docs/alignment/differnet.md); it preserves the detailed strict-alignment evidence, including reference freeze notes, code-path checks, probes, and archived benchmark stop-lines.

<!-- BaoIAD repo-local evidence: end -->
