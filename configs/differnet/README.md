## DifferNet

> Same Same But DifferNet: Unsupervised Anomaly Detection

- **Paper**: [publication](https://arxiv.org/abs/2008.12577)
- **Implementation source**: [upstream repository](https://github.com/marco-rudolph/differnet); revision: `9bdf02686297a093fb206ffeba64b1c0e78182b6`
- **Category**: Normalizing Flow
- **Backbone**: ResNet-18

DifferNet uses a pre-trained teacher and a student network with different architectures for anomaly detection. The architectural difference creates an asymmetric feature mapping where the student can only learn normal patterns. During training, the student network is trained to match the teacher's output on normal images. At inference, the discrepancy between student and teacher features serves as the anomaly score, with larger differences indicating anomalous regions.

### Configs

| Config | Description |
|--------|-------------|
| [`differnet_alexnet_256_mvtec_strict.py`](differnet_alexnet_256_mvtec_strict.py) | MVTec AD reference configuration |
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

This method is part of the repo-local BaoIAD inventory under the **Normalizing flow** family. The [implementation provenance and reproducibility record](../../docs/alignment/differnet.md) is indexed by the [public method-status manifest](../../docs/alignment/method_status.json). Its manifest validation state is **historical evidence**. Referenced raw evidence is not distributed with this repository. The result tables above are historical repository summaries, not a unified public ranking, and should be interpreted according to that manifest state.

<!-- BaoIAD repo-local evidence: end -->
