## CFA

> CFA: Coupled-Hypersphere-based Feature Adaptation for Anomaly Detection

- **Paper**: [publication](https://arxiv.org/abs/2206.04325)
- **Implementation source**: [upstream repository](https://github.com/open-edge-platform/anomalib); revision: `4f6af1acb0ee7b81f54cda036dd9f1c27f63b69a`
- **Category**: Discriminator
- **Backbone**: WRN-50-2

CFA adapts features into coupled hyperspheres in a projected space for anomaly detection. The method learns to map normal features onto compact hypersphere clusters while anomalous features fall outside these clusters. During training, CFA optimizes the projection to minimize the distance of normal features to their assigned hypersphere centers using a coupled loss function. At inference, anomaly scores are computed as the distance from test features to the nearest hypersphere center, with large distances indicating anomalies.

### Configs

| Config | Description |
|--------|-------------|
| [`cfa_256_mvtec_strict.py`](cfa_256_mvtec_strict.py) | MVTec AD reference configuration |
| [`cfa_256_visa.py`](cfa_256_visa.py) | VisA |

<!-- BaoIAD repo-local evidence: start -->

### MVTec AD result summary

| #cat | img mean±std | pxl mean±std | AUPRO mean±std | AUPIMO mean±std | iECE mean±std | pECE mean±std |
|---|---|---|---|---|---|---|
| 15 | 0.9578±0.0006 | 0.9793±0.0001 | 0.9291±0.0003 | 0.5345±0.0061 | 0.0736±0.0184 | 0.0563±0.0197 |

### VisA result summary

| img_AUROC | pxl_AUROC | img_F1max | pxl_F1max | img_AP | pxl_AP | AUPRO | cats |
|---|---|---|---|---|---|---|---|
| 0.9131 | 0.9818 | 0.8864 | 0.3935 | 0.9294 | 0.3400 | 0.8716 | 12/12 |

### Speed summary

| avg_ms_per_img | fps | img_size | forward_mode |
|---|---|---|---|
| 9.4 | 106.4 | 256 | tensor |

### Alignment note

This method is part of the repo-local BaoIAD inventory under the **Discriminative** family. The [implementation provenance and reproducibility record](../../docs/alignment/cfa.md) is indexed by the [public method-status manifest](../../docs/alignment/method_status.json). Its manifest validation state is **partially verified**. Referenced raw evidence is not distributed with this repository. The result tables above are historical repository summaries, not a unified public ranking, and should be interpreted according to that manifest state.

<!-- BaoIAD repo-local evidence: end -->
