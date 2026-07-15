## UniAD

> UniAD: A Unified Model for Multi-class Anomaly Detection

- **Paper**: [publication](https://arxiv.org/abs/2206.03687)
- **Implementation source**: [upstream repository](https://github.com/zhangzjn/ADer); revision: `902937a7ed7fa7689674a4ac9b8fe9a72a40c402`
- **Category**: Other
- **Backbone**: EffNet-B4

UniAD performs multi-class anomaly detection using a unified transformer-based architecture that reconstructs normal features across all categories in a single model. The key innovation is using neighbor masking and query-based reconstruction to handle multiple categories simultaneously without category-specific components. During training, the transformer learns to reconstruct normal features from all categories using masked attention and learnable queries. At inference, the reconstruction error serves as the anomaly score, with a single model detecting anomalies across all trained categories.

### Configs

| Config | Description |
|--------|-------------|
| [`uniad_wrn50_256_mvtec_strict.py`](uniad_wrn50_256_mvtec_strict.py) | MVTec AD reference configuration |
| [`uniad_wrn50_256_visa.py`](uniad_wrn50_256_visa.py) | VisA |

<!-- BaoIAD repo-local evidence: start -->

### MVTec AD result summary

| #cat | img mean±std | pxl mean±std | AUPRO mean±std | AUPIMO mean±std | iECE mean±std | pECE mean±std |
|---|---|---|---|---|---|---|
| 15 | 0.9321±0.0012 | 0.9620±0.0010 | 0.8965±0.0008 | 0.1957±0.0406 | 0.0000±0.0000 | 0.0000±0.0000 |

### VisA result summary

| img_AUROC | pxl_AUROC | img_F1max | pxl_F1max | img_AP | pxl_AP | AUPRO | cats |
|---|---|---|---|---|---|---|---|
| 0.8133 | 0.9673 | 0.8295 | 0.3213 | 0.8302 | 0.2653 | 0.8112 | 12/12 |

### Speed summary

| avg_ms_per_img | fps | img_size | forward_mode |
|---|---|---|---|
| 15.74 | 63.5 | 256 | predict |

### Alignment note

This method is part of the repo-local BaoIAD inventory under the **Reconstruction / ViT** family. The [implementation provenance and reproducibility record](../../docs/alignment/uniad.md) is indexed by the [public method-status manifest](../../docs/alignment/method_status.json). Its manifest validation state is **historical evidence**. Referenced raw evidence is not distributed with this repository. The result tables above are historical repository summaries, not a unified public ranking, and should be interpreted according to that manifest state.

<!-- BaoIAD repo-local evidence: end -->
