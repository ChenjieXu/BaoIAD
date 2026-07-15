## UniNet

> UniNet: Unified Architecture for Anomaly Detection

- **Paper**: [publication](https://openaccess.thecvf.com/content/CVPR2025/html/Wei_UniNet_A_Contrastive_Learning-guided_Unified_Framework_with_Feature_Selection_for_CVPR_2025_paper.html)
- **Implementation source**: [upstream repository](https://github.com/open-edge-platform/anomalib); revision: `4f6af1acb0ee7b81f54cda036dd9f1c27f63b69a`
- **Category**: Other
- **Backbone**: WRN-50-2

UniNet provides a unified architecture for anomaly detection that supports multiple detection paradigms through a shared backbone with task-specific heads. The key idea is sharing feature extraction across different detection methods while maintaining specialized scoring heads for each paradigm. During training, the shared backbone and task-specific heads are trained jointly on normal images from multiple categories. At inference, the appropriate head produces anomaly scores based on the shared features, supporting multiple detection modes in one model.

### Configs

| Config | Description |
|--------|-------------|
| [`uninet_256_mvtec_strict.py`](uninet_256_mvtec_strict.py) | MVTec AD reference configuration |
| [`uninet_256_visa.py`](uninet_256_visa.py) | VisA |

<!-- BaoIAD repo-local evidence: start -->

### MVTec AD result summary

| #cat | img mean±std | pxl mean±std | AUPRO mean±std | AUPIMO mean±std | iECE mean±std | pECE mean±std |
|---|---|---|---|---|---|---|
| 15 | 0.9832±0.0003 | 0.9784±0.0021 | 0.9275±0.0063 | 0.5423±0.0277 | 0.0000±0.0000 | 0.0000±0.0000 |

### VisA result summary

| img_AUROC | pxl_AUROC | img_F1max | pxl_F1max | img_AP | pxl_AP | AUPRO | cats |
|---|---|---|---|---|---|---|---|
| 0.9733 | 0.9826 | 0.9392 | 0.4608 | 0.9779 | 0.4214 | 0.9067 | 12/12 |

### Speed summary

| avg_ms_per_img | fps | img_size | forward_mode |
|---|---|---|---|
| 14.89 | 67.1 | 256 | predict |

### Alignment note

This method is part of the repo-local BaoIAD inventory under the **Hybrid / unified** family. The [implementation provenance and reproducibility record](../../docs/alignment/uninet.md) is indexed by the [public method-status manifest](../../docs/alignment/method_status.json). Its manifest validation state is **historical evidence**. Referenced raw evidence is not distributed with this repository. The result tables above are historical repository summaries, not a unified public ranking, and should be interpreted according to that manifest state.

<!-- BaoIAD repo-local evidence: end -->
