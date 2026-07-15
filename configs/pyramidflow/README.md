## PyramidFlow

> PyramidFlow: High-Resolution Defect Contrastive Localization using Pyramid Normalizing Flow

- **Paper**: [publication](https://arxiv.org/abs/2303.02595)
- **Implementation source**: [upstream repository](https://github.com/gasharper/PyramidFlow); revision not pinned in the method-status manifest
- **Category**: Normalizing Flow
- **Backbone**: WRN-50-2

PyramidFlow performs anomaly detection using pyramid normalizing flows that model multi-scale feature distributions. The pyramid architecture captures feature distributions at multiple resolutions simultaneously, enabling high-resolution defect localization that single-scale flows cannot achieve. During training, the pyramid flow learns to map multi-scale normal features to a standard normal distribution. At inference, the negative log-likelihood at each scale and position provides the anomaly map, with high-resolution localization from the finest pyramid level.

### Configs

| Config | Description |
|--------|-------------|
| [`pyramidflow_fnf_256_mvtec_strict.py`](pyramidflow_fnf_256_mvtec_strict.py) | MVTec AD reference configuration (FNF backbone) |
| [`pyramidflow_resnet18_1024_mvtec_strict.py`](pyramidflow_resnet18_1024_mvtec_strict.py) | MVTec AD reference configuration |
| [`pyramidflow_resnet18_1024_visa.py`](pyramidflow_resnet18_1024_visa.py) | VisA |

<!-- BaoIAD repo-local evidence: start -->

### MVTec AD result summary

| #cat | img mean±std | pxl mean±std | AUPRO mean±std | AUPIMO mean±std | iECE mean±std | pECE mean±std |
|---|---|---|---|---|---|---|
| 15 | 0.8581±0.0389 | 0.9429±0.0183 | 0.8061±0.0466 | 0.3104±0.0049 | 0.5889±0.1061 | 0.0732±0.0374 |

### VisA result summary

| img_AUROC | pxl_AUROC | img_F1max | pxl_F1max | img_AP | pxl_AP | AUPRO | cats |
|---|---|---|---|---|---|---|---|
| 0.9034 | 0.9643 | 0.8638 | 0.3107 | 0.9108 | 0.2304 | 0.8499 | 12/12 |

### Speed summary

| avg_ms_per_img | fps | img_size | forward_mode |
|---|---|---|---|
| Unavailable | Unavailable | Unavailable | Unavailable |

No speed benchmark is available for PyramidFlow, so no latency or FPS value is reported.

### Alignment note

This method is part of the repo-local BaoIAD inventory under the **Normalizing flow** family. The [implementation provenance and reproducibility record](../../docs/alignment/pyramidflow.md) is indexed by the [public method-status manifest](../../docs/alignment/method_status.json). Its manifest validation state is **partially verified**. Referenced raw evidence is not distributed with this repository. The result tables above are historical repository summaries, not a unified public ranking, and should be interpreted according to that manifest state.

<!-- BaoIAD repo-local evidence: end -->
