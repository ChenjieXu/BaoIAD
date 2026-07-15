## AnomalyDINO

> AnomalyDINO: DINOv2 for Anomaly Detection

- **Paper**: [publication](https://arxiv.org/abs/2405.14529)
- **Implementation source**: [upstream repository](https://github.com/dammsi/AnomalyDINO); revision: `b9d1c2648e3a5247437d4d953d907a8f3d994457`
- **Category**: Vision-Language
- **Backbone**: DINOv2-ViT/L-14

AnomalyDINO leverages DINOv2's self-supervised features for anomaly detection by computing patch-level distances between test and reference features. The key insight is that DINOv2 produces semantically meaningful features without supervision, making it naturally suited for anomaly detection. No training is required — the method stores reference patch features from normal images and compares test features against them. At inference, anomaly scores are computed as the minimum distance between each test patch feature and the reference feature bank, with both image-level and pixel-level scores derived from these distances.

### Configs

| Config | Description |
|--------|-------------|
| [`anomalydino_vitb14_448_mvtec_strict.py`](anomalydino_vitb14_448_mvtec_strict.py) | MVTec AD reference configuration |
| [`anomalydino_vitb14_448_visa.py`](anomalydino_vitb14_448_visa.py) | VisA |

<!-- BaoIAD repo-local evidence: start -->

### MVTec AD result summary

| #cat | img mean±std | pxl mean±std | AUPRO mean±std | AUPIMO mean±std | iECE mean±std | pECE mean±std |
|---|---|---|---|---|---|---|
| 15 | 0.9721±0.0000 | 0.9570±0.0000 | 0.9095±0.0000 | 0.5673±0.0000 | 0.4073±0.0000 | 0.0958±0.0000 |

### VisA result summary

| img_AUROC | pxl_AUROC | img_F1max | pxl_F1max | img_AP | pxl_AP | AUPRO | cats |
|---|---|---|---|---|---|---|---|
| 0.8284 | 0.9589 | 0.8204 | 0.3816 | 0.8354 | 0.3266 | 0.8549 | 12/12 |

### Speed summary

| avg_ms_per_img | fps | img_size | forward_mode |
|---|---|---|---|
| 55.53 | 18.0 | 448 | tensor |

### Alignment note

This method is part of the repo-local BaoIAD inventory under the **Few-shot / registration** family. The [implementation provenance and reproducibility record](../../docs/alignment/anomalydino.md) is indexed by the [public method-status manifest](../../docs/alignment/method_status.json). Its manifest validation state is **partially verified**. Referenced raw evidence is not distributed with this repository. The result tables above are historical repository summaries, not a unified public ranking, and should be interpreted according to that manifest state.

<!-- BaoIAD repo-local evidence: end -->
