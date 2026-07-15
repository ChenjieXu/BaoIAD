## RegAD

> RegAD: Registration Based Few-Shot Anomaly Detection

- **Paper**: [publication](https://arxiv.org/abs/2207.07361)
- **Implementation source**: [upstream repository](https://github.com/MediaBrain-SJTU/RegAD); revision: `5e2c1f8c18d302b0354471567846fee3ed2ff063`
- **Category**: Memory Bank
- **Backbone**: WRN-50-2

RegAD performs few-shot anomaly detection by learning spatial registration between test and support images. The key innovation is learning a category-agnostic registration network that aligns test images with normal reference images, making anomalies visible as registration errors. During training, the registration network learns to align normal image pairs using spatial transformations. At inference, the registration error between the test image and its nearest normal reference serves as the anomaly score, requiring only a few normal examples per category.

### Configs

| Config | Description |
|--------|-------------|
| [`regad_wrn50_256_mvtec_strict.py`](regad_wrn50_256_mvtec_strict.py) | MVTec AD reference configuration |
| [`regad_wrn50_256_visa.py`](regad_wrn50_256_visa.py) | VisA |

<!-- BaoIAD repo-local evidence: start -->

### MVTec AD result summary

| #cat | img mean±std | pxl mean±std | AUPRO mean±std | AUPIMO mean±std | iECE mean±std | pECE mean±std |
|---|---|---|---|---|---|---|
| 15 | 0.8711±0.0128 | 0.9512±0.0070 | — | — | — | — |

### VisA result summary

| img_AUROC | pxl_AUROC | img_F1max | pxl_F1max | img_AP | pxl_AP | AUPRO | cats |
|---|---|---|---|---|---|---|---|
| 0.7665 | 0.9641 | — | — | — | — | — | 12/12 |

### Speed summary

| avg_ms_per_img | fps | img_size | forward_mode |
|---|---|---|---|
| 4.2 | 238.2 | 224 | tensor |

### Alignment note

This method is part of the repo-local BaoIAD inventory under the **Few-shot / registration** family. The [implementation provenance and reproducibility record](../../docs/alignment/regad.md) is indexed by the [public method-status manifest](../../docs/alignment/method_status.json). Its manifest validation state is **partially verified**. Referenced raw evidence is not distributed with this repository. The result tables above are historical repository summaries, not a unified public ranking, and should be interpreted according to that manifest state.

<!-- BaoIAD repo-local evidence: end -->
