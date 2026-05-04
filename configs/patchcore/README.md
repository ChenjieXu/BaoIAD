## PatchCore

> PatchCore: Towards Total Recall in Industrial Anomaly Detection

- **Paper**: [PatchCore: Towards Total Recall in Industrial Anomaly Detection](https://arxiv.org/abs/2106.08265)
- **Category**: Memory Bank
- **Backbone**: WRN-50-2

PatchCore builds a memory bank of core patch features from normal training data using greedy coreset subsampling for anomaly detection. The key innovation is the coreset subsampling that selects a small representative subset of patch features, reducing memory and computation while preserving detection accuracy. No model training is required — only feature extraction and coreset selection on normal images. At inference, anomaly scores are computed as the distance from each test patch feature to its nearest neighbor in the coreset memory bank, with both image-level and pixel-level scores.

### Configs

| Config | Description |
|--------|-------------|
| [`patchcore_wrn50_256_mvtec_strict.py`](patchcore_wrn50_256_mvtec_strict.py) | MVTec AD strict alignment |
| [`patchcore_wrn50_256_visa.py`](patchcore_wrn50_256_visa.py) | VisA |

<!-- BaoIAD repo-local evidence: start -->

### MVTec AD result summary

| #cat | img mean±std | pxl mean±std | AUPRO mean±std | AUPIMO mean±std | iECE mean±std | pECE mean±std |
|---|---|---|---|---|---|---|
| 15 | 0.9794±0.0000 | 0.9790±0.0000 | 0.9152±0.0000 | 0.6505±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 |

### VisA result summary

| img_AUROC | pxl_AUROC | img_F1max | pxl_F1max | img_AP | pxl_AP | AUPRO | cats |
|---|---|---|---|---|---|---|---|
| 0.8896 | 0.9770 | 0.8700 | 0.4075 | 0.9071 | 0.3834 | 0.8492 | 12/12 |

### Speed summary

| avg_ms_per_img | fps | img_size | forward_mode |
|---|---|---|---|
| 6.58 | 152.0 | 256 | predict |

### Alignment note

This method is part of the repo-local BaoIAD inventory under the **Feature-memory / density** family. The alignment record is [`docs/alignment/patchcore.md`](../../docs/alignment/patchcore.md); it preserves the detailed strict-alignment evidence, including reference freeze notes, code-path checks, probes, and archived benchmark stop-lines.

<!-- BaoIAD repo-local evidence: end -->
