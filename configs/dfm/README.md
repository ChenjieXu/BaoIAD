## DFM

> Deep Feature Modeling for Anomaly Detection

- **Paper**: [publication](https://arxiv.org/abs/1909.11786)
- **Implementation source**: [upstream repository](https://github.com/open-edge-platform/anomalib); revision: `4f6af1acb0ee7b81f54cda036dd9f1c27f63b69a`
- **Category**: Memory Bank
- **Backbone**: WRN-50-2

DFM fits a linear subspace to deep features of normal samples using PCA, then detects anomalies via the reconstruction error from projecting onto the subspace. The key idea is that normal features lie near a low-dimensional subspace, while anomalous features have significant components outside it. No training is required beyond computing the PCA projection on normal features. At inference, the reconstruction error (distance from the subspace) serves as the anomaly score, with larger errors indicating anomalies.

### Configs

| Config | Description |
|--------|-------------|
| [`dfm_256_mvtec_strict.py`](dfm_256_mvtec_strict.py) | MVTec AD reference configuration |
| [`dfm_256_visa.py`](dfm_256_visa.py) | VisA |

<!-- BaoIAD repo-local evidence: start -->

### MVTec AD result summary

| #cat | img mean±std | pxl mean±std | AUPRO mean±std | AUPIMO mean±std | iECE mean±std | pECE mean±std |
|---|---|---|---|---|---|---|
| 15 | 0.9362±0.0000 | 0.9387±0.0000 | 0.6825±0.0000 | 0.3044±0.0000 | 0.0000±0.0000 | 0.0156±0.0000 |

### VisA result summary

| img_AUROC | pxl_AUROC | img_F1max | pxl_F1max | img_AP | pxl_AP | AUPRO | cats |
|---|---|---|---|---|---|---|---|
| 0.8917 | 0.9215 | 0.8659 | 0.2387 | 0.8985 | 0.1964 | 0.6092 | 12/12 |

### Speed summary

| avg_ms_per_img | fps | img_size | forward_mode |
|---|---|---|---|
| 6.47 | 154.5 | 256 | tensor |

### Alignment note

This method is part of the repo-local BaoIAD inventory under the **Feature-memory / density** family. The [implementation provenance and reproducibility record](../../docs/alignment/dfm.md) is indexed by the [public method-status manifest](../../docs/alignment/method_status.json). Its manifest validation state is **partially verified**. Referenced raw evidence is not distributed with this repository. The result tables above are historical repository summaries, not a unified public ranking, and should be interpreted according to that manifest state.

<!-- BaoIAD repo-local evidence: end -->
