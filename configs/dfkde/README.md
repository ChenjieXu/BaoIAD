## DFKDE

> Deep Feature Kernel Density Estimation for Anomaly Detection

- **Paper**: Not recorded in the method-status manifest.
- **Implementation source**: [upstream repository](https://github.com/open-edge-platform/anomalib); revision: `4f6af1acb0ee7b81f54cda036dd9f1c27f63b69a`
- **Category**: Memory Bank
- **Backbone**: WRN-50-2

DFKDE applies kernel density estimation on deep features extracted from a pre-trained backbone to model the normal distribution. The method stores a subset of normal features and uses a Gaussian kernel to estimate the density at test points. No training is required beyond feature extraction and selecting the reference subset. At inference, the estimated density at each test feature serves as the anomaly score — low density indicates anomalies, with both image-level and pixel-level scores available.

### Configs

| Config | Description |
|--------|-------------|
| [`dfkde_256_mvtec_strict.py`](dfkde_256_mvtec_strict.py) | MVTec AD reference configuration |
| [`dfkde_256_visa.py`](dfkde_256_visa.py) | VisA |

<!-- BaoIAD repo-local evidence: start -->

### MVTec AD result summary

| #cat | img mean±std | pxl mean±std | AUPRO mean±std | AUPIMO mean±std | iECE mean±std | pECE mean±std |
|---|---|---|---|---|---|---|
| 15 | 0.7558±0.0000 | 0.7164±0.0000 | 0.2106±0.0000 | 0.0000±0.0000 | 0.1986±0.0000 | 0.5385±0.0000 |

### VisA result summary

| img_AUROC | pxl_AUROC | img_F1max | pxl_F1max | img_AP | pxl_AP | AUPRO | cats |
|---|---|---|---|---|---|---|---|
| 0.7212 | 0.7302 | 0.7577 | 0.0980 | 0.7755 | 0.0492 | 0.2277 | 12/12 |

### Speed summary

| avg_ms_per_img | fps | img_size | forward_mode |
|---|---|---|---|
| 3.51 | 284.7 | 256 | predict |

### Alignment note

This method is part of the repo-local BaoIAD inventory under the **Feature-memory / density** family. The [implementation provenance and reproducibility record](../../docs/alignment/dfkde.md) is indexed by the [public method-status manifest](../../docs/alignment/method_status.json). Its manifest validation state is **historical evidence**. Referenced raw evidence is not distributed with this repository. The result tables above are historical repository summaries, not a unified public ranking, and should be interpreted according to that manifest state.

<!-- BaoIAD repo-local evidence: end -->
