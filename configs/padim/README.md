## PaDiM

> PaDiM: A Patch Distribution Modeling Framework for Anomaly Detection

- **Paper**: [publication](https://arxiv.org/abs/2011.08785)
- **Implementation source**: [upstream repository](https://github.com/open-edge-platform/anomalib); revision: `0ef8ab1e43340bddf4d92d1f046c3d34a83af6b0`
- **Category**: Memory Bank
- **Backbone**: WRN-50-2

PaDiM models the distribution of patch-level features using multivariate Gaussian distributions per spatial position for anomaly detection. The key idea is that normal patches at each position follow a Gaussian distribution, and anomalies deviate from their position-specific distribution. No training is required beyond extracting features and computing per-position mean and covariance from normal images. At inference, the Mahalanobis distance from each test patch to its position-specific Gaussian serves as the anomaly score.

### Configs

| Config | Description |
|--------|-------------|
| [`padim_wrn50_256_mvtec_strict.py`](padim_wrn50_256_mvtec_strict.py) | MVTec AD reference configuration |
| [`padim_wrn50_256_visa.py`](padim_wrn50_256_visa.py) | VisA |

<!-- BaoIAD repo-local evidence: start -->

### MVTec AD result summary

| #cat | img mean±std | pxl mean±std | AUPRO mean±std | AUPIMO mean±std | iECE mean±std | pECE mean±std |
|---|---|---|---|---|---|---|
| 15 | 0.9589±0.0000 | 0.9773±0.0000 | 0.9323±0.0000 | 0.4651±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 |

### VisA result summary

| img_AUROC | pxl_AUROC | img_F1max | pxl_F1max | img_AP | pxl_AP | AUPRO | cats |
|---|---|---|---|---|---|---|---|
| 0.8812 | 0.9837 | 0.8675 | 0.3515 | 0.8893 | 0.2951 | 0.8712 | 12/12 |

### Speed summary

| avg_ms_per_img | fps | img_size | forward_mode |
|---|---|---|---|
| 6.49 | 154.2 | 256 | tensor |

### Alignment note

This method is part of the repo-local BaoIAD inventory under the **Feature-memory / density** family. The [implementation provenance and reproducibility record](../../docs/alignment/padim.md) is indexed by the [public method-status manifest](../../docs/alignment/method_status.json). Its manifest validation state is **historical evidence**. Referenced raw evidence is not distributed with this repository. The result tables above are historical repository summaries, not a unified public ranking, and should be interpreted according to that manifest state.

<!-- BaoIAD repo-local evidence: end -->
