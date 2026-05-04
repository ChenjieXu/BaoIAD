## FastFlow

> FastFlow: Unsupervised Anomaly Detection and Localization via Normalizing Flow

- **Paper**: [FastFlow: Unsupervised Anomaly Detection and Localization via Normalizing Flow](https://arxiv.org/abs/2111.07677)
- **Category**: Normalizing Flow
- **Backbone**: WRN-50-2

FastFlow uses fully convolutional normalizing flows to estimate the likelihood of visual features for anomaly detection. The fully convolutional architecture enables efficient per-position likelihood computation, making both training and inference fast. During training, the flow model learns to map normal features to a standard normal distribution by maximizing log-likelihood. At inference, the negative log-likelihood at each spatial position provides the anomaly map, with image-level scores from the average likelihood.

### Configs

| Config | Description |
|--------|-------------|
| [`fastflow_wrn50_256_mvtec_strict.py`](fastflow_wrn50_256_mvtec_strict.py) | MVTec AD strict alignment |
| [`fastflow_wrn50_256_visa.py`](fastflow_wrn50_256_visa.py) | VisA |

<!-- BaoIAD repo-local evidence: start -->

### MVTec AD result summary

| #cat | img mean±std | pxl mean±std | AUPRO mean±std | AUPIMO mean±std | iECE mean±std | pECE mean±std |
|---|---|---|---|---|---|---|
| 15 | 0.9328±0.0042 | 0.9688±0.0025 | 0.8943±0.0041 | 0.3233±0.0168 | 0.0000±0.0000 | 0.0000±0.0000 |

### VisA result summary

| img_AUROC | pxl_AUROC | img_F1max | pxl_F1max | img_AP | pxl_AP | AUPRO | cats |
|---|---|---|---|---|---|---|---|
| 0.9259 | 0.9763 | 0.8910 | 0.3608 | 0.9375 | 0.2848 | 0.8919 | 12/12 |

### Speed summary

| avg_ms_per_img | fps | img_size | forward_mode |
|---|---|---|---|
| 14.81 | 67.5 | 256 | predict |

### Alignment note

This method is part of the repo-local BaoIAD inventory under the **Normalizing flow** family. The alignment record is [`docs/alignment/fastflow.md`](../../docs/alignment/fastflow.md); it preserves the detailed strict-alignment evidence, including reference freeze notes, code-path checks, probes, and archived benchmark stop-lines.

<!-- BaoIAD repo-local evidence: end -->
