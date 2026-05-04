## CFlow-AD

> CFlow-AD: Real-Time Unsupervised Anomaly Detection with Localization via Conditional Normalizing Flows

- **Paper**: [CFlow-AD: Real-Time Unsupervised Anomaly Detection with Localization via Conditional Normalizing Flows](https://arxiv.org/abs/2107.12571)
- **Category**: Normalizing Flow
- **Backbone**: WRN-50-2

CFlow-AD detects anomalies by estimating the likelihood of visual features using conditional normalizing flows. Normalizing flows learn an invertible mapping from the feature distribution to a standard normal distribution, where low-likelihood regions correspond to anomalies. During training, the flow model is trained on normal features to maximize their log-likelihood. At inference, the negative log-likelihood of test features under the learned flow serves as the anomaly score, enabling both image-level detection and pixel-level localization.

### Configs

| Config | Description |
|--------|-------------|
| [`cflow_mvtec_strict.py`](cflow_mvtec_strict.py) | MVTec AD strict alignment |
| [`cflow_visa.py`](cflow_visa.py) | VisA |

<!-- BaoIAD repo-local evidence: start -->

### MVTec AD result summary

| #cat | img mean±std | pxl mean±std | AUPRO mean±std | AUPIMO mean±std | iECE mean±std | pECE mean±std |
|---|---|---|---|---|---|---|
| 15 | 0.9705±0.0010 | 0.9828±0.0001 | 0.9337±0.0002 | 0.4952±0.0269 | 0.0000±0.0000 | 0.0000±0.0000 |

### VisA result summary

| img_AUROC | pxl_AUROC | img_F1max | pxl_F1max | img_AP | pxl_AP | AUPRO | cats |
|---|---|---|---|---|---|---|---|
| 0.9136 | 0.9829 | 0.8676 | 0.3853 | 0.8967 | 0.3290 | 0.8720 | 12/12 |

### Speed summary

| avg_ms_per_img | fps | img_size | forward_mode |
|---|---|---|---|
| 25.66 | 39.0 | 256 | predict |

### Alignment note

This method is part of the repo-local BaoIAD inventory under the **Normalizing flow** family. The alignment record is [`docs/alignment/cflow.md`](../../docs/alignment/cflow.md); it preserves the detailed strict-alignment evidence, including reference freeze notes, code-path checks, probes, and archived benchmark stop-lines.

<!-- BaoIAD repo-local evidence: end -->
