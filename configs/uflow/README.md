## U-Flow

> U-Flow: A U-shaped Normalizing Flow for Anomaly Detection

- **Paper**: [U-Flow: A U-shaped Normalizing Flow for Anomaly Detection](https://arxiv.org/abs/2207.09506)
- **Category**: Normalizing Flow
- **Backbone**: WRN-50-2

U-Flow uses a U-shaped normalizing flow architecture that models features at multiple scales with skip connections for anomaly detection. The U-shape with skip connections allows the flow to capture both fine-grained and coarse-grained feature distributions, improving localization accuracy over single-scale flows. During training, the U-shaped flow learns to map multi-scale normal features to a standard normal distribution. At inference, the negative log-likelihood at each position provides the anomaly map, with skip connections enabling precise localization.

### Configs

| Config | Description |
|--------|-------------|
| [`uflow_mcait_448_mvtec_strict.py`](uflow_mcait_448_mvtec_strict.py) | MVTec AD strict alignment |
| [`uflow_mcait_448_visa.py`](uflow_mcait_448_visa.py) | VisA |

<!-- BaoIAD repo-local evidence: start -->

### MVTec AD result summary

| #cat | img mean±std | pxl mean±std | AUPRO mean±std | AUPIMO mean±std | iECE mean±std | pECE mean±std |
|---|---|---|---|---|---|---|
| 15 | 0.9819±0.0012 | 0.9854±0.0005 | 0.9439±0.0012 | 0.5456±0.0110 | 0.2446±0.0054 | 0.4074±0.0059 |

### VisA result summary

| img_AUROC | pxl_AUROC | img_F1max | pxl_F1max | img_AP | pxl_AP | AUPRO | cats |
|---|---|---|---|---|---|---|---|
| 0.9826 | 0.9936 | 0.9561 | 0.6383 | 0.9914 | 0.6007 | 0.9179 | 12/12 |

### Speed summary

| avg_ms_per_img | fps | img_size | forward_mode |
|---|---|---|---|
| 121.87 | 8.2 | 448 | predict |

### Alignment note

This method is part of the repo-local BaoIAD inventory under the **Normalizing flow** family. The alignment record is [`docs/alignment/uflow.md`](../../docs/alignment/uflow.md); it preserves the detailed strict-alignment evidence, including reference freeze notes, code-path checks, probes, and archived benchmark stop-lines.

<!-- BaoIAD repo-local evidence: end -->
