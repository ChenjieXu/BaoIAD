## U-Flow

> U-Flow: A U-shaped Normalizing Flow for Anomaly Detection

- **Paper**: [publication](https://arxiv.org/abs/2211.12353)
- **Implementation source**: [upstream repository](https://github.com/mtailanian/uflow); revision: `d6217844836790773f2c4b91ff3046c59b23f027`
- **Category**: Normalizing Flow
- **Backbone**: WRN-50-2

U-Flow uses a U-shaped normalizing flow architecture that models features at multiple scales with skip connections for anomaly detection. The U-shape with skip connections allows the flow to capture both fine-grained and coarse-grained feature distributions, improving localization accuracy over single-scale flows. During training, the U-shaped flow learns to map multi-scale normal features to a standard normal distribution. At inference, the negative log-likelihood at each position provides the anomaly map, with skip connections enabling precise localization.

### Configs

| Config | Description |
|--------|-------------|
| [`uflow_mcait_448_mvtec_strict.py`](uflow_mcait_448_mvtec_strict.py) | MVTec AD reference configuration |
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

This method is part of the repo-local BaoIAD inventory under the **Normalizing flow** family. The [implementation provenance and reproducibility record](../../docs/alignment/uflow.md) is indexed by the [public method-status manifest](../../docs/alignment/method_status.json). Its manifest validation state is **historical evidence**. Referenced raw evidence is not distributed with this repository. The result tables above are historical repository summaries, not a unified public ranking, and should be interpreted according to that manifest state.

<!-- BaoIAD repo-local evidence: end -->
