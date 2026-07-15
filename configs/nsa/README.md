## NSA

> NSA: Natural Synthetic Anomalies for Self-Supervised Anomaly Detection and Segmentation

- **Paper**: [publication](https://arxiv.org/abs/2109.15222)
- **Implementation source**: [upstream repository](https://github.com/hmsch/natural-synthetic-anomalies); revision: `919591685307ce030fe27cb77687509dc277189c`
- **Category**: Other
- **Backbone**: WRN-50-2

NSA generates natural synthetic anomalies by pasting patches from other images onto training samples for self-supervised anomaly detection. The key innovation is creating more realistic synthetic anomalies compared to random noise, as the pasted patches contain real image content that blends naturally. During training, a segmentation model learns to detect these synthetic perturbations on augmented training images. At inference, the segmentation model outputs pixel-level anomaly predictions for real anomalies.

### Configs

| Config | Description |
|--------|-------------|
| [`nsa_rn18_256_mvtec_strict.py`](nsa_rn18_256_mvtec_strict.py) | MVTec AD reference configuration |
| [`nsa_rn18_256_visa.py`](nsa_rn18_256_visa.py) | VisA |

<!-- BaoIAD repo-local evidence: start -->

### MVTec AD result summary

| #cat | img mean±std | pxl mean±std | AUPRO mean±std | AUPIMO mean±std | iECE mean±std | pECE mean±std |
|---|---|---|---|---|---|---|
| 15 | 0.8490±0.0149 | 0.9311±0.0047 | 0.7956±0.0101 | 0.2769±0.0273 | 0.6549±0.0132 | 0.0501±0.0104 |

### VisA result summary

| img_AUROC | pxl_AUROC | img_F1max | pxl_F1max | img_AP | pxl_AP | AUPRO | cats |
|---|---|---|---|---|---|---|---|
| 0.6349 | 0.8196 | 0.7595 | 0.1179 | 0.7158 | 0.0847 | 0.4687 | 12/12 |

### Speed summary

| avg_ms_per_img | fps | img_size | forward_mode |
|---|---|---|---|
| 3.54 | 282.7 | 256 | predict |

### Alignment note

This method is part of the repo-local BaoIAD inventory under the **Self-supervised synthesis** family. The [implementation provenance and reproducibility record](../../docs/alignment/nsa.md) is indexed by the [public method-status manifest](../../docs/alignment/method_status.json). Its manifest validation state is **historical evidence**. Referenced raw evidence is not distributed with this repository. The result tables above are historical repository summaries, not a unified public ranking, and should be interpreted according to that manifest state.

<!-- BaoIAD repo-local evidence: end -->
