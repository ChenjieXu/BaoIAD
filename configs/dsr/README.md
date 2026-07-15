## DSR

> DSR: A Dual Subspace Re-projection Network for Surface Anomaly Detection

- **Paper**: [publication](https://arxiv.org/abs/2208.01521)
- **Implementation source**: [upstream repository](https://github.com/open-edge-platform/anomalib); revision: `4f6af1acb0ee7b81f54cda036dd9f1c27f63b69a`
- **Category**: Reconstruction
- **Backbone**: AE + Subspace

DSR uses a dual subspace re-projection network that projects features onto normal and anomaly subspaces. The method learns two complementary subspaces — one capturing normal patterns and one capturing anomaly patterns — and detects anomalies via the re-projection residual. During training, the dual subspace is learned from normal features using a reconstruction objective. At inference, the residual from re-projecting test features onto the normal subspace serves as the anomaly score.

### Configs

| Config | Description |
|--------|-------------|
| [`dsr_256_mvtec_strict.py`](dsr_256_mvtec_strict.py) | MVTec AD reference configuration |
| [`dsr_256_visa.py`](dsr_256_visa.py) | VisA |

<!-- BaoIAD repo-local evidence: start -->

### MVTec AD result summary

| #cat | img mean±std | pxl mean±std | AUPRO mean±std | AUPIMO mean±std | iECE mean±std | pECE mean±std |
|---|---|---|---|---|---|---|
| 15 | 0.9186±0.0120 | 0.8261±0.0073 | 0.7322±0.0149 | 0.1939±0.0125 | 0.3910±0.0234 | 0.1073±0.0030 |

### VisA result summary

| img_AUROC | pxl_AUROC | img_F1max | pxl_F1max | img_AP | pxl_AP | AUPRO | cats |
|---|---|---|---|---|---|---|---|
| 0.9256 | 0.7696 | 0.8873 | 0.3209 | 0.9451 | 0.2433 | 0.5706 | 12/12 |

### Speed summary

| avg_ms_per_img | fps | img_size | forward_mode |
|---|---|---|---|
| 13.3 | 75.2 | 256 | predict |

### Alignment note

This method is part of the repo-local BaoIAD inventory under the **Self-supervised synthesis** family. The [implementation provenance and reproducibility record](../../docs/alignment/dsr.md) is indexed by the [public method-status manifest](../../docs/alignment/method_status.json). Its manifest validation state is **historical evidence**. Referenced raw evidence is not distributed with this repository. The result tables above are historical repository summaries, not a unified public ranking, and should be interpreted according to that manifest state.

<!-- BaoIAD repo-local evidence: end -->
