## SAA+

> SAA+: Segment Any Anomaly+

- **Paper**: [publication](https://arxiv.org/abs/2305.10724)
- **Implementation source**: [upstream repository](https://github.com/caoyunkang/Segment-Any-Anomaly); revision: `ff564ed09bef91d86452f62aa1564e778580513e`
- **Category**: Other
- **Backbone**: -

SAA+ extends Segment Any Anomaly by combining GroundingDINO and SAM for zero-shot anomaly detection and segmentation. The key idea is leveraging vision-language models to detect anomalies through text prompts (e.g., "defect" or "damage") and then segmenting them using SAM's powerful mask prediction. No training is required — the method uses pre-trained GroundingDINO for anomaly grounding and SAM for mask refinement. At inference, text prompts are used to ground anomalous regions, which are then segmented by SAM for pixel-level anomaly maps.

### Configs

| Config | Description |
|--------|-------------|
| [`saaplus_400_mvtec_strict.py`](saaplus_400_mvtec_strict.py) | MVTec AD reference configuration |
| [`saaplus_400_visa.py`](saaplus_400_visa.py) | VisA |

<!-- BaoIAD repo-local evidence: start -->

### MVTec AD result summary

| #cat | img mean±std | pxl mean±std | AUPRO mean±std | AUPIMO mean±std | iECE mean±std | pECE mean±std |
|---|---|---|---|---|---|---|
| 15 | 0.6647±0.0000 | 0.7493±0.0000 | 0.5332±0.0000 | 0.0733±0.0000 | 0.2316±0.0000 | 0.0815±0.0000 |

### VisA result summary

| img_AUROC | pxl_AUROC | img_F1max | pxl_F1max | img_AP | pxl_AP | AUPRO | cats |
|---|---|---|---|---|---|---|---|
| 0.5857 | 0.7710 | 0.7485 | 0.1970 | 0.6573 | 0.1481 | 0.5463 | 12/12 |

### Speed summary

| avg_ms_per_img | fps | img_size | forward_mode |
|---|---|---|---|
| 387.37 | 2.6 | 256 | tensor |

### Alignment note

This method is part of the repo-local BaoIAD inventory under the **Vision-language / foundation** family. The [implementation provenance and reproducibility record](../../docs/alignment/saaplus.md) is indexed by the [public method-status manifest](../../docs/alignment/method_status.json). Its manifest validation state is **historical evidence**. Referenced raw evidence is not distributed with this repository. The result tables above are historical repository summaries, not a unified public ranking, and should be interpreted according to that manifest state.

<!-- BaoIAD repo-local evidence: end -->
