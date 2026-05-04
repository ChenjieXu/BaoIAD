## SAA+

> SAA+: Segment Any Anomaly+

- **Paper**: Preprint
- **Category**: Other
- **Backbone**: -

SAA+ extends Segment Any Anomaly by combining GroundingDINO and SAM for zero-shot anomaly detection and segmentation. The key idea is leveraging vision-language models to detect anomalies through text prompts (e.g., "defect" or "damage") and then segmenting them using SAM's powerful mask prediction. No training is required — the method uses pre-trained GroundingDINO for anomaly grounding and SAM for mask refinement. At inference, text prompts are used to ground anomalous regions, which are then segmented by SAM for pixel-level anomaly maps.

### Configs

| Config | Description |
|--------|-------------|
| [`saaplus_400_mvtec_strict.py`](saaplus_400_mvtec_strict.py) | MVTec AD strict alignment |
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

This method is part of the repo-local BaoIAD inventory under the **Vision-language / foundation** family. The alignment record is [`docs/alignment/saaplus.md`](../../docs/alignment/saaplus.md); it preserves the detailed strict-alignment evidence, including reference freeze notes, code-path checks, probes, and archived benchmark stop-lines.

<!-- BaoIAD repo-local evidence: end -->
