## RD

> RD: Anomaly Detection via Reverse Distillation

- **Paper**: [RD: Anomaly Detection via Reverse Distillation](https://arxiv.org/abs/2201.10703)
- **Category**: Knowledge Distillation
- **Backbone**: WRN-50-2

RD detects anomalies via reverse distillation — a student network learns to invert the teacher's feature mapping. The key idea is that the student can only learn to invert normal patterns; anomalous features produce large inversion errors because they were never seen during training. During training, the student network is trained to reconstruct the teacher's multi-scale features from normal images using a reverse bottleneck. At inference, the cosine distance between student and teacher features serves as the anomaly score, with both image-level and pixel-level localization.

### Configs

| Config | Description |
|--------|-------------|
| [`rd_wrn50_256_mvtec_strict.py`](rd_wrn50_256_mvtec_strict.py) | MVTec AD strict alignment |
| [`rd_wrn50_256_visa.py`](rd_wrn50_256_visa.py) | VisA |

<!-- BaoIAD repo-local evidence: start -->

### MVTec AD result summary

| #cat | img mean±std | pxl mean±std | AUPRO mean±std | AUPIMO mean±std | iECE mean±std | pECE mean±std |
|---|---|---|---|---|---|---|
| 15 | 0.9865±0.0007 | 0.9783±0.0003 | 0.9327±0.0011 | 0.6685±0.0149 | 0.0476±0.0118 | 0.0225±0.0076 |

### VisA result summary

| img_AUROC | pxl_AUROC | img_F1max | pxl_F1max | img_AP | pxl_AP | AUPRO | cats |
|---|---|---|---|---|---|---|---|
| 0.9564 | 0.9839 | 0.9239 | 0.4460 | 0.9639 | 0.4104 | 0.8969 | 12/12 |

### Speed summary

| avg_ms_per_img | fps | img_size | forward_mode |
|---|---|---|---|
| 10.54 | 94.9 | 256 | predict |

### Alignment note

This method is part of the repo-local BaoIAD inventory under the **Knowledge distillation** family. The alignment record is [`docs/alignment/rd.md`](../../docs/alignment/rd.md); it preserves the detailed strict-alignment evidence, including reference freeze notes, code-path checks, probes, and archived benchmark stop-lines.

<!-- BaoIAD repo-local evidence: end -->
