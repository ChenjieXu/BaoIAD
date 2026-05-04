## RD++

> RD++: Anomaly Detection via Iterative Reverse Distillation

- **Paper**: [RD++: Anomaly Detection via Iterative Reverse Distillation](https://arxiv.org/abs/2307.01348)
- **Category**: Knowledge Distillation
- **Backbone**: WRN-50-2

RD++ extends Reverse Distillation with iterative approximation for anomaly detection. The key innovation is progressively refining the student's inversion through multiple iterations, allowing it to better capture complex normal feature distributions while still failing on anomalies. During training, the student network with iterative refinement is trained to match the teacher's features on normal images. At inference, the accumulated discrepancy across iterations serves as the anomaly score, providing more sensitive detection than single-pass reverse distillation.

### Configs

| Config | Description |
|--------|-------------|
| [`rdpp_wrn50_256_mvtec_strict.py`](rdpp_wrn50_256_mvtec_strict.py) | MVTec AD strict alignment |
| [`rdpp_wrn50_256_visa.py`](rdpp_wrn50_256_visa.py) | VisA |

<!-- BaoIAD repo-local evidence: start -->

### MVTec AD result summary

| #cat | img mean±std | pxl mean±std | AUPRO mean±std | AUPIMO mean±std | iECE mean±std | pECE mean±std |
|---|---|---|---|---|---|---|
| 15 | 0.9931±0.0011 | 0.9817±0.0005 | 0.9329±0.0206 | 0.6206±0.1528 | 0.0144±0.0125 | 0.0069±0.0060 |

### VisA result summary

| img_AUROC | pxl_AUROC | img_F1max | pxl_F1max | img_AP | pxl_AP | AUPRO | cats |
|---|---|---|---|---|---|---|---|
| 0.9608 | 0.9855 | — | — | — | — | — | 12/12 |

### Speed summary

| avg_ms_per_img | fps | img_size | forward_mode |
|---|---|---|---|
| 11.14 | 89.8 | 256 | predict |

### Alignment note

This method is part of the repo-local BaoIAD inventory under the **Knowledge distillation** family. The alignment record is [`docs/alignment/rdpp.md`](../../docs/alignment/rdpp.md); it preserves the detailed strict-alignment evidence, including reference freeze notes, code-path checks, probes, and archived benchmark stop-lines.

<!-- BaoIAD repo-local evidence: end -->
