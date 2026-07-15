## RD++

> RD++: Anomaly Detection via Iterative Reverse Distillation

- **Paper**: [publication](https://openaccess.thecvf.com/content/CVPR2023/html/Tien_Revisiting_Reverse_Distillation_for_Anomaly_Detection_CVPR_2023_paper.html)
- **Implementation source**: [upstream repository](https://github.com/tientrandinh/Revisiting-Reverse-Distillation); revision: `7f2ceb7c87e602617b8600e1a498f7ef7f5247d6`
- **Category**: Knowledge Distillation
- **Backbone**: WRN-50-2

RD++ extends Reverse Distillation with iterative approximation for anomaly detection. The key innovation is progressively refining the student's inversion through multiple iterations, allowing it to better capture complex normal feature distributions while still failing on anomalies. During training, the student network with iterative refinement is trained to match the teacher's features on normal images. At inference, the accumulated discrepancy across iterations serves as the anomaly score, providing more sensitive detection than single-pass reverse distillation.

### Configs

| Config | Description |
|--------|-------------|
| [`rdpp_wrn50_256_mvtec_strict.py`](rdpp_wrn50_256_mvtec_strict.py) | MVTec AD reference configuration |
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

This method is part of the repo-local BaoIAD inventory under the **Knowledge distillation** family. The [implementation provenance and reproducibility record](../../docs/alignment/rdpp.md) is indexed by the [public method-status manifest](../../docs/alignment/method_status.json). Its manifest validation state is **historical evidence**. Referenced raw evidence is not distributed with this repository. The result tables above are historical repository summaries, not a unified public ranking, and should be interpreted according to that manifest state.

<!-- BaoIAD repo-local evidence: end -->
