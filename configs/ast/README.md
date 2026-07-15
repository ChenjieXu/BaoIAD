## AST

> AST: Asymmetric Student-Teacher for Anomaly Detection

- **Paper**: [publication](https://arxiv.org/abs/2210.07829)
- **Implementation source**: [upstream repository](https://github.com/marco-rudolph/AST); revision: `8c243ad9adac68e874f87edc6618aa5ea2827228`
- **Category**: Knowledge Distillation
- **Backbone**: WRN-50-2

AST uses an asymmetric student-teacher architecture for anomaly detection. The key idea is that the student network has a smaller capacity than the teacher, so it can only learn to reconstruct normal patterns — anomalous regions produce large reconstruction errors due to the capacity bottleneck. During training, the student learns to match the teacher's output on normal images only. At inference, the pixel-wise difference between student and teacher outputs serves as the anomaly map, with image-level scores derived from the maximum discrepancy.

### Configs

| Config | Description |
|--------|-------------|
| [`ast_effnet_b5_768_mvtec_strict.py`](ast_effnet_b5_768_mvtec_strict.py) | MVTec AD reference configuration |
| [`ast_effnet_b5_768_visa.py`](ast_effnet_b5_768_visa.py) | VisA |

<!-- BaoIAD repo-local evidence: start -->

### MVTec AD result summary

| #cat | img mean±std | pxl mean±std | AUPRO mean±std | AUPIMO mean±std | iECE mean±std | pECE mean±std |
|---|---|---|---|---|---|---|
| 15 | 0.9867±0.0005 | 0.9548±0.0002 | 0.8625±0.0028 | — | — | — |

### VisA result summary

| img_AUROC | pxl_AUROC | img_F1max | pxl_F1max | img_AP | pxl_AP | AUPRO | cats |
|---|---|---|---|---|---|---|---|
| 0.9258 | 0.9177 | 0.8877 | 0.3284 | 0.9436 | 0.2479 | 0.7256 | 12/12 |

### Speed summary

| avg_ms_per_img | fps | img_size | forward_mode |
|---|---|---|---|
| 17.84 | 56.1 | 768 | predict |

### Alignment note

This method is part of the repo-local BaoIAD inventory under the **Knowledge distillation** family. The [implementation provenance and reproducibility record](../../docs/alignment/ast.md) is indexed by the [public method-status manifest](../../docs/alignment/method_status.json). Its manifest validation state is **partially verified**. Referenced raw evidence is not distributed with this repository. The result tables above are historical repository summaries, not a unified public ranking, and should be interpreted according to that manifest state.

<!-- BaoIAD repo-local evidence: end -->
