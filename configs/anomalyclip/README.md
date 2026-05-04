## AnomalyCLIP

> AnomalyCLIP: Object-Agnostic Prompt Learning for Zero-Shot Anomaly Detection

- **Paper**: [AnomalyCLIP: Object-Agnostic Prompt Learning for Zero-Shot Anomaly Detection](https://arxiv.org/abs/2403.02199)
- **Category**: Vision-Language
- **Backbone**: OpenCLIP-ViT/L-14

AnomalyCLIP learns object-agnostic text prompts that generalize across categories for zero-shot anomaly detection. The core idea is to decouple object-specific semantics from anomaly cues in CLIP's text encoder, learning universal anomaly descriptions that transfer to unseen categories. It uses learnable text prompts with an auxiliary classifier to separate object identity from anomaly information. At inference, the learned prompts are paired with CLIP's visual encoder to compute patch-level anomaly scores without any category-specific training data.

### Configs

| Config | Description |
|--------|-------------|
| [`anomalyclip_vitl14_336_518_mvtec_strict.py`](anomalyclip_vitl14_336_518_mvtec_strict.py) | MVTec AD strict alignment |
| [`anomalyclip_vitl14_336_518_visa.py`](anomalyclip_vitl14_336_518_visa.py) | VisA |

<!-- BaoIAD repo-local evidence: start -->

### MVTec AD result summary

| #cat | img mean±std | pxl mean±std | AUPRO mean±std | AUPIMO mean±std | iECE mean±std | pECE mean±std |
|---|---|---|---|---|---|---|
| 15 | 0.9203±0.0010 | 0.9072±0.0011 | 0.8477±0.0017 | — | — | — |

### VisA result summary

| img_AUROC | pxl_AUROC | img_F1max | pxl_F1max | img_AP | pxl_AP | AUPRO | cats |
|---|---|---|---|---|---|---|---|
| 0.8020 | 0.9531 | — | — | — | — | 0.8499 | agg |

### Speed summary

| avg_ms_per_img | fps | img_size | forward_mode |
|---|---|---|---|
| 111.57 | 9.0 | 518 | predict |

### Alignment note

This method is part of the repo-local BaoIAD inventory under the **Vision-language / foundation** family. The alignment record is [`docs/alignment/anomalyclip.md`](../../docs/alignment/anomalyclip.md); it preserves the detailed strict-alignment evidence, including reference freeze notes, code-path checks, probes, and archived benchmark stop-lines.

<!-- BaoIAD repo-local evidence: end -->
