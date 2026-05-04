## AdaCLIP

> AdaCLIP: Adaptive Vision-Language Model for Anomaly Detection

- **Paper**: [AdaCLIP: Adaptive Vision-Language Model for Anomaly Detection](https://arxiv.org/abs/2409.18928)
- **Category**: Vision-Language
- **Backbone**: OpenCLIP-ViT/L-14

AdaCLIP adaptively adjusts CLIP's visual and textual representations using learnable adapters for anomaly detection. It introduces visual and textual adapter modules that modulate CLIP features based on input content, enabling both zero-shot and few-shot detection. The adapters are trained to enhance the discriminability of CLIP features for anomaly vs. normal patterns, with separate adaptation pathways for image-level classification and pixel-level segmentation. At inference, the adapted CLIP computes anomaly scores by comparing adapted visual features against normal and abnormal text prompts.

### Configs

| Config | Description |
|--------|-------------|
| [`adaclip_vitl14_336_518_mvtec_strict.py`](adaclip_vitl14_336_518_mvtec_strict.py) | MVTec AD strict alignment |
| [`adaclip_vitl14_336_518_visa.py`](adaclip_vitl14_336_518_visa.py) | VisA |

<!-- BaoIAD repo-local evidence: start -->

### MVTec AD result summary

| #cat | img mean±std | pxl mean±std | AUPRO mean±std | AUPIMO mean±std | iECE mean±std | pECE mean±std |
|---|---|---|---|---|---|---|
| 15 | 0.8455±0.0087 | 0.8587±0.0229 | 0.7812±0.0235 | — | — | — |

### VisA result summary

| img_AUROC | pxl_AUROC | img_F1max | pxl_F1max | img_AP | pxl_AP | AUPRO | cats |
|---|---|---|---|---|---|---|---|
| 0.8385 | 0.9528 | 0.8123 | 0.3481 | 0.8707 | 0.2974 | 0.8854 | agg |

### Speed summary

| avg_ms_per_img | fps | img_size | forward_mode |
|---|---|---|---|
| 141.74 | 7.1 | 518 | predict |

### Alignment note

This method is part of the repo-local BaoIAD inventory under the **Vision-language / foundation** family. The alignment record is [`docs/alignment/adaclip.md`](../../docs/alignment/adaclip.md); it preserves the detailed strict-alignment evidence, including reference freeze notes, code-path checks, probes, and archived benchmark stop-lines.

<!-- BaoIAD repo-local evidence: end -->
