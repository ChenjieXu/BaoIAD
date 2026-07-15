## AnoVL

> AnoVL: Self-Supervised Anomaly Detection by Contrastive Learning

- **Paper**: [publication](https://arxiv.org/abs/2308.15939)
- **Implementation source**: [upstream repository](https://github.com/hq-deng/AnoVL); revision: `3a70bfdaea6baf1eeb140c5de8155b535bd94833`
- **Category**: Vision-Language
- **Backbone**: OpenCLIP-ViT/L-14

AnoVL performs self-supervised anomaly detection by contrastive learning of visual-linguistic representations. It aligns CLIP's visual and textual features for normal patterns through contrastive learning, while anomalies naturally deviate from the aligned normal manifold. The method trains projection heads that map visual and textual features into a shared space where normal features cluster tightly. At inference, anomaly scores are computed from the distance between visual features and the nearest normal textual prototype in the shared space.

### Configs

| Config | Description |
|--------|-------------|
| [`anovl_vitb16plus_240_mvtec_strict.py`](anovl_vitb16plus_240_mvtec_strict.py) | MVTec AD reference configuration |
| [`anovl_vitb16plus_240_visa.py`](anovl_vitb16plus_240_visa.py) | VisA |

<!-- BaoIAD repo-local evidence: start -->

### MVTec AD result summary

| #cat | img mean±std | pxl mean±std | AUPRO mean±std | AUPIMO mean±std | iECE mean±std | pECE mean±std |
|---|---|---|---|---|---|---|
| 15 | 0.9216±0.0000 | 0.8985±0.0012 | 0.7757±0.0007 | 0.3233±0.0173 | 0.2372±0.0000 | 0.4500±0.0000 |

### VisA result summary

| img_AUROC | pxl_AUROC | img_F1max | pxl_F1max | img_AP | pxl_AP | AUPRO | cats |
|---|---|---|---|---|---|---|---|
| 0.7884 | 0.8775 | 0.7975 | 0.1482 | 0.8155 | 0.1004 | 0.6693 | agg |

### Speed summary

| avg_ms_per_img | fps | img_size | forward_mode |
|---|---|---|---|
| 117.48 | 8.5 | 240 | predict |

### Alignment note

This method is part of the repo-local BaoIAD inventory under the **Vision-language / foundation** family. The [implementation provenance and reproducibility record](../../docs/alignment/anovl.md) is indexed by the [public method-status manifest](../../docs/alignment/method_status.json). Its manifest validation state is **historical evidence**. Referenced raw evidence is not distributed with this repository. The result tables above are historical repository summaries, not a unified public ranking, and should be interpreted according to that manifest state.

<!-- BaoIAD repo-local evidence: end -->
