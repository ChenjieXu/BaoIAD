## GLASS

> GLASS: GLobal Attention-based Anomaly Detection

- **Paper**: [publication](https://arxiv.org/abs/2407.09359)
- **Implementation source**: [upstream repository](https://github.com/cqylunlun/GLASS); revision: `6af03b9d7f7b33a1aebd69cd4c30a41bf020a2d1`
- **Category**: Other
- **Backbone**: WRN-50-2

GLASS generates anomalies via global attention-based Perlin noise masks and DTD textures, then trains a discriminator to detect both synthetic and real anomalies. The key innovation is using global attention to generate more realistic and diverse synthetic anomalies compared to random Perlin masks. During training, synthetic anomalies are blended with normal images using attention-guided masks, and a segmentation network learns to detect these patterns. At inference, the segmentation network outputs pixel-level anomaly predictions.

### Configs

| Config | Description |
|--------|-------------|
| [`glass_wrn50_288_mvtec_strict.py`](glass_wrn50_288_mvtec_strict.py) | MVTec AD reference configuration |
| [`glass_wrn50_288_visa.py`](glass_wrn50_288_visa.py) | VisA |

<!-- BaoIAD repo-local evidence: start -->

### MVTec AD result summary

| #cat | img mean±std | pxl mean±std | AUPRO mean±std | AUPIMO mean±std | iECE mean±std | pECE mean±std |
|---|---|---|---|---|---|---|
| 15 | 0.9982±0.0004 | 0.9885±0.0055 | 0.9674±0.0002 | 0.7632±0.0018 | 0.1882±0.0397 | 0.1221±0.0674 |

### VisA result summary

| img_AUROC | pxl_AUROC | img_F1max | pxl_F1max | img_AP | pxl_AP | AUPRO | cats |
|---|---|---|---|---|---|---|---|
| 0.6632 | 0.8478 | 0.7618 | 0.1166 | 0.7274 | 0.0675 | 0.4764 | 12/12 |

### Speed summary

| avg_ms_per_img | fps | img_size | forward_mode |
|---|---|---|---|
| 14.42 | 69.4 | 288 | predict |

### Alignment note

This method is part of the repo-local BaoIAD inventory under the **Self-supervised synthesis** family. The [implementation provenance and reproducibility record](../../docs/alignment/glass.md) is indexed by the [public method-status manifest](../../docs/alignment/method_status.json). Its manifest validation state is **partially verified**. Referenced raw evidence is not distributed with this repository. The result tables above are historical repository summaries, not a unified public ranking, and should be interpreted according to that manifest state.

<!-- BaoIAD repo-local evidence: end -->
