## SimpleNet

> SimpleNet: A Simple Network for Image Anomaly Detection and Localization

- **Paper**: [SimpleNet: A Simple Network for Image Anomaly Detection and Localization](https://arxiv.org/abs/2303.15140)
- **Category**: Discriminator
- **Backbone**: WRN-50-2

SimpleNet adds a simple feature adaptor and discriminator on top of a frozen pre-trained backbone for anomaly detection. The key innovation is the feature adaptor that adds Gaussian noise to features before discrimination, creating a simple but effective one-class classification boundary. During training, the adaptor and discriminator are trained on normal features with added noise as the negative class. At inference, the discriminator score on adapted features serves as the anomaly score, with both image-level and pixel-level outputs.

### Configs

| Config | Description |
|--------|-------------|
| [`simplenet_wrn50_288_mvtec_strict.py`](simplenet_wrn50_288_mvtec_strict.py) | MVTec AD strict alignment |
| [`simplenet_wrn50_288_visa.py`](simplenet_wrn50_288_visa.py) | VisA |

<!-- BaoIAD repo-local evidence: start -->

### MVTec AD result summary

| #cat | img mean±std | pxl mean±std | AUPRO mean±std | AUPIMO mean±std | iECE mean±std | pECE mean±std |
|---|---|---|---|---|---|---|
| 15 | 0.9950±0.0010 | 0.9758±0.0012 | 0.9073±0.0022 | 0.7374±0.0230 | 0.3879±0.0072 | 0.0000±0.0000 |

### VisA result summary

| img_AUROC | pxl_AUROC | img_F1max | pxl_F1max | img_AP | pxl_AP | AUPRO | cats |
|---|---|---|---|---|---|---|---|
| 0.9704 | 0.9795 | 0.9408 | 0.4200 | 0.9786 | 0.3717 | 0.8777 | 12/12 |

### Speed summary

| avg_ms_per_img | fps | img_size | forward_mode |
|---|---|---|---|
| 18.52 | 54.0 | 288 | predict |

### Alignment note

This method is part of the repo-local BaoIAD inventory under the **Discriminative** family. The alignment record is [`docs/alignment/simplenet.md`](../../docs/alignment/simplenet.md); it preserves the detailed strict-alignment evidence, including reference freeze notes, code-path checks, probes, and archived benchmark stop-lines.

<!-- BaoIAD repo-local evidence: end -->
