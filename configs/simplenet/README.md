## SimpleNet

> SimpleNet: A Simple Network for Image Anomaly Detection and Localization

- **Paper**: [publication](https://arxiv.org/abs/2303.15140)
- **Implementation source**: [upstream repository](https://github.com/DonaldRR/SimpleNet); revision: `351a2b8d4e8cfc944dbccbf9bc6ceda930c6f26b`
- **Category**: Discriminator
- **Backbone**: WRN-50-2

SimpleNet adds a simple feature adaptor and discriminator on top of a frozen pre-trained backbone for anomaly detection. The key innovation is the feature adaptor that adds Gaussian noise to features before discrimination, creating a simple but effective one-class classification boundary. During training, the adaptor and discriminator are trained on normal features with added noise as the negative class. At inference, the discriminator score on adapted features serves as the anomaly score, with both image-level and pixel-level outputs.

### Configs

| Config | Description |
|--------|-------------|
| [`simplenet_wrn50_288_mvtec_strict.py`](simplenet_wrn50_288_mvtec_strict.py) | MVTec AD reference configuration |
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

This method is part of the repo-local BaoIAD inventory under the **Discriminative** family. The [implementation provenance and reproducibility record](../../docs/alignment/simplenet.md) is indexed by the [public method-status manifest](../../docs/alignment/method_status.json). Its manifest validation state is **historical evidence**. Referenced raw evidence is not distributed with this repository. The result tables above are historical repository summaries, not a unified public ranking, and should be interpreted according to that manifest state.

<!-- BaoIAD repo-local evidence: end -->
