## GANomaly

> GANomaly: Semi-Supervised Anomaly Detection

- **Paper**: [publication](https://arxiv.org/abs/1805.06725)
- **Implementation source**: [upstream repository](https://github.com/samet-akcay/ganomaly); revision: `78da4ea9a99f5b02ab60dd651a18def929176d77`
- **Category**: Reconstruction
- **Backbone**: GAN

GANomaly uses an encoder-decoder-encoder GAN architecture for anomaly detection. The first encoder compresses the input, the decoder reconstructs it, and the second encoder re-encodes the reconstruction — the anomaly score comes from the latent distance between the two encoders' outputs. During training, the network is trained on normal images with reconstruction and latent consistency losses. At inference, the L2 distance between the two latent representations serves as the anomaly score, as anomalous inputs produce inconsistent latent codes.

### Configs

| Config | Description |
|--------|-------------|
| [`ganomaly_256_mvtec_strict.py`](ganomaly_256_mvtec_strict.py) | MVTec AD reference configuration |
| [`ganomaly_256_visa.py`](ganomaly_256_visa.py) | VisA |

<!-- BaoIAD repo-local evidence: start -->

### MVTec AD result summary

| #cat | img mean±std | pxl mean±std | AUPRO mean±std | AUPIMO mean±std | iECE mean±std | pECE mean±std |
|---|---|---|---|---|---|---|
| 15 | 0.5934±0.0091 | — | — | — | — | — |

### VisA result summary

| img_AUROC | pxl_AUROC | img_F1max | pxl_F1max | img_AP | pxl_AP | AUPRO | cats |
|---|---|---|---|---|---|---|---|
| 0.6387 | — | 0.7551 | — | 0.6777 | — | — | 12/12 |

### Speed summary

| avg_ms_per_img | fps | img_size | forward_mode |
|---|---|---|---|
| 2.36 | 423.0 | 256 | predict |

### Alignment note

This method is part of the repo-local BaoIAD inventory under the **Reconstruction / ViT** family. The [implementation provenance and reproducibility record](../../docs/alignment/ganomaly.md) is indexed by the [public method-status manifest](../../docs/alignment/method_status.json). Its manifest validation state is **partially verified**. Referenced raw evidence is not distributed with this repository. The result tables above are historical repository summaries, not a unified public ranking, and should be interpreted according to that manifest state.

<!-- BaoIAD repo-local evidence: end -->
