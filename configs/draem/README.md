## DRAEM

> DRAEM -- A discriminatively trained reconstruction embedding for surface anomaly detection

- **Paper**: [DRAEM -- A discriminatively trained reconstruction embedding for surface anomaly detection](https://arxiv.org/abs/2108.07610)
- **Category**: Reconstruction
- **Backbone**: AE

DRAEM trains a reconstruction network alongside a discriminative network for anomaly detection. The reconstructor restores normal appearance from potentially anomalous inputs, while the discriminator compares the reconstructed and original images to localize anomalies. During training, synthetic anomalies are generated using Perlin noise and DTD textures, and both networks are trained on these augmented samples. At inference, the discriminator outputs a pixel-level anomaly map by comparing the reconstructed and original images.

### Configs

| Config | Description |
|--------|-------------|
| [`draem_256_mvtec_strict.py`](draem_256_mvtec_strict.py) | MVTec AD strict alignment |
| [`draem_256_visa.py`](draem_256_visa.py) | VisA |

<!-- BaoIAD repo-local evidence: start -->

### MVTec AD result summary

| #cat | img mean±std | pxl mean±std | AUPRO mean±std | AUPIMO mean±std | iECE mean±std | pECE mean±std |
|---|---|---|---|---|---|---|
| 15 | 0.9677±0.0061 | 0.9259±0.0081 | 0.8536±0.0089 | 0.4725±0.0287 | 0.3762±0.0114 | 0.0234±0.0038 |

### VisA result summary

| img_AUROC | pxl_AUROC | img_F1max | pxl_F1max | img_AP | pxl_AP | AUPRO | cats |
|---|---|---|---|---|---|---|---|
| 0.9445 | 0.9026 | 0.9045 | 0.3729 | 0.9562 | 0.3060 | 0.6913 | 12/12 |

### Speed summary

| avg_ms_per_img | fps | img_size | forward_mode |
|---|---|---|---|
| 9.91 | 100.9 | 256 | predict |

### Alignment note

This method is part of the repo-local BaoIAD inventory under the **Self-supervised synthesis** family. The alignment record is [`docs/alignment/draem.md`](../../docs/alignment/draem.md); it preserves the detailed strict-alignment evidence, including reference freeze notes, code-path checks, probes, and archived benchmark stop-lines.

<!-- BaoIAD repo-local evidence: end -->
