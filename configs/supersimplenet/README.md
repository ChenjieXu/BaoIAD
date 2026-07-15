## SuperSimpleNet

> SuperSimpleNet: Unifying Unsupervised and Supervised Learning for Fast Anomaly Detection

- **Paper**: [publication](https://arxiv.org/abs/2408.03143)
- **Implementation source**: [upstream repository](https://github.com/open-edge-platform/anomalib); revision: `4f6af1acb0ee7b81f54cda036dd9f1c27f63b69a`
- **Category**: Discriminator
- **Backbone**: WRN-50-2

SuperSimpleNet unifies unsupervised and supervised anomaly detection in a single lightweight architecture. The key innovation is supporting both one-class (unsupervised) and few-shot (supervised) settings through a shared backbone with task-specific heads, enabling flexible deployment. During training, the network is trained on normal images for unsupervised mode, or with both normal and anomalous examples for supervised mode. At inference, the appropriate head produces anomaly scores, with the unsupervised head using feature distance and the supervised head using classification.

### Configs

| Config | Description |
|--------|-------------|
| [`supersimplenet_256_mvtec_strict.py`](supersimplenet_256_mvtec_strict.py) | MVTec AD reference configuration |
| [`supersimplenet_256_visa.py`](supersimplenet_256_visa.py) | VisA |

<!-- BaoIAD repo-local evidence: start -->

### MVTec AD result summary

| #cat | img mean±std | pxl mean±std | AUPRO mean±std | AUPIMO mean±std | iECE mean±std | pECE mean±std |
|---|---|---|---|---|---|---|
| 15 | 0.9826±0.0010 | 0.9762±0.0004 | 0.9088±0.0013 | 0.6549±0.0142 | 0.1660±0.0029 | 0.2892±0.0040 |

### VisA result summary

| img_AUROC | pxl_AUROC | img_F1max | pxl_F1max | img_AP | pxl_AP | AUPRO | cats |
|---|---|---|---|---|---|---|---|
| 0.9443 | 0.9832 | 0.9083 | 0.4023 | 0.9575 | 0.3554 | 0.8748 | 12/12 |

### Speed summary

| avg_ms_per_img | fps | img_size | forward_mode |
|---|---|---|---|
| 7.18 | 139.2 | 256 | predict |

### Alignment note

This method is part of the repo-local BaoIAD inventory under the **Discriminative** family. The [implementation provenance and reproducibility record](../../docs/alignment/supersimplenet.md) is indexed by the [public method-status manifest](../../docs/alignment/method_status.json). Its manifest validation state is **partially verified**. Referenced raw evidence is not distributed with this repository. The result tables above are historical repository summaries, not a unified public ranking, and should be interpreted according to that manifest state.

<!-- BaoIAD repo-local evidence: end -->
