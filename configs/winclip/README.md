## WinCLIP

> WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation

- **Paper**: [publication](https://arxiv.org/abs/2303.14814)
- **Implementation source**: [upstream repository](https://github.com/open-edge-platform/anomalib); revision not pinned in the method-status manifest
- **Category**: Vision-Language
- **Backbone**: OpenCLIP-ViT/L-14

WinCLIP performs zero-shot and few-shot anomaly detection by computing windowed CLIP similarity scores between image patches and text descriptions of normal/abnormal concepts. The key innovation is the multi-scale windowed scoring that aggregates CLIP similarities at different spatial granularities, capturing both local defects and global anomalies. No training is required for zero-shot mode — the method uses hand-crafted text prompts with CLIP's pre-trained encoders. At inference, anomaly scores are computed from the similarity between windowed visual features and normal/abnormal text embeddings, with few-shot mode using reference images to refine the scores.

### Configs

| Config | Description |
|--------|-------------|
| [`winclip_256_mvtec.py`](winclip_256_mvtec.py) | MVTec AD |
| [`winclip_256_visa.py`](winclip_256_visa.py) | VisA |

<!-- BaoIAD repo-local evidence: start -->

### MVTec AD result summary

| #cat | img mean±std | pxl mean±std | AUPRO mean±std | AUPIMO mean±std | iECE mean±std | pECE mean±std |
|---|---|---|---|---|---|---|
| 15 | 0.8973±0.0000 | 0.6318±0.0000 | 0.2412±0.0000 | 0.0000±0.0000 | 0.3579±0.0000 | 0.3433±0.0000 |

### VisA result summary

| img_AUROC | pxl_AUROC | img_F1max | pxl_F1max | img_AP | pxl_AP | AUPRO | cats |
|---|---|---|---|---|---|---|---|
| 0.7505 | 0.5996 | 0.7846 | 0.0296 | 0.7853 | 0.0102 | 0.2475 | 12/12 |

### Speed summary

| avg_ms_per_img | fps | img_size | forward_mode |
|---|---|---|---|
| 935.89 | 1.1 | 256 | predict |

### Alignment note

This method is part of the repo-local BaoIAD inventory under the **Vision-language / foundation** family. The [implementation provenance and reproducibility record](../../docs/alignment/winclip.md) is indexed by the [public method-status manifest](../../docs/alignment/method_status.json). Its manifest validation state is **historical evidence**. Referenced raw evidence is not distributed with this repository. The result tables above are historical repository summaries, not a unified public ranking, and should be interpreted according to that manifest state.

<!-- BaoIAD repo-local evidence: end -->
