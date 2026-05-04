## MemSeg

> MemSeg: A Memory-based Segmentation Method for Anomaly Detection

- **Paper**: [MemSeg: A Memory-based Segmentation Method for Anomaly Detection](https://arxiv.org/abs/2206.13116)
- **Category**: Reconstruction
- **Backbone**: ResNet-18

MemSeg uses a memory bank of normal feature descriptors to guide a segmentation network for anomaly detection. The memory bank stores representative normal features, and the segmentation network compares input features against the memory to identify anomalous regions. During training, the memory bank is populated with normal features, and the segmentation network learns to detect discrepancies between input and memory features. At inference, the segmentation network outputs pixel-level anomaly predictions by comparing against the stored normal patterns.

### Configs

| Config | Description |
|--------|-------------|
| [`memseg_rn18_256_mvtec_strict.py`](memseg_rn18_256_mvtec_strict.py) | MVTec AD strict alignment |
| [`memseg_rn18_256_visa.py`](memseg_rn18_256_visa.py) | VisA |

<!-- BaoIAD repo-local evidence: start -->

### MVTec AD result summary

| #cat | img mean±std | pxl mean±std | AUPRO mean±std | AUPIMO mean±std | iECE mean±std | pECE mean±std |
|---|---|---|---|---|---|---|
| 15 | 0.9769±0.0066 | 0.9496±0.0039 | 0.8940±0.0049 | 0.4700±0.0577 | 0.1824±0.0132 | 0.0239±0.0007 |

### VisA result summary

| img_AUROC | pxl_AUROC | img_F1max | pxl_F1max | img_AP | pxl_AP | AUPRO | cats |
|---|---|---|---|---|---|---|---|
| 0.8785 | 0.8153 | 0.8461 | 0.2376 | 0.8948 | 0.1650 | 0.5746 | 12/12 |

### Speed summary

| avg_ms_per_img | fps | img_size | forward_mode |
|---|---|---|---|
| 5.76 | 173.5 | 256 | predict |

### Alignment note

This method is part of the repo-local BaoIAD inventory under the **Reconstruction / ViT** family. The alignment record is [`docs/alignment/memseg.md`](../../docs/alignment/memseg.md); it preserves the detailed strict-alignment evidence, including reference freeze notes, code-path checks, probes, and archived benchmark stop-lines.

<!-- BaoIAD repo-local evidence: end -->
