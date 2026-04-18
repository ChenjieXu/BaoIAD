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
