## PatchCore

> PatchCore: Towards Total Recall in Industrial Anomaly Detection

- **Paper**: [PatchCore: Towards Total Recall in Industrial Anomaly Detection](https://arxiv.org/abs/2106.08265)
- **Category**: Memory Bank
- **Backbone**: WRN-50-2

PatchCore builds a memory bank of core patch features from normal training data using greedy coreset subsampling for anomaly detection. The key innovation is the coreset subsampling that selects a small representative subset of patch features, reducing memory and computation while preserving detection accuracy. No model training is required — only feature extraction and coreset selection on normal images. At inference, anomaly scores are computed as the distance from each test patch feature to its nearest neighbor in the coreset memory bank, with both image-level and pixel-level scores.

### Configs

| Config | Description |
|--------|-------------|
| [`patchcore_wrn50_256_mvtec_strict.py`](patchcore_wrn50_256_mvtec_strict.py) | MVTec AD strict alignment |
| [`patchcore_wrn50_256_visa.py`](patchcore_wrn50_256_visa.py) | VisA |
