## SPADE

> SPADE: Sub-Image Anomaly Detection

- **Paper**: [SPADE: Sub-Image Anomaly Detection](https://arxiv.org/abs/2005.02357)
- **Category**: Memory Bank
- **Backbone**: WRN-50-2

SPADE detects anomalies by comparing test patch features against a memory of normal patch features at both image and pixel levels. The key idea is using k-nearest-neighbor distances in feature space — normal patches have nearby neighbors in the normal memory, while anomalous patches do not. No training is required beyond feature extraction and storing normal patch features. At inference, anomaly scores are computed as the distance to the k-nearest normal neighbors, with image-level scores from image features and pixel-level scores from patch features.

### Configs

| Config | Description |
|--------|-------------|
| [`spade_wrn50_256_mvtec_strict.py`](spade_wrn50_256_mvtec_strict.py) | MVTec AD strict alignment |
| [`spade_wrn50_256_visa.py`](spade_wrn50_256_visa.py) | VisA |
