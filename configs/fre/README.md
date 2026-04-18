## FRE

> Feature Reconstruction Error for Anomaly Detection

- **Paper**: [Feature Reconstruction Error for Anomaly Detection](https://arxiv.org/abs/2104.01315)
- **Category**: Reconstruction
- **Backbone**: WRN-50-2

FRE detects anomalies by reconstructing features from a pre-trained encoder using a learned decoder. The key idea is that a decoder trained on normal features will fail to reconstruct anomalous features, producing large reconstruction errors. During training, the decoder learns to reconstruct the encoder's features for normal images. At inference, the feature reconstruction error serves as the anomaly score, with both image-level scores from global pooling and pixel-level scores from per-position errors.

### Configs

| Config | Description |
|--------|-------------|
| [`fre_256_mvtec_strict.py`](fre_256_mvtec_strict.py) | MVTec AD strict alignment |
| [`fre_256_visa.py`](fre_256_visa.py) | VisA |
