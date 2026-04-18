## InvAD

> InvAD: Invertible Anomaly Detection

- **Paper**: [InvAD: Invertible Anomaly Detection](https://arxiv.org/abs/2312.02369)
- **Category**: Other
- **Backbone**: WRN-50-2

InvAD uses invertible transformations to map normal features to a latent space for multi-class anomaly detection. The invertible architecture ensures no information loss during the forward pass, and anomalies are detected via the inverse mapping error. During training, the invertible network learns to map normal features from multiple categories to well-structured latent representations. At inference, the reconstruction error from the inverse mapping serves as the anomaly score, supporting multi-class detection in a single model.

### Configs

| Config | Description |
|--------|-------------|
| [`invad_wrn50_256_mvtec_strict.py`](invad_wrn50_256_mvtec_strict.py) | MVTec AD strict alignment |
| [`invad_wrn50_256_visa.py`](invad_wrn50_256_visa.py) | VisA |
