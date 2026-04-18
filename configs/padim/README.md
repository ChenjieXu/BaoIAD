## PaDiM

> PaDiM: A Patch Distribution Modeling Framework for Anomaly Detection

- **Paper**: [PaDiM: A Patch Distribution Modeling Framework for Anomaly Detection](https://arxiv.org/abs/2011.08785)
- **Category**: Memory Bank
- **Backbone**: WRN-50-2

PaDiM models the distribution of patch-level features using multivariate Gaussian distributions per spatial position for anomaly detection. The key idea is that normal patches at each position follow a Gaussian distribution, and anomalies deviate from their position-specific distribution. No training is required beyond extracting features and computing per-position mean and covariance from normal images. At inference, the Mahalanobis distance from each test patch to its position-specific Gaussian serves as the anomaly score.

### Configs

| Config | Description |
|--------|-------------|
| [`padim_wrn50_256_mvtec_strict.py`](padim_wrn50_256_mvtec_strict.py) | MVTec AD strict alignment |
| [`padim_wrn50_256_visa.py`](padim_wrn50_256_visa.py) | VisA |
