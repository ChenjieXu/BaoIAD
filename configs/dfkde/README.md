## DFKDE

> Deep Feature Kernel Density Estimation for Anomaly Detection

- **Paper**: [Deep Feature Kernel Density Estimation for Anomaly Detection](https://arxiv.org/abs/1909.10786)
- **Category**: Memory Bank
- **Backbone**: WRN-50-2

DFKDE applies kernel density estimation on deep features extracted from a pre-trained backbone to model the normal distribution. The method stores a subset of normal features and uses a Gaussian kernel to estimate the density at test points. No training is required beyond feature extraction and selecting the reference subset. At inference, the estimated density at each test feature serves as the anomaly score — low density indicates anomalies, with both image-level and pixel-level scores available.

### Configs

| Config | Description |
|--------|-------------|
| [`dfkde_256_mvtec_strict.py`](dfkde_256_mvtec_strict.py) | MVTec AD strict alignment |
| [`dfkde_256_visa.py`](dfkde_256_visa.py) | VisA |
