## DFM

> Deep Feature Modeling for Anomaly Detection

- **Paper**: [Deep Feature Modeling for Anomaly Detection](https://arxiv.org/abs/1909.10786)
- **Category**: Memory Bank
- **Backbone**: WRN-50-2

DFM fits a linear subspace to deep features of normal samples using PCA, then detects anomalies via the reconstruction error from projecting onto the subspace. The key idea is that normal features lie near a low-dimensional subspace, while anomalous features have significant components outside it. No training is required beyond computing the PCA projection on normal features. At inference, the reconstruction error (distance from the subspace) serves as the anomaly score, with larger errors indicating anomalies.

### Configs

| Config | Description |
|--------|-------------|
| [`dfm_256_mvtec_strict.py`](dfm_256_mvtec_strict.py) | MVTec AD strict alignment |
| [`dfm_256_visa.py`](dfm_256_visa.py) | VisA |
