## CFA

> CFA: Coupled-Hypersphere-based Feature Adaptation for Anomaly Detection

- **Paper**: [CFA: Coupled-Hypersphere-based Feature Adaptation for Anomaly Detection](https://arxiv.org/abs/2206.04603)
- **Category**: Discriminator
- **Backbone**: WRN-50-2

CFA adapts features into coupled hyperspheres in a projected space for anomaly detection. The method learns to map normal features onto compact hypersphere clusters while anomalous features fall outside these clusters. During training, CFA optimizes the projection to minimize the distance of normal features to their assigned hypersphere centers using a coupled loss function. At inference, anomaly scores are computed as the distance from test features to the nearest hypersphere center, with large distances indicating anomalies.

### Configs

| Config | Description |
|--------|-------------|
| [`cfa_256_mvtec_strict.py`](cfa_256_mvtec_strict.py) | MVTec AD strict alignment |
| [`cfa_256_visa.py`](cfa_256_visa.py) | VisA |
