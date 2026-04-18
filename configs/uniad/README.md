## UniAD

> UniAD: A Unified Model for Multi-class Anomaly Detection

- **Paper**: [UniAD: A Unified Model for Multi-class Anomaly Detection](https://arxiv.org/abs/2206.03687)
- **Category**: Other
- **Backbone**: EffNet-B4

UniAD performs multi-class anomaly detection using a unified transformer-based architecture that reconstructs normal features across all categories in a single model. The key innovation is using neighbor masking and query-based reconstruction to handle multiple categories simultaneously without category-specific components. During training, the transformer learns to reconstruct normal features from all categories using masked attention and learnable queries. At inference, the reconstruction error serves as the anomaly score, with a single model detecting anomalies across all trained categories.

### Configs

| Config | Description |
|--------|-------------|
| [`uniad_wrn50_256_mvtec_strict.py`](uniad_wrn50_256_mvtec_strict.py) | MVTec AD strict alignment |
| [`uniad_wrn50_256_visa.py`](uniad_wrn50_256_visa.py) | VisA |
