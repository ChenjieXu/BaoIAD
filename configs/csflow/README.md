## CSFlow

> Cross-Scale-Flows for Image-based Anomaly Detection

- **Paper**: [Cross-Scale-Flows for Image-based Anomaly Detection](https://arxiv.org/abs/2110.02855)
- **Category**: Normalizing Flow
- **Backbone**: CSFlow Feature Ext.

Cross-Scale-Flows performs anomaly detection by modeling the distribution of multi-scale features using cross-scale normalizing flows. Unlike single-scale flows, Cross-Scale-Flows captures dependencies between features at different resolutions through cross-scale coupling layers. During training, the flow learns to map multi-scale normal features to a standard normal distribution. At inference, the negative log-likelihood across all scales provides the anomaly score, with localization achieved by computing per-position likelihoods.

### Configs

| Config | Description |
|--------|-------------|
| [`csflow_256_mvtec_strict.py`](csflow_256_mvtec_strict.py) | MVTec AD strict alignment |
| [`csflow_256_visa.py`](csflow_256_visa.py) | VisA |
