## PyramidFlow

> PyramidFlow: High-Resolution Defect Contrastive Localization using Pyramid Normalizing Flow

- **Paper**: [PyramidFlow: High-Resolution Defect Contrastive Localization using Pyramid Normalizing Flow](https://arxiv.org/abs/2306.02612)
- **Category**: Normalizing Flow
- **Backbone**: WRN-50-2

PyramidFlow performs anomaly detection using pyramid normalizing flows that model multi-scale feature distributions. The pyramid architecture captures feature distributions at multiple resolutions simultaneously, enabling high-resolution defect localization that single-scale flows cannot achieve. During training, the pyramid flow learns to map multi-scale normal features to a standard normal distribution. At inference, the negative log-likelihood at each scale and position provides the anomaly map, with high-resolution localization from the finest pyramid level.

### Configs

| Config | Description |
|--------|-------------|
| [`pyramidflow_fnf_256_mvtec_strict.py`](pyramidflow_fnf_256_mvtec_strict.py) | MVTec AD strict alignment (FNF backbone) |
| [`pyramidflow_resnet18_1024_mvtec_strict.py`](pyramidflow_resnet18_1024_mvtec_strict.py) | MVTec AD strict alignment |
| [`pyramidflow_resnet18_1024_visa.py`](pyramidflow_resnet18_1024_visa.py) | VisA |
