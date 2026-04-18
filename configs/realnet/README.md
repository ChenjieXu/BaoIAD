## RealNet

> RealNet: Anomaly Detection via Residual Analysis

- **Paper**: [RealNet: Anomaly Detection via Residual Analysis](https://arxiv.org/abs/2401.09822)
- **Category**: Other
- **Backbone**: WRN-50-2

RealNet detects anomalies by residual analysis across multiple feature scales for accurate localization. The key idea is adaptively selecting and fusing reconstruction residuals from different scales, emphasizing the most discriminative residuals for each anomaly type. During training, the reconstruction network and residual selection module are trained jointly on normal images. At inference, the adaptively weighted multi-scale residuals provide the anomaly map, with image-level scores from the aggregated residual magnitude.

### Configs

| Config | Description |
|--------|-------------|
| [`realnet_wrn50_256_mvtec_strict.py`](realnet_wrn50_256_mvtec_strict.py) | MVTec AD strict alignment |
| [`realnet_wrn50_256_visa.py`](realnet_wrn50_256_visa.py) | VisA |
