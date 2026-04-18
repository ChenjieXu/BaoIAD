## ResAD

> ResAD: Residual Attention Anomaly Detection

- **Paper**: [ResAD: Residual Attention Anomaly Detection](https://arxiv.org/abs/2407.11657)
- **Category**: Other
- **Backbone**: WRN-50-2

ResAD uses residual attention mechanisms to focus on anomalous regions for anomaly detection. The key idea is that attention-weighted feature residuals highlight subtle anomalies that uniform feature comparison might miss, by adaptively weighting feature differences based on their discriminative importance. During training, the attention module learns to focus on informative feature channels for distinguishing normal from anomalous patterns. At inference, the attention-weighted residual between test and reference features serves as the anomaly score.

### Configs

| Config | Description |
|--------|-------------|
| [`resad_official_visa_to_mvtec.py`](resad_official_visa_to_mvtec.py) | MVTec AD, VisA, official, transfer VisA to MVTec AD |
| [`resad_wrn50_256_mvtec_strict.py`](resad_wrn50_256_mvtec_strict.py) | MVTec AD strict alignment |
