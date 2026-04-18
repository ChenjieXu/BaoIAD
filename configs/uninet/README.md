## UniNet

> UniNet: Unified Architecture for Anomaly Detection

- **Paper**: [UniNet: Unified Architecture for Anomaly Detection](https://arxiv.org/abs/2405.01827)
- **Category**: Other
- **Backbone**: WRN-50-2

UniNet provides a unified architecture for anomaly detection that supports multiple detection paradigms through a shared backbone with task-specific heads. The key idea is sharing feature extraction across different detection methods while maintaining specialized scoring heads for each paradigm. During training, the shared backbone and task-specific heads are trained jointly on normal images from multiple categories. At inference, the appropriate head produces anomaly scores based on the shared features, supporting multiple detection modes in one model.

### Configs

| Config | Description |
|--------|-------------|
| [`uninet_256_mvtec_strict.py`](uninet_256_mvtec_strict.py) | MVTec AD strict alignment |
| [`uninet_256_visa.py`](uninet_256_visa.py) | VisA |
