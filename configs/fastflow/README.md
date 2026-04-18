## FastFlow

> FastFlow: Unsupervised Anomaly Detection and Localization via Normalizing Flow

- **Paper**: [FastFlow: Unsupervised Anomaly Detection and Localization via Normalizing Flow](https://arxiv.org/abs/2111.07677)
- **Category**: Normalizing Flow
- **Backbone**: WRN-50-2

FastFlow uses fully convolutional normalizing flows to estimate the likelihood of visual features for anomaly detection. The fully convolutional architecture enables efficient per-position likelihood computation, making both training and inference fast. During training, the flow model learns to map normal features to a standard normal distribution by maximizing log-likelihood. At inference, the negative log-likelihood at each spatial position provides the anomaly map, with image-level scores from the average likelihood.

### Configs

| Config | Description |
|--------|-------------|
| [`fastflow_wrn50_256_mvtec_strict.py`](fastflow_wrn50_256_mvtec_strict.py) | MVTec AD strict alignment |
| [`fastflow_wrn50_256_visa.py`](fastflow_wrn50_256_visa.py) | VisA |
