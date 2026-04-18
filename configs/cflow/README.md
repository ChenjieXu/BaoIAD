## CFlow-AD

> CFlow-AD: Real-Time Unsupervised Anomaly Detection with Localization via Conditional Normalizing Flows

- **Paper**: [CFlow-AD: Real-Time Unsupervised Anomaly Detection with Localization via Conditional Normalizing Flows](https://arxiv.org/abs/2107.12571)
- **Category**: Normalizing Flow
- **Backbone**: WRN-50-2

CFlow-AD detects anomalies by estimating the likelihood of visual features using conditional normalizing flows. Normalizing flows learn an invertible mapping from the feature distribution to a standard normal distribution, where low-likelihood regions correspond to anomalies. During training, the flow model is trained on normal features to maximize their log-likelihood. At inference, the negative log-likelihood of test features under the learned flow serves as the anomaly score, enabling both image-level detection and pixel-level localization.

### Configs

| Config | Description |
|--------|-------------|
| [`cflow_mvtec_strict.py`](cflow_mvtec_strict.py) | MVTec AD strict alignment |
| [`cflow_visa.py`](cflow_visa.py) | VisA |
