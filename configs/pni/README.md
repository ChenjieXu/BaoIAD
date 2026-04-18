## PNI

> PNI: Probabilistic Noise Identification for Anomaly Detection

- **Paper**: [PNI: Probabilistic Noise Identification for Anomaly Detection](https://arxiv.org/abs/2308.04668)
- **Category**: Other
- **Backbone**: WRN-50-2

PNI identifies anomalies by modeling the noise in feature space as a probabilistic distribution. The key idea is that normal features exhibit consistent noise patterns, and deviations from the learned noise distribution indicate anomalies. During training, the noise model learns the distribution of feature perturbations for normal images. At inference, the likelihood of observed noise under the learned model serves as the anomaly score, with unlikely noise patterns indicating anomalies.

### Configs

| Config | Description |
|--------|-------------|
| [`pni_wrn101_480_mvtec_strict.py`](pni_wrn101_480_mvtec_strict.py) | MVTec AD strict alignment |
| [`pni_wrn101_480_visa.py`](pni_wrn101_480_visa.py) | VisA |
