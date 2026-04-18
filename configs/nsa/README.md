## NSA

> NSA: Natural Synthetic Anomalies for Self-Supervised Anomaly Detection and Segmentation

- **Paper**: [NSA: Natural Synthetic Anomalies for Self-Supervised Anomaly Detection and Segmentation](https://arxiv.org/abs/2109.15222)
- **Category**: Other
- **Backbone**: WRN-50-2

NSA generates natural synthetic anomalies by pasting patches from other images onto training samples for self-supervised anomaly detection. The key innovation is creating more realistic synthetic anomalies compared to random noise, as the pasted patches contain real image content that blends naturally. During training, a segmentation model learns to detect these synthetic perturbations on augmented training images. At inference, the segmentation model outputs pixel-level anomaly predictions for real anomalies.

### Configs

| Config | Description |
|--------|-------------|
| [`nsa_rn18_256_mvtec_strict.py`](nsa_rn18_256_mvtec_strict.py) | MVTec AD strict alignment |
| [`nsa_rn18_256_visa.py`](nsa_rn18_256_visa.py) | VisA |
