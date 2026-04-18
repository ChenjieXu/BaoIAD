## Dinomaly

> Dinomaly: Less Is More in Anomaly Detection

- **Paper**: [Dinomaly: Less Is More in Anomaly Detection](https://arxiv.org/abs/2405.14525)
- **Category**: Knowledge Distillation
- **Backbone**: DINOv2-ViT-B/14

Dinomaly uses DINOv2 as a frozen encoder with a lightweight decoder for anomaly detection. The key innovation is using DINOv2's powerful self-supervised features with minimal architectural overhead — only a simple reconstruction decoder is trained. During training, the decoder learns to reconstruct DINOv2 features of normal images. At inference, the reconstruction error in feature space serves as the anomaly score, with both image-level and pixel-level localization.

### Configs

| Config | Description |
|--------|-------------|
| [`dinomaly_392_mvtec_strict.py`](dinomaly_392_mvtec_strict.py) | MVTec AD strict alignment |
| [`dinomaly_392_visa.py`](dinomaly_392_visa.py) | VisA |
