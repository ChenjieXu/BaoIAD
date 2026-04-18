## ViTAD

> ViTAD: Vision Transformer for Anomaly Detection

- **Paper**: [ViTAD: Vision Transformer for Anomaly Detection](https://arxiv.org/abs/2404.09163)
- **Category**: Other
- **Backbone**: ViT-AD

ViTAD uses a Vision Transformer as the backbone for anomaly detection with a masked knowledge distillation scheme. The key innovation is using masked patches during distillation to force the student to learn robust normal representations, preventing it from simply copying the teacher's behavior. During training, the student ViT learns to reconstruct the teacher's features from partially masked input patches on normal images. At inference, the feature discrepancy between student and teacher serves as the anomaly score, with both image-level and pixel-level localization.

### Configs

| Config | Description |
|--------|-------------|
| [`vitad_256_mvtec_strict.py`](vitad_256_mvtec_strict.py) | MVTec AD strict alignment |
| [`vitad_256_visa.py`](vitad_256_visa.py) | VisA |
