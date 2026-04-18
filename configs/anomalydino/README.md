## AnomalyDINO

> AnomalyDINO: DINOv2 for Anomaly Detection

- **Paper**: [AnomalyDINO: DINOv2 for Anomaly Detection](https://arxiv.org/abs/2405.14525)
- **Category**: Vision-Language
- **Backbone**: DINOv2-ViT/L-14

AnomalyDINO leverages DINOv2's self-supervised features for anomaly detection by computing patch-level distances between test and reference features. The key insight is that DINOv2 produces semantically meaningful features without supervision, making it naturally suited for anomaly detection. No training is required — the method stores reference patch features from normal images and compares test features against them. At inference, anomaly scores are computed as the minimum distance between each test patch feature and the reference feature bank, with both image-level and pixel-level scores derived from these distances.

### Configs

| Config | Description |
|--------|-------------|
| [`anomalydino_vitb14_448_mvtec_strict.py`](anomalydino_vitb14_448_mvtec_strict.py) | MVTec AD strict alignment |
| [`anomalydino_vitb14_448_visa.py`](anomalydino_vitb14_448_visa.py) | VisA |
