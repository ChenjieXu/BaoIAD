## AnoVL

> AnoVL: Self-Supervised Anomaly Detection by Contrastive Learning

- **Paper**: [AnoVL: Self-Supervised Anomaly Detection by Contrastive Learning](https://arxiv.org/abs/2308.15939)
- **Category**: Vision-Language
- **Backbone**: OpenCLIP-ViT/L-14

AnoVL performs self-supervised anomaly detection by contrastive learning of visual-linguistic representations. It aligns CLIP's visual and textual features for normal patterns through contrastive learning, while anomalies naturally deviate from the aligned normal manifold. The method trains projection heads that map visual and textual features into a shared space where normal features cluster tightly. At inference, anomaly scores are computed from the distance between visual features and the nearest normal textual prototype in the shared space.

### Configs

| Config | Description |
|--------|-------------|
| [`anovl_vitb16plus_240_mvtec_strict.py`](anovl_vitb16plus_240_mvtec_strict.py) | MVTec AD strict alignment |
| [`anovl_vitb16plus_240_visa.py`](anovl_vitb16plus_240_visa.py) | VisA |
