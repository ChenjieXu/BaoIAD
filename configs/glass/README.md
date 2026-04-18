## GLASS

> GLASS: GLobal Attention-based Anomaly Detection

- **Paper**: [GLASS: GLobal Attention-based Anomaly Detection](https://arxiv.org/abs/2402.12650)
- **Category**: Other
- **Backbone**: WRN-50-2

GLASS generates anomalies via global attention-based Perlin noise masks and DTD textures, then trains a discriminator to detect both synthetic and real anomalies. The key innovation is using global attention to generate more realistic and diverse synthetic anomalies compared to random Perlin masks. During training, synthetic anomalies are blended with normal images using attention-guided masks, and a segmentation network learns to detect these patterns. At inference, the segmentation network outputs pixel-level anomaly predictions.

### Configs

| Config | Description |
|--------|-------------|
| [`glass_wrn50_288_mvtec_strict.py`](glass_wrn50_288_mvtec_strict.py) | MVTec AD strict alignment |
| [`glass_wrn50_288_visa.py`](glass_wrn50_288_visa.py) | VisA |
