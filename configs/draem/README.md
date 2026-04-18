## DRAEM

> DRAEM: A discriminatively trained anomaly detection method

- **Paper**: [DRAEM: A discriminatively trained anomaly detection method](https://arxiv.org/abs/2108.07610)
- **Category**: Reconstruction
- **Backbone**: AE

DRAEM trains a reconstruction network alongside a discriminative network for anomaly detection. The reconstructor restores normal appearance from potentially anomalous inputs, while the discriminator compares the reconstructed and original images to localize anomalies. During training, synthetic anomalies are generated using Perlin noise and DTD textures, and both networks are trained on these augmented samples. At inference, the discriminator outputs a pixel-level anomaly map by comparing the reconstructed and original images.

### Configs

| Config | Description |
|--------|-------------|
| [`draem_256_mvtec_strict.py`](draem_256_mvtec_strict.py) | MVTec AD strict alignment |
| [`draem_256_visa.py`](draem_256_visa.py) | VisA |
