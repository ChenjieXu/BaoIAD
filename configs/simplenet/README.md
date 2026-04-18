## SimpleNet

> SimpleNet: A Simple Network for Image Anomaly Detection and Localization

- **Paper**: [SimpleNet: A Simple Network for Image Anomaly Detection and Localization](https://arxiv.org/abs/2303.15140)
- **Category**: Discriminator
- **Backbone**: WRN-50-2

SimpleNet adds a simple feature adaptor and discriminator on top of a frozen pre-trained backbone for anomaly detection. The key innovation is the feature adaptor that adds Gaussian noise to features before discrimination, creating a simple but effective one-class classification boundary. During training, the adaptor and discriminator are trained on normal features with added noise as the negative class. At inference, the discriminator score on adapted features serves as the anomaly score, with both image-level and pixel-level outputs.

### Configs

| Config | Description |
|--------|-------------|
| [`simplenet_wrn50_288_mvtec_strict.py`](simplenet_wrn50_288_mvtec_strict.py) | MVTec AD strict alignment |
| [`simplenet_wrn50_288_visa.py`](simplenet_wrn50_288_visa.py) | VisA |
