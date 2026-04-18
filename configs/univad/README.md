## UniVAD

> UniVAD: Universal Visual Anomaly Detection

- **Paper**: [UniVAD: Universal Visual Anomaly Detection](https://arxiv.org/abs/2406.03687)
- **Category**: Other
- **Backbone**: DINOv2

UniVAD performs universal visual anomaly detection by leveraging DINOv2 features with a lightweight reconstruction module for generalization across categories. The key innovation is using DINOv2's powerful self-supervised features as a universal feature extractor, requiring only a lightweight reconstruction module per category. During training, the reconstruction module learns to reconstruct DINOv2 features of normal images. At inference, the reconstruction error provides anomaly scores, with the universal DINOv2 features enabling strong generalization to unseen categories.

### Configs

| Config | Description |
|--------|-------------|
| [`univad_mvtec.py`](univad_mvtec.py) | MVTec AD |
| [`univad_mvtec_strict.py`](univad_mvtec_strict.py) | MVTec AD strict alignment |
| [`univad_mvtec_visa.py`](univad_mvtec_visa.py) | MVTec AD, VisA |
