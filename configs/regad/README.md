## RegAD

> RegAD: Registration Based Few-Shot Anomaly Detection

- **Paper**: [RegAD: Registration Based Few-Shot Anomaly Detection](https://arxiv.org/abs/2207.01878)
- **Category**: Memory Bank
- **Backbone**: WRN-50-2

RegAD performs few-shot anomaly detection by learning spatial registration between test and support images. The key innovation is learning a category-agnostic registration network that aligns test images with normal reference images, making anomalies visible as registration errors. During training, the registration network learns to align normal image pairs using spatial transformations. At inference, the registration error between the test image and its nearest normal reference serves as the anomaly score, requiring only a few normal examples per category.

### Configs

| Config | Description |
|--------|-------------|
| [`regad_wrn50_256_mvtec_strict.py`](regad_wrn50_256_mvtec_strict.py) | MVTec AD strict alignment |
| [`regad_wrn50_256_visa.py`](regad_wrn50_256_visa.py) | VisA |
