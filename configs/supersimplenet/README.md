## SuperSimpleNet

> SuperSimpleNet: Unifying Unsupervised and Supervised Learning for Fast Anomaly Detection

- **Paper**: [SuperSimpleNet: Unifying Unsupervised and Supervised Learning for Fast Anomaly Detection](https://arxiv.org/abs/2407.02959)
- **Category**: Discriminator
- **Backbone**: WRN-50-2

SuperSimpleNet unifies unsupervised and supervised anomaly detection in a single lightweight architecture. The key innovation is supporting both one-class (unsupervised) and few-shot (supervised) settings through a shared backbone with task-specific heads, enabling flexible deployment. During training, the network is trained on normal images for unsupervised mode, or with both normal and anomalous examples for supervised mode. At inference, the appropriate head produces anomaly scores, with the unsupervised head using feature distance and the supervised head using classification.

### Configs

| Config | Description |
|--------|-------------|
| [`supersimplenet_256_mvtec_strict.py`](supersimplenet_256_mvtec_strict.py) | MVTec AD strict alignment |
| [`supersimplenet_256_visa.py`](supersimplenet_256_visa.py) | VisA |
