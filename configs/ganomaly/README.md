## GANomaly

> GANomaly: Semi-Supervised Anomaly Detection

- **Paper**: [GANomaly: Semi-Supervised Anomaly Detection](https://arxiv.org/abs/1805.06725)
- **Category**: Reconstruction
- **Backbone**: GAN

GANomaly uses an encoder-decoder-encoder GAN architecture for anomaly detection. The first encoder compresses the input, the decoder reconstructs it, and the second encoder re-encodes the reconstruction — the anomaly score comes from the latent distance between the two encoders' outputs. During training, the network is trained on normal images with reconstruction and latent consistency losses. At inference, the L2 distance between the two latent representations serves as the anomaly score, as anomalous inputs produce inconsistent latent codes.

### Configs

| Config | Description |
|--------|-------------|
| [`ganomaly_256_mvtec_strict.py`](ganomaly_256_mvtec_strict.py) | MVTec AD strict alignment |
| [`ganomaly_256_visa.py`](ganomaly_256_visa.py) | VisA |
