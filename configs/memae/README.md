## MemAE

> MemAE: Memory-Augmented Deep Autoencoding

- **Paper**: [MemAE: Memory-Augmented Deep Autoencoding](https://arxiv.org/abs/1904.02639)
- **Category**: Reconstruction
- **Backbone**: Conv AE

MemAE augments an autoencoder with a memory module that stores prototypical normal patterns for anomaly detection. The memory module forces the decoder to reconstruct only normal-like patterns by attending to stored normal prototypes, even when given anomalous inputs. During training, both the autoencoder and memory module are trained on normal images with a sparsity constraint on memory attention. At inference, anomalous inputs produce large reconstruction errors because the memory-constrained decoder cannot reconstruct anomalous patterns.

### Configs

| Config | Description |
|--------|-------------|
| [`memae_avenue_256_official.py`](memae_avenue_256_official.py) | Avenue (video), official |
| [`memae_ucsdped1_256_official.py`](memae_ucsdped1_256_official.py) | UCSD Ped (video), official |
| [`memae_ucsdped2_256_official.py`](memae_ucsdped2_256_official.py) | UCSD Ped (video), official |
| [`memae_wrn50_256_mvtec.py`](memae_wrn50_256_mvtec.py) | MVTec AD |
| [`memae_wrn50_256_mvtec_adapted.py`](memae_wrn50_256_mvtec_adapted.py) | MVTec AD, adapted |
| [`memae_wrn50_256_visa.py`](memae_wrn50_256_visa.py) | VisA |
