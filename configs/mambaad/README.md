## MambaAD

> MambaAD: Exploring State Space Models for Multi-class Unsupervised Anomaly Detection

- **Paper**: [MambaAD: Exploring State Space Models for Multi-class Unsupervised Anomaly Detection](https://arxiv.org/abs/2404.06564)
- **Category**: Other
- **Backbone**: EffNet-B4

MambaAD replaces the traditional decoder with Mamba (state space model) blocks for efficient multi-class anomaly detection. The key innovation is using Mamba's selective state space mechanism to capture long-range dependencies with linear complexity, replacing the quadratic cost of attention-based decoders. During training, the Mamba decoder learns to reconstruct normal features across multiple categories simultaneously. At inference, the feature reconstruction error from the Mamba decoder provides anomaly scores for both image-level detection and pixel-level localization.

### Configs

| Config | Description |
|--------|-------------|
| [`mambaad_effnet_b4_256_mvtec_strict.py`](mambaad_effnet_b4_256_mvtec_strict.py) | MVTec AD strict alignment |
| [`mambaad_effnet_b4_256_visa.py`](mambaad_effnet_b4_256_visa.py) | VisA |
