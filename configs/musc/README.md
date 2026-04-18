## MuSc

> MuSc: Mutual Scoring of Pseudo Pairs for Zero-Shot Anomaly Classification and Segmentation

- **Paper**: [MuSc: Mutual Scoring of Pseudo Pairs for Zero-Shot Anomaly Classification and Segmentation](https://arxiv.org/abs/2405.01827)
- **Category**: Vision-Language
- **Backbone**: OpenCLIP-ViT/L-14

MuSc performs zero-shot anomaly detection by mutual scoring of pseudo-pairs between test and reference images using CLIP features. The key idea is to score each test patch against all reference patches without any training, using CLIP's pre-aligned visual features. No training is required — the method computes pairwise similarity scores between test and reference CLIP features at inference time. Anomaly scores are derived from the minimum similarity to reference patches, with both image-level and pixel-level scores available.

### Configs

| Config | Description |
|--------|-------------|
| [`musc_vitl14_336_518_mvtec_strict.py`](musc_vitl14_336_518_mvtec_strict.py) | MVTec AD strict alignment |
| [`musc_vitl14_336_518_visa.py`](musc_vitl14_336_518_visa.py) | VisA |
