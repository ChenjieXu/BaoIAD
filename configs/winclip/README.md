## WinCLIP

> WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation

- **Paper**: [WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation](https://arxiv.org/abs/2303.14814)
- **Category**: Vision-Language
- **Backbone**: OpenCLIP-ViT/L-14

WinCLIP performs zero-shot and few-shot anomaly detection by computing windowed CLIP similarity scores between image patches and text descriptions of normal/abnormal concepts. The key innovation is the multi-scale windowed scoring that aggregates CLIP similarities at different spatial granularities, capturing both local defects and global anomalies. No training is required for zero-shot mode — the method uses hand-crafted text prompts with CLIP's pre-trained encoders. At inference, anomaly scores are computed from the similarity between windowed visual features and normal/abnormal text embeddings, with few-shot mode using reference images to refine the scores.

### Configs

| Config | Description |
|--------|-------------|
| [`winclip_256_mvtec.py`](winclip_256_mvtec.py) | MVTec AD |
| [`winclip_256_visa.py`](winclip_256_visa.py) | VisA |
