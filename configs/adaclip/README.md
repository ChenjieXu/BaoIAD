## AdaCLIP

> AdaCLIP: Adaptive Vision-Language Model for Anomaly Detection

- **Paper**: [AdaCLIP: Adaptive Vision-Language Model for Anomaly Detection](https://arxiv.org/abs/2409.18928)
- **Category**: Vision-Language
- **Backbone**: OpenCLIP-ViT/L-14

AdaCLIP adaptively adjusts CLIP's visual and textual representations using learnable adapters for anomaly detection. It introduces visual and textual adapter modules that modulate CLIP features based on input content, enabling both zero-shot and few-shot detection. The adapters are trained to enhance the discriminability of CLIP features for anomaly vs. normal patterns, with separate adaptation pathways for image-level classification and pixel-level segmentation. At inference, the adapted CLIP computes anomaly scores by comparing adapted visual features against normal and abnormal text prompts.

### Configs

| Config | Description |
|--------|-------------|
| [`adaclip_vitl14_336_518_mvtec_strict.py`](adaclip_vitl14_336_518_mvtec_strict.py) | MVTec AD strict alignment |
| [`adaclip_vitl14_336_518_visa.py`](adaclip_vitl14_336_518_visa.py) | VisA |
