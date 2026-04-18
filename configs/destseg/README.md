## DeSTSeg

> DeSTSeg: Segmentation Guided Denoising for Anomaly Detection

- **Paper**: [DeSTSeg: Segmentation Guided Denoising for Anomaly Detection](https://arxiv.org/abs/2304.08401)
- **Category**: Reconstruction
- **Backbone**: ResNet-18

DeSTSeg combines a denoising student-teacher network with a segmentation module for anomaly detection. The student network learns to denoise the teacher's features, and the denoising error highlights anomalous regions where the student fails to reconstruct normal patterns. During training, both the denoising network and segmentation head are trained jointly, with the denoising error providing a training signal for segmentation. At inference, the segmentation network directly outputs pixel-level anomaly predictions, guided by the denoising error.

### Configs

| Config | Description |
|--------|-------------|
| [`destseg_rn18_256_mvtec_strict.py`](destseg_rn18_256_mvtec_strict.py) | MVTec AD strict alignment |
| [`destseg_rn18_256_visa.py`](destseg_rn18_256_visa.py) | VisA |
