## CutPaste

> CutPaste: Self-Supervised Learning for Anomaly Detection and Segmentation

- **Paper**: [CutPaste: Self-Supervised Learning for Anomaly Detection and Segmentation](https://arxiv.org/abs/2104.02515)
- **Category**: Other
- **Backbone**: ResNet-18

CutPaste creates synthetic anomalies by cut-and-paste augmentation of training patches, then trains a binary classifier to distinguish normal from augmented samples. The key idea is that self-supervised classification on synthetic anomalies provides a good proxy for real anomaly detection. During training, patches are either cut out and pasted at random locations (CutPaste) or swapped between images (CutSwap), and a classifier learns to detect these perturbations. At inference, the classifier's anomaly probability serves as the detection score.

### Configs

| Config | Description |
|--------|-------------|
| [`cutpaste_rn18_256_mvtec_strict.py`](cutpaste_rn18_256_mvtec_strict.py) | MVTec AD strict alignment |
| [`cutpaste_rn18_256_visa.py`](cutpaste_rn18_256_visa.py) | VisA |
