## RD

> RD: Anomaly Detection via Reverse Distillation

- **Paper**: [RD: Anomaly Detection via Reverse Distillation](https://arxiv.org/abs/2201.10703)
- **Category**: Knowledge Distillation
- **Backbone**: WRN-50-2

RD detects anomalies via reverse distillation — a student network learns to invert the teacher's feature mapping. The key idea is that the student can only learn to invert normal patterns; anomalous features produce large inversion errors because they were never seen during training. During training, the student network is trained to reconstruct the teacher's multi-scale features from normal images using a reverse bottleneck. At inference, the cosine distance between student and teacher features serves as the anomaly score, with both image-level and pixel-level localization.

### Configs

| Config | Description |
|--------|-------------|
| [`rd_wrn50_256_mvtec_strict.py`](rd_wrn50_256_mvtec_strict.py) | MVTec AD strict alignment |
| [`rd_wrn50_256_visa.py`](rd_wrn50_256_visa.py) | VisA |
