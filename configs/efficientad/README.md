## EfficientAD

> EfficientAD: Accurate Visual Anomaly Detection at Millisecond Level

- **Paper**: [EfficientAD: Accurate Visual Anomaly Detection at Millisecond Level](https://arxiv.org/abs/2303.14535)
- **Category**: Knowledge Distillation
- **Backbone**: PDN + WRN-50-2

EfficientAD combines a lightweight patch description network (PDN) with a student-teacher distillation branch for anomaly detection. The PDN provides fast feature extraction, while the student-teacher branch captures more subtle anomalies through knowledge distillation. During training, the student network learns to match the teacher's output on normal images, and both branches are trained jointly. At inference, anomaly scores combine the PDN's distance to the normal feature distribution and the student-teacher discrepancy, achieving millisecond-level inference.

### Configs

| Config | Description |
|--------|-------------|
| [`efficientad_256_mvtec_strict.py`](efficientad_256_mvtec_strict.py) | MVTec AD strict alignment |
| [`efficientad_256_visa.py`](efficientad_256_visa.py) | VisA |
