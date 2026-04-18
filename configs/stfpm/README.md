## STFPM

> STFPM: Student-Teacher Feature Pyramid Matching

- **Paper**: [STFPM: Student-Teacher Feature Pyramid Matching](https://arxiv.org/abs/2010.06348)
- **Category**: Knowledge Distillation
- **Backbone**: WRN-50-2

STFPM trains a student network to match the multi-scale feature pyramid of a pre-trained teacher for anomaly detection. The key idea is that the student can only learn to reproduce normal feature patterns; anomalous inputs produce feature discrepancies at one or more scales. During training, the student network is trained to minimize the cosine distance between its feature pyramid and the teacher's on normal images. At inference, the multi-scale feature discrepancy serves as the anomaly score, with localization from the finest-scale difference.

### Configs

| Config | Description |
|--------|-------------|
| [`stfpm_rn18_256_mvtec_strict.py`](stfpm_rn18_256_mvtec_strict.py) | MVTec AD strict alignment |
| [`stfpm_rn18_256_visa.py`](stfpm_rn18_256_visa.py) | VisA |
