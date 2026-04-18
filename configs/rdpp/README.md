## RD++

> RD++: Anomaly Detection via Iterative Reverse Distillation

- **Paper**: [RD++: Anomaly Detection via Iterative Reverse Distillation](https://arxiv.org/abs/2307.01348)
- **Category**: Knowledge Distillation
- **Backbone**: WRN-50-2

RD++ extends Reverse Distillation with iterative approximation for anomaly detection. The key innovation is progressively refining the student's inversion through multiple iterations, allowing it to better capture complex normal feature distributions while still failing on anomalies. During training, the student network with iterative refinement is trained to match the teacher's features on normal images. At inference, the accumulated discrepancy across iterations serves as the anomaly score, providing more sensitive detection than single-pass reverse distillation.

### Configs

| Config | Description |
|--------|-------------|
| [`rdpp_wrn50_256_mvtec_strict.py`](rdpp_wrn50_256_mvtec_strict.py) | MVTec AD strict alignment |
| [`rdpp_wrn50_256_visa.py`](rdpp_wrn50_256_visa.py) | VisA |
