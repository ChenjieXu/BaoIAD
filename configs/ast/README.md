## AST

> AST: Asymmetric Student-Teacher for Anomaly Detection

- **Paper**: [AST: Asymmetric Student-Teacher for Anomaly Detection](https://arxiv.org/abs/2207.06808)
- **Category**: Knowledge Distillation
- **Backbone**: WRN-50-2

AST uses an asymmetric student-teacher architecture for anomaly detection. The key idea is that the student network has a smaller capacity than the teacher, so it can only learn to reconstruct normal patterns — anomalous regions produce large reconstruction errors due to the capacity bottleneck. During training, the student learns to match the teacher's output on normal images only. At inference, the pixel-wise difference between student and teacher outputs serves as the anomaly map, with image-level scores derived from the maximum discrepancy.

### Configs

| Config | Description |
|--------|-------------|
| [`ast_effnet_b5_768_mvtec_strict.py`](ast_effnet_b5_768_mvtec_strict.py) | MVTec AD strict alignment |
| [`ast_effnet_b5_768_visa.py`](ast_effnet_b5_768_visa.py) | VisA |
