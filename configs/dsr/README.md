## DSR

> DSR: A Dual Subspace Re-projection Network for Surface Anomaly Detection

- **Paper**: [DSR: A Dual Subspace Re-projection Network for Surface Anomaly Detection](https://arxiv.org/abs/2207.06608)
- **Category**: Reconstruction
- **Backbone**: AE + Subspace

DSR uses a dual subspace re-projection network that projects features onto normal and anomaly subspaces. The method learns two complementary subspaces — one capturing normal patterns and one capturing anomaly patterns — and detects anomalies via the re-projection residual. During training, the dual subspace is learned from normal features using a reconstruction objective. At inference, the residual from re-projecting test features onto the normal subspace serves as the anomaly score.

### Configs

| Config | Description |
|--------|-------------|
| [`dsr_256_mvtec_strict.py`](dsr_256_mvtec_strict.py) | MVTec AD strict alignment |
| [`dsr_256_visa.py`](dsr_256_visa.py) | VisA |
