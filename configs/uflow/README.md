## U-Flow

> U-Flow: A U-shaped Normalizing Flow for Anomaly Detection

- **Paper**: [U-Flow: A U-shaped Normalizing Flow for Anomaly Detection](https://arxiv.org/abs/2207.09506)
- **Category**: Normalizing Flow
- **Backbone**: WRN-50-2

U-Flow uses a U-shaped normalizing flow architecture that models features at multiple scales with skip connections for anomaly detection. The U-shape with skip connections allows the flow to capture both fine-grained and coarse-grained feature distributions, improving localization accuracy over single-scale flows. During training, the U-shaped flow learns to map multi-scale normal features to a standard normal distribution. At inference, the negative log-likelihood at each position provides the anomaly map, with skip connections enabling precise localization.

### Configs

| Config | Description |
|--------|-------------|
| [`uflow_mcait_448_mvtec_strict.py`](uflow_mcait_448_mvtec_strict.py) | MVTec AD strict alignment |
| [`uflow_mcait_448_visa.py`](uflow_mcait_448_visa.py) | VisA |
