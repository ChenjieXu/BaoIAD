## GraphCore

> GraphCore: Few-Shot Industrial Anomaly Detection

- **Paper**: [GraphCore: Few-Shot Industrial Anomaly Detection](https://arxiv.org/abs/2407.11657)
- **Category**: Memory Bank
- **Backbone**: WRN-50-2

GraphCore models normal features as a graph where nodes are patch features and edges capture spatial relationships for anomaly detection. The key idea is that normal images exhibit consistent graph structures, while anomalies disrupt these relationships. During training, the graph reconstruction network learns to reconstruct normal graph structures from node features. At inference, the graph reconstruction error serves as the anomaly score, with anomalous regions producing large errors due to disrupted spatial relationships.

### Configs

| Config | Description |
|--------|-------------|
| [`graphcore_vig_ti_224_mvtec_strict.py`](graphcore_vig_ti_224_mvtec_strict.py) | MVTec AD strict alignment |
| [`graphcore_vig_ti_224_visa.py`](graphcore_vig_ti_224_visa.py) | VisA |
