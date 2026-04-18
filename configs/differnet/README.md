## DifferNet

> Same Same But DifferNet: Unsupervised Anomaly Detection

- **Paper**: [Same Same But DifferNet: Unsupervised Anomaly Detection](https://arxiv.org/abs/2008.12577)
- **Category**: Normalizing Flow
- **Backbone**: ResNet-18

DifferNet uses a pre-trained teacher and a student network with different architectures for anomaly detection. The architectural difference creates an asymmetric feature mapping where the student can only learn normal patterns. During training, the student network is trained to match the teacher's output on normal images. At inference, the discrepancy between student and teacher features serves as the anomaly score, with larger differences indicating anomalous regions.

### Configs

| Config | Description |
|--------|-------------|
| [`differnet_alexnet_256_mvtec_strict.py`](differnet_alexnet_256_mvtec_strict.py) | MVTec AD strict alignment |
| [`differnet_alexnet_256_visa.py`](differnet_alexnet_256_visa.py) | VisA |
