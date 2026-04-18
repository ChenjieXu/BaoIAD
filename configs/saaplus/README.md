## SAA+

> SAA+: Segment Any Anomaly+

- **Paper**: Preprint
- **Category**: Other
- **Backbone**: -

SAA+ extends Segment Any Anomaly by combining GroundingDINO and SAM for zero-shot anomaly detection and segmentation. The key idea is leveraging vision-language models to detect anomalies through text prompts (e.g., "defect" or "damage") and then segmenting them using SAM's powerful mask prediction. No training is required — the method uses pre-trained GroundingDINO for anomaly grounding and SAM for mask refinement. At inference, text prompts are used to ground anomalous regions, which are then segmented by SAM for pixel-level anomaly maps.

### Configs

| Config | Description |
|--------|-------------|
| [`saaplus_400_mvtec_strict.py`](saaplus_400_mvtec_strict.py) | MVTec AD strict alignment |
| [`saaplus_400_visa.py`](saaplus_400_visa.py) | VisA |
