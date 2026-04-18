## AnomalyCLIP

> AnomalyCLIP: Object-Agnostic Prompt Learning for Zero-Shot Anomaly Detection

- **Paper**: [AnomalyCLIP: Object-Agnostic Prompt Learning for Zero-Shot Anomaly Detection](https://arxiv.org/abs/2403.02199)
- **Category**: Vision-Language
- **Backbone**: OpenCLIP-ViT/L-14

AnomalyCLIP learns object-agnostic text prompts that generalize across categories for zero-shot anomaly detection. The core idea is to decouple object-specific semantics from anomaly cues in CLIP's text encoder, learning universal anomaly descriptions that transfer to unseen categories. It uses learnable text prompts with an auxiliary classifier to separate object identity from anomaly information. At inference, the learned prompts are paired with CLIP's visual encoder to compute patch-level anomaly scores without any category-specific training data.

### Configs

| Config | Description |
|--------|-------------|
| [`anomalyclip_vitl14_336_518_mvtec_strict.py`](anomalyclip_vitl14_336_518_mvtec_strict.py) | MVTec AD strict alignment |
| [`anomalyclip_vitl14_336_518_visa.py`](anomalyclip_vitl14_336_518_visa.py) | VisA |
