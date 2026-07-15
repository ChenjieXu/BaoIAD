## AACLIP

> AACLIP: Attentive-Affinitive CLIP for Zero-Shot Anomaly Detection

- **Paper**: [publication](https://arxiv.org/abs/2503.06661)
- **Implementation source**: [upstream repository](https://github.com/Mwxinnn/AA-CLIP); revision: `53db195f230442aa118c246876c94ba1c76139cc`
- **Category**: Vision-Language
- **Backbone**: OpenCLIP-ViT/L-14

AACLIP learns attentive-affinitive visual and textual prompts to align CLIP features for zero-shot anomaly detection. The key innovation is the attentive affinitive module that captures both global semantic alignment and local patch-level anomaly cues through cross-attention between visual and textual features. During training, AACLIP optimizes the prompt parameters on normal images to align normal representations while separating anomalous patterns. At inference, the learned prompts are applied to CLIP to compute anomaly scores from the discrepancy between visual and textual embeddings.

### Configs

| Config | Description |
|--------|-------------|
| [`aaclip_vitl14_336_256_mvtec.py`](aaclip_vitl14_336_256_mvtec.py) | MVTec AD |
| [`aaclip_vitl14_336_518_mvtec_strict.py`](aaclip_vitl14_336_518_mvtec_strict.py) | MVTec AD reference configuration |
| [`aaclip_vitl14_336_518_visa_32shot_stage1.py`](aaclip_vitl14_336_518_visa_32shot_stage1.py) | VisA, Stage 1, 32-shot |
| [`aaclip_vitl14_336_518_visa_32shot_stage2.py`](aaclip_vitl14_336_518_visa_32shot_stage2.py) | VisA, Stage 2, 32-shot |
| [`aaclip_vitl14_336_518_visa_fullshot_stage1.py`](aaclip_vitl14_336_518_visa_fullshot_stage1.py) | VisA, Stage 1, full-shot |
| [`aaclip_vitl14_336_518_visa_fullshot_stage2.py`](aaclip_vitl14_336_518_visa_fullshot_stage2.py) | VisA, Stage 2, full-shot |

<!-- BaoIAD repo-local evidence: start -->

### MVTec AD result summary

| #cat | img mean±std | pxl mean±std | AUPRO mean±std | AUPIMO mean±std | iECE mean±std | pECE mean±std |
|---|---|---|---|---|---|---|
| 15 | 0.9484±0.0000 | 0.9763±0.0000 | — | — | — | — |

### VisA result summary

| img_AUROC | pxl_AUROC | img_F1max | pxl_F1max | img_AP | pxl_AP | AUPRO | cats |
|---|---|---|---|---|---|---|---|
| 0.4616 | 0.4450 | — | — | — | — | — | agg |

### Speed summary

| avg_ms_per_img | fps | img_size | forward_mode |
|---|---|---|---|
| 106.05 | 9.4 | 518 | predict |

### Alignment note

This method is part of the repo-local BaoIAD inventory under the **Vision-language / foundation** family. The [implementation provenance and reproducibility record](../../docs/alignment/aaclip.md) is indexed by the [public method-status manifest](../../docs/alignment/method_status.json). Its manifest validation state is **partially verified**. Referenced raw evidence is not distributed with this repository. The result tables above are historical repository summaries, not a unified public ranking, and should be interpreted according to that manifest state.

<!-- BaoIAD repo-local evidence: end -->
