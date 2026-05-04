# WinCLIP strict-alignment evidence

- **Method slug**: `winclip`
- **Family**: Vision-language / foundation
- **Method README**: [`configs/winclip/README.md`](../../configs/winclip/README.md)

This file preserves the fixed alignment evidence migrated from the manuscript evidence workspace. The content below is translated to English where needed and kept inside this repository so the strict-alignment state can be reviewed without relying on an external paper checkout.

## Config entry points

- [`configs/winclip/winclip_256_mvtec.py`](../../configs/winclip/winclip_256_mvtec.py)
- [`configs/winclip/winclip_256_visa.py`](../../configs/winclip/winclip_256_visa.py)

## Detailed alignment report

**Status**: `aligned`
**Date**: `2026-03-24`

## 1. Reference freezing

- Reference repository: local `.refs/anomalib`
- Reference commit: The current environment cannot read the git metadata of nested repo (`safe.directory` limit). This freeze is based on the `2026-03-24` workspace snapshot.
- Refer to config/checkpoint:
  - `.refs/anomalib/src/anomalib/models/image/winclip/lightning_model.py`
  - `.refs/anomalib/src/anomalib/models/image/winclip/torch_model.py`
  - `.refs/anomalib/src/anomalib/models/image/winclip/prompting.py`
  - `.refs/anomalib/src/anomalib/models/image/winclip/utils.py`
- Dataset/Category: MVTec AD, zero-shot, generate prompt by category
- Input resolution: The reference implementation relies on external pre-processor to do `Resize((240, 240)) + CLIP Normalize`
- seed: README historical benchmark, press `42`
- Indicator definition: image AUROC / pixel AUROC
- intentional diff:
  - BaoIAD retains `configs/winclip/winclip_256_mvtec.py`’s `256x256` unified benchmark caliber
  - Instead of changing it to `240x240` on the dataloader side; instead, the detector internally renormalizes the ImageNet-normalized input to CLIP stats and retains the positional encoding interpolation under the `256` input.

## 2. Code path comparison conclusion

See [`winclip_checklist.md`](winclip_checklist.md) for the control matrix.

### Consistency confirmed

- prompt ensemble template, zero-shot image score, window score, harmonic aggregation are consistent with reference `torch_model.py` / `prompting.py` / `utils.py`
- The few-shot branch still follows the visual association path of the reference implementation and has not deviated from this fix.
- `configs/winclip/winclip_256_mvtec.py` continues to be fixed to `apply_transform=False`, which is in line with the main caliber of the README historical benchmark

### Fixed inconsistencies

- The ImageNet-normalized input output by `NormalizeAD` is now renormalized to CLIP mean/std internally in the detector and is no longer directly fed into the CLIP visual encoder
- WinCLIP now builds and caches class prompts as per `data_samples[*].cls_name` and no longer returns the entire batch of samples to the default `class_name='object'`
- `_cosine_similarity` has completed the `2D image embeddings x 3D class-conditioned text embeddings` path to avoid triggering shape mismatch after the batch category prompt is opened.
- `OpenCLIPBackbone` has supported explicit local checkpoint / cache parameters, WinCLIP main configuration now takes priority in the warehouse `pretrained/open_clip/vit_b_16_plus_240-laion400m_e31-8fb26589.pt`
- The warehouse has added `tools/prepare_openclip_weights.py`, you can prepare local weights from the official `open_clip` release/mirror, and no longer rely on the default HF cache

### Items that are still open

- none

## 3. Behavior Probe

Order:

```bash
python tools/alignment_probe.py configs/winclip/winclip_256_mvtec.py \
    --splits test \
    --max-batch-size 2 \
    --device cpu
```
in conclusion:

- `runs/alignment/winclip_probe.json` passed, the `test` split structure check under the real local weight is all `ok`
- The results prove that the current main configuration `apply_transform=False + 256 input + detector-internal CLIP renormalization + cls_name prompt` can stably produce limited score / map on the real model

Key statistics:

- dataset sample: `bottle/test/broken_large/000.png`, input shape is `2 x 3 x 256 x 256`
- loss path: WinCLIP no training loss, targeted tests have covered zero-loss stub
- predict path: `pred_score` is limited, `pred_anomaly_map` shape is `1 x 256 x 256` and limited; `score mean = 0.5190`, `map mean = 0.3229` in probe

## 4. Small-scale controlled experiment

Experimental setup:

- Category: `bottle`
- Training budget: WinCLIP is zero-shot, no training; single class `test` smoke should be executed
- seed: `42`
- Comparison object: current `256 + PE interpolation + ImageNet-to-CLIP renormalization + cls_name prompt` path

observe:

- Use the same lightweight benchmark path to run `bottle` single-class verification first, and the result is:
  - `image_auroc = 0.9730`
  - `pixel_auroc = 0.6184`
- `image_auroc` is significantly higher than the random baseline, and the image score does not collapse to the same platform
- `pixel_auroc` is consistent in magnitude with the README historical full results, indicating that the current `256 + detector compensation` caliber has not damaged the positioning branch.

determination:

- `pass`
- Reason: The single-category result under the real weight is normal and the shutdown line is not triggered.

## 5. Full Benchmark

Target command:

Order:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/winclip_benchmark.py \
    --config configs/winclip/winclip_256_mvtec.py \
    --data-root data/mvtec_ad \
    --categories all \
    --batch-size 16 \
    --output runs/alignment/winclip_v2.json
```
Summary of results:

| Metric | Reference | BaoIAD | Gap |
|--------|-----------|----------|-----|
| image_auroc | `0.893` | `0.8973` | `+0.4%` |
| pixel_auroc | `—` | `0.6318` | `—` |

illustrate:

- The current 15 categories of real results have been re-run, and the average image AUROC is about `+1.7%` higher than the README historical value `0.8802`
- per-class results are saved in `runs/alignment/winclip_v2.json`, where:
  - The strong category is still `leather / tile / carpet / grid / wood`
  - The weaker category is still `capsule / screw / cable / pill`

Shutdown line inspection:

- [x] No large area image AUROC near `0.5` appears
- [x] No unified platform value collapse occurs
- [x] The real benchmark has been rerun.
- [x] Average image AUROC is within an acceptable range from the reference

## 6. Guard

- New/enhanced test: `tests/test_models/test_detectors/test_winclip.py`
- New/enhanced test: `tests/test_models/test_backbones/test_clip_backbone.py`
- New tools: `tools/prepare_openclip_weights.py`, `tools/winclip_benchmark.py`
- Added new anti-regression points:
  - ImageNet -> CLIP renormalization must be performed when `apply_transform=False`
  - `data_samples[*].cls_name` must take precedence over the default `class_name`
  - `256` input path must adapt `grid_size` from `15x15` to `16x16`
  - class-conditioned text embeddings must support batched `2D x 3D` cosine similarity
  - `OpenCLIPBackbone` must pass explicit `cache_dir / pretrained_*_path / load_weights` correctly to `open_clip`
- If you change these paths later, you must rerun:
  - `baoiad/models/detectors/winclip.py`
  - `baoiad/models/backbones/clip_backbone.py`
  - `configs/winclip/winclip_256_mvtec.py`
  - `tests/test_models/test_detectors/test_winclip.py`
  - `tests/test_models/test_backbones/test_clip_backbone.py`
  - `python tools/alignment_probe.py configs/winclip/winclip_256_mvtec.py --splits test --max-batch-size 2 --device cuda --output runs/alignment/winclip_probe.json`

## 7. Residual Risk

- The few-shot branch does not add independent behavioral tests this time, and still mainly relies on the reference path to remain unchanged.
- The current full results come from the direct evaluation harness, not the train-loop shell of `tools/benchmark.py`; both use the same model, data and metric caliber, but the latter is still slower

## 8. Conclusion

- Final decision: `aligned`
- Allowed to proceed to next stage: `yes`
- If not allowed, next action: None

## Alignment checklist

Status only allowed: `matched | mismatch-fixed | intentional-diff | open`

## 1. Enter path

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| train color channel | `lightning_model.py` | `baoiad/datasets/transforms/loading.py` | image hold `RGB` | `LoadImage(to_rgb=True)` | matched |
| test color channel | `lightning_model.py` | `baoiad/datasets/transforms/loading.py` | image hold `RGB` | `LoadImage(to_rgb=True)` | matched |
| resize/crop | `WinClip.configure_pre_processor()` | `configs/_base_/datasets/mvtec_ad.py` | Reference is `Resize((240, 240))` | BaoIAD preserves unity `ResizeAD(256)`, internally compensates for CLIP normalization and PE via detector | intentional-diff |
| normalization / value range | `WinClip.configure_pre_processor()` | `NormalizeAD` + `WinClipDetector._normalize_for_clip()` | Should be CLIP mean/std before entering CLIP visual encoder | Complemented detector internal ImageNet -> CLIP renormalization | mismatch-fixed |
| Default transform switch | `WinClipModel(scales=scales, apply_transform=False)` | `configs/winclip/winclip_256_mvtec.py` | README main caliber should remain `apply_transform=False` | The current config is consistent with the reference lightning entry | matched |
| Weight loading path | `open_clip` Official pre-training weights | `OpenCLIPBackbone` + `tools/prepare_openclip_weights.py` | The real model weights should be stably loaded from the local path | `pretrained/open_clip/vit_b_16_plus_240-laion400m_e31-8fb26589.pt` has been prepared and passed the real probe | mismatch-fixed |

## 2. Anomaly Synthesis

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| anomaly synthesis | — | — | WinCLIP is a zero-shot method, no synthetic exceptions | Not applicable | intentional-diff |
| clean/anomaly sampling | — | — | Same as above | Not applicable | intentional-diff |
| beta range | — | — | Same as above | Not applicable | intentional-diff |
| Texture blending formula | — | — | Same as above | Not applicable | intentional-diff |

## 3. Reconstruct branch

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| Encoder structure | `torch_model.py:encode_image()` | `WinClipDetector._encode_image()` | CLIP image encoder + masked window reuse | The main path is consistent with the reference | matched |
| Decoder structure | — | — | WinCLIP no reconstruct decoder | Not applicable | intentional-diff |
| Output activation | — | — | Same as above | Not applicable | intentional-diff |
| loss input | — | `forward(mode='loss')` | WinCLIP no training loss | zero loss stub only for runner compatibility | intentional-diff |

## 4. Discriminate branch

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| Splicing order | — | — | WinCLIP no discriminate decoder | Not applicable | intentional-diff |
| Number of output categories | `prompting.py` + `class_scores()` | `create_prompt_ensemble()` + `class_scores()` | normal / anomalous two types of text prompt | currently consistent | matched |
| upsampling path | `torch_model.py:forward()` | `WinClipDetector.forward()` | bilinear interpolation anomaly map to input size | currently consistent | matched |
| skip connection | — | — | WinCLIP no skip structure | not applicable | intentional-diff |

## 5. Loss

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| MSE input form | — | — | WinCLIP None MSE loss | Not applicable | intentional-diff |
| SSIM input form | — | — | WinCLIP None SSIM loss | Not applicable | intentional-diff |
| focal input form | — | — | WinCLIP without focal loss | not applicable | intentional-diff |
| loss weights | — | — | WinCLIP no training weights | Not applicable | intentional-diff |
| reduction | — | `forward(mode='loss')` | Returns only runner compatible placeholder loss | currently consistent | matched |

## 6. Predict / Scoring

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| text prompt conditioning | `lightning_model.py:_get_class_name()` + `WinClipModel.setup()` | `WinClipDetector._resolve_batch_classes()` | prompt must vary with category and cannot be fixed to `object` | built and cached as `data_samples[*].cls_name` text embeddings | mismatch-fixed |
| anomaly map source | `torch_model.py:_compute_zero_shot_scores()` | `WinClipDetector._compute_zero_shot_scores()` | full-image score + multi-scale window score harmonic aggregation | current consensus | matched |
| pooling | `torch_model.py:forward()` | `WinClipDetector.forward()` | image score takes anomalous text class softmax score | currently consistent | matched |
| image score aggregation | `torch_model.py:forward()` | `WinClipDetector.forward()` | zero-shot image score and few-shot score are fused according to the reference method | currently consistent | matched |
| post-processing / smoothing | `torch_model.py:forward()` | `WinClipDetector.forward()` | bilinear upsampling back to input size, no additional smoothing | currently consistent | matched |

## 7. Behavior verification conclusion

- [x] `apply_transform=False`'s ImageNet -> CLIP renormalization has been covered by targeted tests
- [x] `cls_name` driven prompt parsing has been overridden by targeted tests
- [x] predict path's score / map has made shape / finite assertion
- [x] `256` Input position encoding interpolation asserted
- [x] `alignment_probe` passed using real `ViT-B-16-plus-240` weights

## 8. Remarks

- The current master configuration can already reliably load real WinCLIP models from local weights.
- The current 15-class mean is `image_auroc=0.8973 / pixel_auroc=0.6318`, which is within the acceptable range compared to the reference image AUROC `0.893`.
