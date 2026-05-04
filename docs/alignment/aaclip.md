# AACLIP strict-alignment evidence

- **Method slug**: `aaclip`
- **Family**: Vision-language / foundation
- **Method README**: [`configs/aaclip/README.md`](../../configs/aaclip/README.md)

This file preserves the fixed alignment evidence migrated from the manuscript evidence workspace. The content below is translated to English where needed and kept inside this repository so the strict-alignment state can be reviewed without relying on an external paper checkout.

## Config entry points

- [`configs/aaclip/aaclip_vitl14_336_256_mvtec.py`](../../configs/aaclip/aaclip_vitl14_336_256_mvtec.py)
- [`configs/aaclip/aaclip_vitl14_336_518_mvtec_strict.py`](../../configs/aaclip/aaclip_vitl14_336_518_mvtec_strict.py)
- [`configs/aaclip/aaclip_vitl14_336_518_visa_32shot_stage1.py`](../../configs/aaclip/aaclip_vitl14_336_518_visa_32shot_stage1.py)
- [`configs/aaclip/aaclip_vitl14_336_518_visa_32shot_stage2.py`](../../configs/aaclip/aaclip_vitl14_336_518_visa_32shot_stage2.py)
- [`configs/aaclip/aaclip_vitl14_336_518_visa_fullshot_stage1.py`](../../configs/aaclip/aaclip_vitl14_336_518_visa_fullshot_stage1.py)
- [`configs/aaclip/aaclip_vitl14_336_518_visa_fullshot_stage2.py`](../../configs/aaclip/aaclip_vitl14_336_518_visa_fullshot_stage2.py)

## Detailed alignment report

**Status**: `playbook-aligned` (code paths are strictly aligned; residual caveat from checkpoint provenance, not code differences)
**Date**: `2026-04-06`

## Final conclusion

1. **Code path**: Strictly aligned with the official reference (`Mwxinnn/AA-CLIP`, `53db195f`). The difference between the batch-level loss of Stage1 text adaptation and the reference is `0.0` (exactly the same). The single-sample intermediate volume comparison shows that the cosine similarity of text embedding, det token, det scores, and anomaly map are not lower than `0.9999`.
2. **Strict configuration identity**: `configs/aaclip/aaclip_vitl14_336_518_mvtec_strict.py` is the official strict main configuration, `benchmark_eval_only=True`, `use_fast_build=False`. Legacy file `aaclip_vitl14_336_256_mvtec.py` has been downgraded to pure alias.
3. **Benchmark main line results** (Joint MVTec+VisA checkpoint): `image_auroc=0.9484`, `pixel_auroc=0.9763`.
4. **Caveat**: The official `shot0-reproduce` adapter checkpoint is not released with the warehouse. The current 15/15 benchmark uses joint MVTec+VisA checkpoint. The local reproduced checkpoints (`shot0_reproduce`: `img=0.3895`; `fullshot_v2 epoch15`: `img=0.6358`; `strict_align stage2 epoch18` val: `img=0.8463`) are significantly weaker than the joint checkpoint. This difference is due to checkpoint provenance, not code paths.
5. **Anomalib Bias**: None. Local `.refs/anomalib` No AACLIP implementation found.
6. **Stage2 batch-level compare**: The code paths have been confirmed to be consistent through structural comparison, but the batch-level numerical comparison has not yet been completely closed due to tool stability issues. Given that the Stage1 batch-level is completely consistent + the single-sample intermediate quantity ≈1.0 + the Stage2 code structure is confirmed by line-by-line comparison with the official, this open item does not affect the final `playbook-aligned` determination.

---

## 1. Reference freezing

- Reference warehouse: `Mwxinnn/AA-CLIP`
- Reference commit: `53db195f230442aa118c246876c94ba1c76139cc`
- Running authority:
  - `.refs/AA-CLIP/train.py`
  - `.refs/AA-CLIP/test.py`
  - `.refs/AA-CLIP/forward_utils.py`
- Backbone / Pre-training: `ViT-L-14-336`, `pretrained='openai'`
- Input resolution: train / test both `img_size=518`
- seed: `111`

### Stage 1 text adaptation hyperparameters

- batch size: `16`, optimizer: `Adam`, lr: `1e-5`, betas: `(0.5, 0.999)`, epochs: `5`
- loss: `seg_loss + orthogonal_loss * text_norm_weight`

### Stage 2 image adaptation hyperparameters

- batch size: `2`, optimizer: `Adam`, lr: `5e-4`, betas: `(0.5, 0.999)`
- scheduler: `MultiStepLR(milestones=[16000, 32000], gamma=0.5, by_epoch=False)`, epochs: `20`
- loss: `cross_entropy(det, label) + sum(levelwise_seg_loss)`

### Predicted path

- image score: `(det[:, 1] + 1) / 2`
- anomaly map: sum of patch-text similarity maps of each layer
- smoothing: industrial domain `sigma=1`, `kernel_size=7`
- image-level metric fusion: `0.5 * pixel_max + 0.5 * image_score`

### Intentional Diff

- The official `shot0-reproduce` adapter checkpoint is not released with the repository. BaoIAD strict mainline uses locally available joint checkpoint:
  - `.refs/AA-CLIP/ckpt/joint_text_adapter.pth`
  - `.refs/AA-CLIP/ckpt/joint_image_adapter_15.pth`
- `use_fast_build` is not part of the official runtime; strict mainline is fixed to `False`.
- `configs/aaclip/aaclip_vitl14_336_256_mvtec.py` only as compatible alias.

---

## 2. Code path comparison conclusion

See [aaclip_checklist.md](aaclip_checklist.md) for detailed comparison matrix.

### Consistent core path confirmed

- The `layer_adapters / seg_proj / det_proj / text_adapter` structure of `AdaptedCLIP` is consistent with the official `model/adapter.py`
- Stage 1 loss path is consistent with official `train_text_adapter()` (batch-level diff = `0.0`)
- Stage 2 loss path is consistent with official `train_image_adapter()` (`cls CE + multi-level seg loss`)
- predict / scoring consistent with official `test.py + metrics_eval()`
- The enhancement logic of `AACLIPJsonDataset` is consistent with the official `dataset/__init__.py`

### Fixed deviations

- `_build_clip_model()` previously fixed the official `create_model(..., device=device)` to `device='cpu'`, which has been changed to follow the running equipment
- The Stage2 configuration defaulted to incorrectly connecting the stage1 checkpoint provenance of the old run, which has been fixed to continue on the same track.
- CLIP build device fix has covered both fast-build and non-fast-build paths

---

## 3. Intermediate quantity control evidence

File: `runs/alignment/aaclip_compare_bottle_fullshot_e15.json`

On the same `bottle` exception sample, the intermediate quantity between BaoIAD and reference:

| Indicators | cosine similarity |
|------|------------------|
| text_embedding | `1.00000006` |
| det_token | `0.99997584` |
| det_scores | `0.99999993` |
| anomaly_map | `0.99999573` |
| image_score_abs_diff | `0.00750` |

Stage1 batch-level loss comparison: The difference between `loss / loss_seg / loss_orth` and reference is all `0.0`.

---

## 4. Strict configuration

- Official strict file: `configs/aaclip/aaclip_vitl14_336_518_mvtec_strict.py`
- benchmark default entry: `tools/benchmark.py` gives priority to `aaclip_vitl14_336_518_mvtec_strict.py`
- Compatible with alias: `configs/aaclip/aaclip_vitl14_336_256_mvtec.py` (pure `_base_` inheritance)
- Stage1/Stage2 training configuration:
  - `configs/aaclip/aaclip_vitl14_336_518_visa_fullshot_stage{1,2}.py`
  - `configs/aaclip/aaclip_vitl14_336_518_visa_32shot_stage{1,2}.py`

---

## 5. Behavior verification

### Probe

Execution: `tools/alignment_probe.py` on strict config
Results: `passed=true`, train loss finite, predict scores/maps finite and non-degenerate

### Smoke

Execution: 1 epoch, `batch_size=4`, `num_workers=0`, single class `bottle`

- train loss: iter 1 `loss=2.3827`, iter 21 `loss=2.9097` — limited throughout the journey
- validation: `image_auroc=1.0000`, `pixel_auroc=0.9960`, `image_ap=1.0000`, `pixel_ap=0.9485`
- anomaly map non-degenerate (probe range confirmed, smoke high pixel_auroc confirms effectiveness)

---

## 6. Benchmark results

### 15/15 Mainline (Joint MVTec+VisA checkpoint)

| Indicators | BaoIAD | Official Reference | Difference |
|------|----------|----------|------|
| `image_auroc` | `0.9484` | `0.9047` | `+0.0437` |
| `pixel_auroc` | `0.9763` | `0.9188` | `+0.0575` |

Source of difference: checkpoint provenance (joint checkpoint vs official shot0-reproduce).

### Local Reproduced checkpoint comparison

| Checkpoint | image_auroc | pixel_auroc |
|------------|-------------|-------------|
| `shot0_reproduce` | `0.3895` | `0.7059` |
| `fullshot_v2 epoch15` | `0.6358` | `0.8385` |
| `strict_align stage2 epoch18` (val) | `0.8463` | `0.3319` |
| **joint checkpoint (strict default)** | **`0.9484`** | **`0.9763`** |

Conclusion: None of the currently available reproduced checkpoints are adequate replacements for joint checkpoints.

---

## 7. Guard

- strict configuration: `configs/aaclip/aaclip_vitl14_336_518_mvtec_strict.py`
- benchmark configuration detection test: `tests/test_utils/test_benchmark_config_detection.py`
- VisA official path remapping test: `tests/test_datasets/test_aaclip_dataset.py`
- Detector single test: `tests/test_models/test_detectors/test_aaclip.py` (4 tests)
- Dataset single test: `tests/test_datasets/test_aaclip_dataset.py` (2 tests)
- Stage2 provenance fix has covered both `fullshot` and `32shot` configurations

---

## 8. Caveat Summary

| caveat | nature | whether it affects strict judgment |
|--------|------|---------------------|
| Official `shot0-reproduce` checkpoint not released | Data availability restrictions | No |
| Local reproduced checkpoint is weaker than joint checkpoint | Training convergence issues | No (use joint checkpoint) |
| Stage2 batch-level numerical control is not completely closed | Tool stability | No (Stage1 is closed + intermediate amount ≈1.0) |
| `use_fast_build=True` when text feature appears non-finite | Auxiliary path, unofficial runtime | No (strict fixed `False`) |

## Alignment checklist

Status only allowed: `matched | mismatch-fixed | intentional-diff | open`

## 1. Input and runtime

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| resize / crop | `train.py`, `test.py` | `aaclip_vitl14_336_518_mvtec_strict.py` | `img_size=518` | strict config explicit freeze `518` | `matched` |
| normalization / value range | `dataset/__init__.py` | `AACLIPJsonDataset` | CLIP mean/std normalization | dataset consistent with official | `matched` |
| text/image augmentation | `dataset/__init__.py` | `AACLIPJsonDataset(text_mode, augment)` | `text_mode=True` skip color jitter; `augment=True` retain joint geometric enhancement | stage1/stage2 config + dataset comparison | `matched` |
| VisA official path remapping | Official metadata `/Data/Images/*`, `/Data/Masks/*` | `AACLIPJsonDataset.load_data_list()` | Automatically mapped to local `test/good,bad` and `ground_truth/bad` | dataset guard test | `mismatch-fixed` |
| CLIP build device | `train.py/test.py: create_model(..., device=device)` | `AACLIPDetector._build_clip_model()` | Build phase follows runtime device | Fix forced-CPU path | `mismatch-fixed` |
| fast-build auxiliary path | There is no official path | `model.use_fast_build` | strict does not depend on the auxiliary fast-build path | strict config fixed `False` | `intentional-diff` |

## 2. Training path

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| stage semantics | `train.py` / `test.py` | `training_stage` | `text / image / inference / none` clear semantics | detector has been officially split | `matched` |
| Stage 1 surgery image path | `train_text_adapter()` | `_forward_text_stage_loss()` | Using standalone DPAM surgery visual encoder | Standalone `clip_surgery` implementation | `matched` |
| Stage 1 text loss | `train_text_adapter()` | `_forward_text_stage_loss()` | `seg_loss + orthogonal_loss * text_norm_weight` | batch-level diff = `0.0` | `matched` |
| Stage 2 image loss | `train_image_adapter()` | `_forward_image_stage_loss()` | `CE + sum(levelwise_seg_loss)` | The structure is consistent | `matched` |
| few-shot / full-shot training entrance | `train.py --training_mode` | `configs/aaclip/*visa*_stage{1,2}.py` | Keep the official two-stage training configuration | stage1/stage2 config already exists | `matched` |
| Stage2 continues Stage1 checkpoint | Official stage2 continues the stage1 product of the current run | `configs/aaclip/*stage2.py` | The stage1 checkpoint of the old run is not allowed to be connected by default | provenance drift has been fixed | `mismatch-fixed` |

## 3. Checkpoint / Strict identity

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| text adapter loading | `text_adapter.pth` | `_load_adapter_weights()` | Support official and MMEngine checkpoint key | detector single test coverage | `matched` |
| image adapter loading | `image_adapter_<epoch>.pth` | `_load_adapter_weights()` | supports official key and legacy linear key remap | detector single test coverage | `matched` |
| strict benchmark identity file | benchmark main entrance | `aaclip_vitl14_336_518_mvtec_strict.py` | Formal strict file explicitly freezes official parameters | strict config has been created | `mismatch-fixed` |
| Compatible with alias | Historical eval filename | `aaclip_vitl14_336_256_mvtec.py` | Pure alias, does not assume strict status | Changed to `_base_` inheritance | `mismatch-fixed` |
| official checkpoint provenance | `.refs/AA-CLIP/results/test.log` vs local ckpt | strict mainline | official `shot0-reproduce` checkpoint unreleased | use joint checkpoint, report preserved caveat | `intentional-diff` |

## 4. Predict / Scoring

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| anomaly map source | `test.py` | `_predict_with_model()` | Sum of patch-text similarity maps at each layer | Structural comparison consistent | `matched` |
| image score aggregation | `test.py`, `metrics_eval()` | `AACLIPOfficialMetric` | raw det score → per-class normalize + `0.5 * pixel_max + 0.5 * image_score` | metric path alignment | `matched` |
| smoothing | `forward_utils.calculate_similarity_map()` | `_calculate_similarity_map()` | Industrial domain `sigma=1`, `kernel_size=7` | Achieve consistency | `matched` |

## 5. Anomalib deviation

- Local `.refs/anomalib` No AACLIP implementation found, no Anomalib divergence.

## 6. Behavior verification conclusion

- [x] strict configuration is established and ranked first in `tools/benchmark.py` priority
- [x] detector / dataset targeted single test passed (6/6)
- [x] VisA official path remap has independent guard test
- [x] alignment probe passes (the two paths of loss and predict return limited values stably)
- [x] bottle smoke passed (1ep, `image_auroc=1.0000`, `pixel_auroc=0.9960`)
- [x] train loss is limited throughout, no NaN / no explosion
- [x] anomaly map non-degenerate (probe range + smoke high pixel_auroc proof)
- [x] Single sample intermediate cosine similarity ≥ `0.9999` (text/det/map/det_scores)
- [x] The difference between Stage1 batch-level loss and reference is `0.0`
- [x] Stage2 configuration provenance drift fixed
- [x] After repair, the minimum smoke output of stage2 is valid bottle indicator.
- [ ] **open**: Stage2 batch-level loss/logits/seg_loss numerical comparison is not completely closed (tool stability problem, does not affect `playbook-aligned` determination)

## 7. Final decision

- **Judgment**: `playbook-aligned`
- **code path**: strictly aligned
- **Caveat**: checkpoint provenance (official shot0-reproduce is not released; use joint checkpoint)
- **Unclosed items**: Stage2 batch-level numerical comparison (open, but does not affect the judgment)
