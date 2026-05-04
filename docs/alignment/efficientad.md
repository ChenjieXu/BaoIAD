# EfficientAD strict-alignment evidence

- **Method slug**: `efficientad`
- **Family**: Knowledge distillation
- **Method README**: [`configs/efficientad/README.md`](../../configs/efficientad/README.md)

This file preserves the fixed alignment evidence migrated from the manuscript evidence workspace. The content below is translated to English where needed and kept inside this repository so the strict-alignment state can be reviewed without relying on an external paper checkout.

## Config entry points

- [`configs/efficientad/efficientad_256_mvtec_strict.py`](../../configs/efficientad/efficientad_256_mvtec_strict.py)
- [`configs/efficientad/efficientad_256_visa.py`](../../configs/efficientad/efficientad_256_visa.py)

## Detailed alignment report

**Status**: `playbook-complete`
**Date**: `2026-03-29`

## 1. Reference freezing

- Reference repository: local `.refs/anomalib`
- Reference commit: `4f6af1acb0ee7b81f54cda036dd9f1c27f63b69a`
- Refer to config/checkpoint:
  - `.refs/anomalib/examples/configs/model/efficient_ad.yaml`
  - `.refs/anomalib/src/anomalib/models/image/efficient_ad/lightning_model.py`
  - `.refs/anomalib/src/anomalib/models/image/efficient_ad/torch_model.py`
- Dataset/Class: MVTec AD, standard single-class training/testing protocol
- Input resolution: `256x256`
- seed: `42`
- Indicator definition: image AUROC / pixel AUROC
- intentional diff:
  - BaoIAD still uses the benchmark entry of `val_dataloader = test_dataloader` and does not reproduce the independent val/test division of anomalib datamodule.
  - strict alignment is currently done through BaoIAD's own `tools/train.py` / `tools/benchmark.py`, and does not fork the full runtime peripheral of anomalib Lightning trainer

## 2. Code path comparison conclusion

See [`efficientad_checklist.md`](efficientad_checklist.md) for the control matrix.

### Consistency confirmed

- strict main config `configs/efficientad/efficientad_256_mvtec_strict.py` frozen to anomalib `EfficientAd-S` mainline: `padding=False`, `pad_maps=True`, `batch_size=1`, `max_iters=70000`
- Data preprocessing maintains `[0,1]` input and does not perform external ImageNet Normalize; PDN/AE performs ImageNet normalization internally.
- `hard example mining`, `ImageNette penalty`, AE augmentation, `0.5 * st_map + 0.5 * ae_map`, and quantile normalization are all consistent with the anomalib code path

### Fixed inconsistencies

- `EfficientADDetector` default `padding` changed from `True` to `False`, consistent with anomalib main caliber
- `conv5/conv6` channel path of `PDNMedium` changed to `out_channels -> out_channels`, no longer hard-coded `384`
- `tools/benchmark.py` now defaults to strict configuration `efficientad_256_mvtec_strict.py`, old `efficientad_wrn50_256_mvtec.py` downgraded to legacy padded variant
- `MemoryBankHook.before_test_epoch()` now also calls `compute_normalization_stats()` to avoid missing quantile calibration in test-only / eval-only paths
- `baoiad.utils.alignment_probe` / `tools/alignment_probe.py` are compensated and executed explicitly before `test/val` probe `compute_normalization_stats()`
- official teacher weight has been successfully implemented to `pre_trained/efficientad_pretrained_weights/pretrained_teacher_{small,medium}.pth`
- The teacher zip download link has been patched with "bad zip cleaning + fallback", so it will no longer be permanently contaminated by half archives

### Items that are still open

- No new blocker; full benchmark has been completed, currently only maintenance risks remain

## 3. Behavior Probe

Two probes have been completed:

- structural fallback probe:

```bash
python tools/alignment_probe.py configs/efficientad/efficientad_256_mvtec_strict.py \
    --splits train test \
    --max-batch-size 2 \
    --device cuda \
    --output runs/alignment/efficientad_probe_bottle_structural.json \
    --cfg-options \
        model.teacher_pretrained='' \
        "train_dataloader.dataset.cls_names=['bottle']" \
        train_dataloader.dataset.multi_class=False \
        "test_dataloader.dataset.cls_names=['bottle']" \
        test_dataloader.dataset.multi_class=False \
        "val_dataloader.dataset.cls_names=['bottle']" \
        val_dataloader.dataset.multi_class=False
```

- official teacher probe:

```bash
python tools/alignment_probe.py configs/efficientad/efficientad_256_mvtec_strict.py \
    --splits train test \
    --max-batch-size 2 \
    --device cuda \
    --output runs/alignment/efficientad_probe_bottle_official.json \
    --cfg-options \
        "train_dataloader.dataset.cls_names=['bottle']" \
        train_dataloader.dataset.multi_class=False \
        "test_dataloader.dataset.cls_names=['bottle']" \
        test_dataloader.dataset.multi_class=False \
        "val_dataloader.dataset.cls_names=['bottle']" \
        val_dataloader.dataset.multi_class=False
```
in conclusion:

- `runs/alignment/efficientad_probe_bottle_structural.json` passed, proving that the structural link can run through strict config when official teacher is disabled
- `runs/alignment/efficientad_probe_bottle_official.json` has passed, and the teacher download failure warning no longer appears, indicating that the current strict config can be completely passed under the official teacher condition `pre_train_setup -> loss -> quantile normalization -> predict`

Key statistics:

- structural probe:
  - `loss = 20.5926`
  - `pred_score = 0.0932`
- official probe:
  - `loss = 19.6112`
  - `pred_score = 0.0599`
  - `pred_anomaly_map mean = -0.1960`
  - `normalization_warmup.used = true`

## 4. Small-scale controlled experiment

Two smokes have been completed:

- structural smoke:

```bash
python tools/train.py configs/efficientad/efficientad_256_mvtec_strict.py \
    --work-dir runs/alignment/efficientad_bottle_smoke_structural \
    --cfg-options \
        model.teacher_pretrained='' \
        train_cfg.max_iters=1 \
        train_cfg.val_interval=1 \
        "train_dataloader.dataset.cls_names=['bottle']" \
        train_dataloader.dataset.multi_class=False \
        "test_dataloader.dataset.cls_names=['bottle']" \
        test_dataloader.dataset.multi_class=False \
        "val_dataloader.dataset.cls_names=['bottle']" \
        val_dataloader.dataset.multi_class=False
```

- official teacher smoke:

```bash
python tools/train.py configs/efficientad/efficientad_256_mvtec_strict.py \
    --work-dir runs/alignment/efficientad_bottle_smoke_official \
    --cfg-options \
        train_cfg.max_iters=100 \
        train_cfg.val_interval=100 \
        "train_dataloader.dataset.cls_names=['bottle']" \
        train_dataloader.dataset.multi_class=False \
        "test_dataloader.dataset.cls_names=['bottle']" \
        test_dataloader.dataset.multi_class=False \
        "val_dataloader.dataset.cls_names=['bottle']" \
        val_dataloader.dataset.multi_class=False
```
observe:

-structural smoke:
  - `MemoryBankHook.before_train -> pre_train_setup()` completed normally
  - `before_val_epoch -> compute_normalization_stats()` completed normally
  - `1 iter` smoke can complete verification and checkpoint saving
  - `image_auroc=0.2238`, `pixel_auroc=0.4181` are only used for structural verification, not for performance judgment
- official teacher smoke:
  - After completing `pre_train_setup()`, enter `100 iter` training, and the loss will drop from `12.4036` to `10.1190`
  - quantile calibration is completed before verification
  - Final `bottle image_auroc=0.9444`, `pixel_auroc=0.6992`
  - `image_ap=0.9830`, `aupro=0.4046`

determination:

- `pass`
- Reason: The training/verification loop, hook, predict, and metric summary can all be completed under the official teacher condition; but the single-category `100 iter` smoke still cannot replace the `15/15` full benchmark

## 5. Full Benchmark

**Completed** (2026-03-29)

Result file: `runs/alignment/efficientad_strict_v1.json`

| Metric | BaoIAD | Official | Gap |
|--------|----------|----------|-----|
| image_auroc | **0.9786** | 0.979 | -0.0004 |
| pixel_auroc | **0.9585** | 0.955 | +0.0035 |

### Detailed results for each category

| category | image_auroc | pixel_auroc |
|------|-------------|-------------|
| bottle | 1.0000 | 0.9838 |
| cable | 0.9235 | 0.9819 |
| capsule | 0.9430 | 0.9801 |
| carpet | 0.9896 | 0.9510 |
| grid | 1.0000 | 0.9370 |
| hazelnut | 0.9450 | 0.9638 |
| leather | 0.9997 | 0.9762 |
| metal_nut | 0.9839 | 0.9749 |
| pill | 0.9858 | 0.9865 |
| screw | 0.9772 | 0.9843 |
| tile | 1.0000 | 0.9054 |
| toothbrush | 0.9778 | 0.9651 |
| transistor | 0.9671 | 0.9514 |
| wood | 0.9904 | 0.8725 |
| zipper | 0.9958 | 0.9633 |

Shutdown line inspection:

- [x] Neither structural nor official `bottle` probe / smoke appears. NaN / inf / missing map / hook leaks.
- [x] official teacher weight has been successfully implemented to `pre_trained/efficientad_pretrained_weights/*.pth`
- [x] official-pretrained `15/15` benchmark completed, the difference is within ±0.01

## 6. Guard

- New/enhanced tests:
  - `tests/test_models/test_detectors/test_efficientad.py`
  - `tests/test_utils/test_alignment_probe.py`
  - `tests/test_utils/test_benchmark_config_detection.py`
- New/fixed guard:
  - strict config `configs/efficientad/efficientad_256_mvtec_strict.py`
  - `tools/alignment_probe.py`
  - `MemoryBankHook.before_test_epoch()` quantile warmup
  - invalid zip clean + fallback download
- If you change these paths later, you must rerun:
  - `python tools/alignment_probe.py configs/efficientad/efficientad_256_mvtec_strict.py ...`
  - `python tools/train.py configs/efficientad/efficientad_256_mvtec_strict.py ...`
  - `pytest tests/test_models/test_detectors/test_efficientad.py tests/test_utils/test_alignment_probe.py -q -k 'efficientad or alignment_probe'`

## 7. Residual Risk

- Although the strict configuration has been switched to the anomalib mainline, the benchmark still uses BaoIAD's `val=test` evaluation habit and is not an independent val/test protocol of anomalib datamodule.
- `image_ece` / `pixel_ece` will still report warnings on non-probability scores; this is not the current blocker, but it shows that the probability assumption of the metric on the original anomaly score still exists

## 8. Conclusion

- Final decision: `playbook-complete`
- Alignment results: image_auroc=0.9786 (official 0.979, difference -0.0004), pixel_auroc=0.9585 (official 0.955, difference +0.0035)
- The differences are all within ±0.01 and are considered aligned.

## Alignment checklist

Status only allowed: `matched | mismatch-fixed | intentional-diff | open`

## 1. Enter path

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| train color channel | anomalib `EfficientAd.configure_pre_processor()` + dataset RGB path | `LoadImage -> PackADInputs` | RGB image into PDN/AE | `efficientad_probe_bottle_official.json` train sample `3x256x256` | matched |
| test color channel | Same as above | Same as above | test and train keep the same RGB order | `efficientad_probe_bottle_official.json` test sample `3x256x256` | matched |
| resize | anomalib `Resize((256,256))` | `ResizeAD(size=256)` | unify input to `256x256` | strict config + probe sample shape | matched |
| normalization / value range | forward does ImageNet norm internally, but not externally | `NormalizeAD(mean=0,std=255)` | pipeline outputs `[0,1]` tensor | official probe inputs `min/max≈0.11/1.0` | matched |

## 2. Teacher / Student / AE structure

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| PDN-S default padding | anomalib `padding: false` | `EfficientADDetector(..., padding=False)` + strict config | strict mainline uses `padding=False` | strict config + detector default | mismatch-fixed |
| PDN-M output channels | anomalib `conv5/conv6 = out_channels` | `PDNMedium` | medium variant should not be hardcoded `384` | detector patch + unit tests | mismatch-fixed |
| student output dimension | anomalib `teacher_out_channels * 2` | `self.student = PDN(..., 2 * pdn_channels)` | student first half pair of teacher, second half pair of AE | detector code | matched |
| AE input and structure | anomalib `AutoEncoder(raw image)` | `AutoEncoder` | AE input raw image, and then do ImageNet norm internally | detector code | matched |

## 3. Loss

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| hard example mining | `torch.quantile(distance_st, 0.999)` | `forward(mode='loss')` | ST loss takes top 0.1% hard region | detector code | matched |
| penalty loss | student on ImageNette batch | `_get_imagenet_batch()` + penalty term | penalty loss consistent with student ST branch | detector code + official smoke | matched |
| AE / STAE loss | teacher-aug / ae-aug / student-aug | `loss_ae + loss_stae` | Both AE and student-AE paths are retained | detector code | matched |

## 4. Predict / Scoring

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| quantile normalization | anomalib `qa/qb` on validation good samples | `compute_normalization_stats()` | calibrate before eval `st_map` / `ae_map` | official probe `normalization_warmup.used=true` | mismatch-fixed |
| anomaly map fusion | `0.5 * map_st + 0.5 * map_stae` | `score_map = 0.5 * map_st + 0.5 * map_ae` | equal weight fusion of two maps | detector code | matched |
| image score aggregation | `amax(anomaly_map)` | `torch.amax(score_map, dim=(-2,-1))` | image score takes the maximum value of the final map | detector code + official probe | matched |
| test-only normalization hook | anomalib `on_validation_start` calculates quantiles first | `MemoryBankHook.before_test_epoch()` | test-only entry must also have quantiles first | hook patch | mismatch-fixed |

## 5. Assets/Toolchain

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| teacher weight download | anomalib release asset | `_download_pretrained_weights()` | bad zip cannot permanently contaminate subsequent downloads | fallback patch + local `.pth` implementation | mismatch-fixed |
| lightweight probe | playbook Gate 2 | `tools/alignment_probe.py` | strict config should be able to pass train/test structural check | structural + official probe JSON | mismatch-fixed |
| benchmark default entry | strict config first | `tools/benchmark.py` priority | benchmark default hit strict config | benchmark config test | mismatch-fixed |

## 6. Protocol differences

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| val / test split | anomalib `val_split_mode='from_test'` | `val_dataloader = test_dataloader` | The current benchmark still uses the BaoIAD unified entrance | method report intentional diff | intentional-diff |

## 7. Behavior verification conclusion

- [x] After fixing the seed, the dataset sample structure is as expected
- [x] The key output of `loss` path is a finite value
- [x] `predict` path's score / map is a finite value
- [x] `compute_normalization_stats()` has entered probe and test-only hook
- [x] official teacher weight has been successfully implemented to `pre_trained/efficientad_pretrained_weights/*.pth`
- [x] official `bottle` probe passed
- [x] official `bottle` `100 iter` smoke has been transferred to `image_auroc=0.9444`
- [ ] official-pretrained `15/15` full benchmark is still unfinished
