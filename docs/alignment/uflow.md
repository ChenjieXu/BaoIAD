# U-Flow strict-alignment evidence

- **Method slug**: `uflow`
- **Family**: Normalizing flow
- **Method README**: [`configs/uflow/README.md`](../../configs/uflow/README.md)

This file preserves the fixed alignment evidence migrated from the manuscript evidence workspace. The content below is translated to English where needed and kept inside this repository so the strict-alignment state can be reviewed without relying on an external paper checkout.

## Config entry points

- [`configs/uflow/uflow_mcait_448_mvtec_strict.py`](../../configs/uflow/uflow_mcait_448_mvtec_strict.py)
- [`configs/uflow/uflow_mcait_448_visa.py`](../../configs/uflow/uflow_mcait_448_visa.py)

## Detailed alignment report

**Status**: `aligned`
**Date**: `2026-03-28`

## 1. Reference freezing

- Reference repository: original official `https://gh-proxy.com/https://github.com/mtailanian/uflow`
- Reference commit: `d6217844836790773f2c4b91ff3046c59b23f027`
- Refer to config/checkpoint:
  - `configs/mvtec_default.yaml`
  - `configs/<category>.yaml` 15 categories YAML
  - `src/model.py`
  - `src/feature_extraction.py`
  - `src/train.py`
  - `src/predict.py`
  - `src/nfa_tree.py`
- Data set/category: `MVTec AD`, single-class training + 15-class standard benchmark
- Input resolution: `448x448`
- seed: The official script is not fixed; BaoIAD strict runtime uses `42` uniformly
- Indicator definition:
  - Main acceptance: `pixel_auroc`, `aupro`
  - Auxiliary record: `image_auroc`
- intentional diff:
  - The benchmark still uses likelihood anomaly map as the unified evaluation input by default.
  - `log(NFA)` is used as strict auxiliary output and targeted diagnostic path, and does not replace the default diagram of general evaluation

## 2. Code path comparison conclusion

See [`uflow_checklist.md`](uflow_checklist.md) for the control matrix.

### Consistency confirmed

- strict main line has been switched from historical `resnet18 + 256` to original official `mcait + 448`
- `CaitFeatureExtractor`, U-shaped flow graph, flow loss, likelihood image/map scoring semantics have been aligned with the original warehouse
- `tools/benchmark.py --methods uflow` The default entry has been switched to strict config
- Subcategory `train batch / lr` has been written as benchmark category overrides according to the original `15` class YAML

### Fixed inconsistencies

- Added strict config `configs/uflow/uflow_mcait_448_mvtec_strict.py`
- Added `UFlowStrictTrainHook`, changed `LinearLR.total_iters` from placeholder value to official `len(train_loader) * epochs`
- Added tree-based `log(NFA)` calculation tool `baoiad/utils/uflow_nfa.py`
- The detector's default `predict` has been changed back to the benchmark-friendly likelihood-only path; `NFA` has been changed to explicitly switch `compute_nfa_in_predict=True` to append output

### Items that are still open

- none

## 3. Behavior Probe

Main probe command:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/alignment_probe.py \
    configs/uflow/uflow_mcait_448_mvtec_strict.py \
    --splits train test \
    --max-batch-size 1 \
    --device cuda \
    --output runs/alignment/uflow_strict_probe.json
```
NFA targeted probe command:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/alignment_probe.py \
    configs/uflow/uflow_mcait_448_mvtec_strict.py \
    --splits test \
    --max-batch-size 1 \
    --device cuda \
    --cfg-options \
        model.compute_nfa_in_predict=True \
        test_dataloader.dataset.cls_names="['bottle']" \
        test_dataloader.dataset.multi_class=False \
        val_dataloader.dataset.cls_names="['bottle']" \
        val_dataloader.dataset.multi_class=False \
    --output runs/alignment/uflow_strict_probe_nfa_bottle.json
```
in conclusion:

- `runs/alignment/uflow_strict_probe.json` passed, all structure checks of train/test are `ok`
- Two real operational risks have been confirmed and fixed in the current round:
  - The first version of strict detector always calculated `NFA tree` in the default `predict`, which has been changed to the likelihood-only default prediction.
  - `timm` will give priority to `hf-hub`. Under `HF_HUB_OFFLINE=1`, it will fail even if the local `.pth` exists; now it has been changed to `CaitFeatureExtractor` to give priority to the local `M48_448.pth / S24_224.pth`.
- There is currently no running pre-blocker; strict `probe` and `NFA` targeted probe have been archived

Key statistics:

- train sample: The actual sample is `zipper/train/good/065.png`, the input shape=`[1,3,448,448]`, the value range is limited
- train loss: `510515.7188`, limited
- test sample: `bottle/test/broken_large/000.png`
- predict path: `pred_score=0.6779`, `pred_anomaly_map` shape=`(1,448,448)`, all finite
- MCait local weights have now fallen to `~/.cache/torch/hub/checkpoints/M48_448.pth` and `S24_224.pth`

## 4. Small-scale controlled experiment

Schedule the smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/benchmark.py \
    --data_root data/mvtec_ad \
    --config configs/uflow/uflow_mcait_448_mvtec_strict.py \
    --methods uflow \
    --categories bottle \
    --epochs 1 \
    --timeout 7200 \
    --output runs/alignment/uflow_bottle_strict_smoke.json
```
Auxiliary low learning rate smoke:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/benchmark.py \
    --data_root data/mvtec_ad \
    --config configs/uflow/uflow_mcait_448_mvtec_strict.py \
    --methods uflow \
    --categories grid \
    --epochs 1 \
    --timeout 7200 \
    --output runs/alignment/uflow_grid_strict_smoke.json
```
Current observations:

- `runs/alignment/uflow_bottle_strict_smoke.json` Archived: `image_auroc=0.8897`, `pixel_auroc=0.9425`, `aupro=0.8396`, `time=273.8s`
- `runs/alignment/uflow_grid_strict_smoke.json` Archived: `image_auroc=0.6642`, `pixel_auroc=0.9060`, `aupro=0.7373`, `time=254.2s`
- `grid` The first time smoke failed on GPU0 due to memory competition, it was successfully completed after migrating to GPU2; this shows that blocker is a runtime resource allocation, not a strict config logic error.
- The direct `bottle` single batch check of `compute_nfa_in_predict=True` has verified that both `pred_nfa_score` and `pred_nfa_anomaly_map` exist and are limited

## 5. Full Benchmark

strict full command:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/benchmark.py \
    --data_root data/mvtec_ad \
    --config configs/uflow/uflow_mcait_448_mvtec_strict.py \
    --methods uflow \
    --categories all \
    --timeout 7200 \
    --output runs/alignment/uflow_strict_full.json
```
Summary of results:

| Metric | Reference | BaoIAD | Gap |
|--------|-----------|----------|-----|
| image_auroc | N/A | **0.9827** | N/A |
| pixel_auroc | `0.9834` | **0.9859** | `+0.25%` |
| aupro | N/A | **0.9447** | N/A |

strict `15/15` results archived:

- merged: `runs/alignment/uflow_strict_full.json`
- shards:
  - `runs/alignment/uflow_strict_full_part1.json`
  - `runs/alignment/uflow_strict_full_part2.json`
  - `runs/alignment/uflow_strict_full_part3.json`
  - `runs/alignment/uflow_strict_full_part4.json`

Category-by-category results:

| Category | Image AUROC | Pixel AUROC | AUPRO |
|----------|-------------|-------------|-------|
| bottle | 0.9929 | 0.9848 | 0.9552 |
| cable | 0.9781 | 0.9859 | 0.9345 |
| capsule | 0.9860 | 0.9890 | 0.9460 |
| carpet | 0.9988 | 0.9938 | 0.9831 |
| grid | 0.9933 | 0.9831 | 0.9388 |
| hazelnut | 0.9979 | 0.9939 | 0.9440 |
| leather | 1.0000 | 0.9951 | 0.9823 |
| metal_nut | 1.0000 | 0.9838 | 0.9304 |
| pill | 0.9763 | 0.9916 | 0.9563 |
| screw | 0.9322 | 0.9947 | 0.9670 |
| tile | 0.9942 | 0.9729 | 0.9177 |
| toothbrush | 0.9139 | 0.9882 | 0.8866 |
| transistor | 0.9983 | 0.9776 | 0.9173 |
| wood | 0.9939 | 0.9670 | 0.9547 |
| zipper | 0.9850 | 0.9869 | 0.9573 |
| **Mean** | **0.9827** | **0.9859** | **0.9447** |

Shutdown line inspection:

- [x] No large area image AUROC near `0.5` appears
- [x] Multiple categories did not collapse to similar platform values.
- [x] score histogram No obvious abnormal shrinkage
- [x] The gap from the reference can still be explained

Current status:

- full benchmark archived
- Marking `UFlow` as `playbook-complete` is currently allowed

## 6. Guard

- New test:
  - `tests/test_models/test_detectors/test_uflow.py`
  - `tests/test_engine/test_uflow_strict_hook.py`
  - `tests/test_utils/test_benchmark_config_detection.py`
- Added probe/assertion:
  - `UFlowStrictTrainHook` dynamic correction `LinearLR.total_iters`
  - benchmark category overrides cover the official `15` class `lr / batch_train`
  - detector default `predict` is no longer forced to calculate `NFA`
  - detector can be appended with `pred_nfa_score / pred_nfa_anomaly_map` when `compute_nfa_in_predict=True`
  - `CaitFeatureExtractor` now preferentially reads `M48_448.pth / S24_224.pth` from the local torch-hub cache
- If you change these paths later, you must rerun:
  -`pytest tests/test_models/test_detectors/test_uflow.py tests/test_engine/test_uflow_strict_hook.py tests/test_utils/test_benchmark_config_detection.py -q -k 'uflow'`
  - strict `alignment_probe`
  - strict `bottle` smoke
  - strict `grid` smoke
  - strict `15/15` full benchmark

## 7. Residual Risk

- `NFA tree` is still a high-cost auxiliary path; if it is accidentally opened in the benchmark default path later, the running time will be significantly worse.
- The current weakest image categories are `toothbrush=0.9139` and `screw=0.9322`, but there is no stop-line level systematic collapse.
- The main number publicly given by the original official warehouse is mainly pixel-side; therefore, the current strict closure should still be based on `pixel_auroc / aupro`, and `image_auroc` is only used as supporting evidence.

## 8. Conclusion

- Final decision: `aligned`
- Allowed to proceed to next stage: `yes`
- Current conclusions:
  - strict `15/15` completed, current `pixel_auroc=0.9859`, relative reference `0.9834` is `+0.25%`
  - playbook stop-line not triggered
  - `UFlow` can be written as `playbook-complete`

## Alignment checklist

Status only allowed: `matched | mismatch-fixed | intentional-diff | open`

## 1. Reference freezing

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| Main reference warehouse | `https://gh-proxy.com/https://github.com/mtailanian/uflow` | `docs/alignment/uflow.md` | strict The main line is based on the original official warehouse, and anomalib is no longer the main reference | Reference frozen written method report | matched |
| Reference commit | `main@d6217844836790773f2c4b91ff3046c59b23f027` | Same as above | Fixed unique code size | Method report frozen | matched |
| Main configuration caliber | `configs/<category>.yaml` | `configs/uflow/uflow_mcait_448_mvtec_strict.py` | Main line fixed `mcait + 448 + 200e + pixel-first` | strict config has been implemented | matched |
| Category hyperparameters | Original `15` class YAML | `benchmark_category_cfg_options` | `train batch / lr` followed category override | `test_uflow_strict_category_cfg_options_follow_official_yaml` | matched |
| Main indicator caliber | Official training script `pixel_auroc / pixel_aupro / image_auroc` | strict report + benchmark selector | benchmark main conclusion is closed according to `pixel_auroc` | strict config `benchmark_result_selector=best(pixel_auroc)` | matched |

## 2. Enter path

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| train color channel | `src/datamodule.py` + `PIL.Image.convert('RGB')` | `LoadImage` | The training input enters the backbone as an RGB tensor | The existing dataset pipeline keeps the default RGB | matched |
| test color channel | Same as above | Same as above | test consistent with train | train/test pipeline symmetric | matched |
| resize | `transforms.Resize(input_size)` | `ResizeAD(size=448)` | strict mainline unified to `448x448` | strict config fixed | matched |
| normalization / value range | `Normalize(mean,std)` | `NormalizeAD()` | ImageNet mean/std normalization | strict config fixed | matched |

## 3. Backbone / Flow

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| MCait extractor | `src/feature_extraction.py::MCaitFeatureExtractor` | `CaitFeatureExtractor` | `cait_m48_448 + cait_s24_224` dual scale, block truncation and reshape semantics are consistent | Code has been compared | matched |
| U-shaped flow diagram | `src/model.py::UFlow` | `UFlowDetector._build_flow()` | split / upsample / concat paths are consistent | structure has been compared | matched |
| coupling subnet | `src/model.py::get_affine_coupling_subnet()` | `AffineCouplingSubnet` | Two-layer convolution + ReLU, alternating `3x3 / 1x1` | Single test + code comparison | matched |
| scheduler budget | `src/train.py::get_total_number_of_iterations()` | `UFlowStrictTrainHook` | `LinearLR.total_iters = len(train_loader) * epochs` | `test_uflow_strict_hook_updates_linear_lr_total_iters` | mismatch-fixed |

## 4. Loss / Predict

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| flow loss | `src/train.py::step()` | `UFlowDetector._compute_loss()` | `mean(sum(0.5*z^2) - ljd)` | detector single test | matched |
| likelihood map | `src/model.py::get_probability()` | `compute_likelihood_anomaly_map()` | `1 - mean(prob_i)` | detector single test | matched |
| image score | `src/train.py::validation_step()` | `predict -> pred_score=max(map)` | image score takes the maximum value of anomaly map | detector single test | matched |
| NFA tree auxiliary path | `src/nfa_tree.py::compute_nfa_anomaly_score_tree()` | `baoiad/utils/uflow_nfa.py` + `compute_nfa_anomaly_map()` | tree-based `-log(NFA)` can be calculated on demand | `test_forward_predict_can_attach_nfa_outputs` | mismatch-fixed |
| benchmark default graph | official default training verification uses likelihood; NFA is used for additional segmentation analysis | strict config + detector `compute_nfa_in_predict=False` | benchmark does not force calculation of NFA by default | strict config has been explicitly frozen | mismatch-fixed |

## 5. Runtime / Benchmark

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| early stopping | `src/train.py::EarlyStopping(monitor='pixel_auroc', patience=20)` | strict config `EarlyStoppingHook` | monitor `ad/pixel_auroc`, patience=20 | strict config + benchmark guard | matched |
| best checkpoint | `ModelCheckpointByAuROC / AuPRO / mIoU` | strict config `CheckpointHook(save_best='ad/pixel_auroc')` | strict main line is subject to best pixel checkpoint | strict config + benchmark guard | mismatch-fixed |
| benchmark default entry | strict mainline | `tools/benchmark.py::find_config('uflow')` | default hit strict `mcait-448` config | `test_uflow_benchmark_prefers_strict_config` | mismatch-fixed |
| MCait weight preparation | Official dependency pre-training CaiT weight | `CaitFeatureExtractor` + local torch-hub cache | strict mainline preferentially uses local `M48_448.pth / S24_224.pth` to avoid `hf-hub offline` blocking | local cache has been prepared; `CaitFeatureExtractor` now takes precedence over `pretrained_cfg_overlay(file=...)` | mismatch-fixed |

## 6. Behavior verification conclusion

- [x] UFlow strict config, benchmark priority, category override, scheduler hook already have guard
- [x] Default `predict` only outputs likelihood anomaly map; NFA is verified by explicit switch single test
- [x] strict `alignment_probe` passed in its entirety on the real `mcait-448` path
- [x] strict `bottle` smoke archived
- [x] strict `grid` smoke archived
- [x] strict `15/15` full benchmark archived

## 7. Remarks

- `UFlow` does not have anomaly synthesis, nor is it a reconstruct/discriminate dual-branch method, so the checklist is cut into four parts: input, backbone/flow, loss/predict, and runtime.
- strict `15/15` full benchmark has been archived in `runs/alignment/uflow_strict_full.json`, and the current mean is `img=0.9827`, `pxl=0.9859`, `aupro=0.9447`.
- There is currently no open item, UFlow can be recorded as `playbook-complete` according to the playbook.
