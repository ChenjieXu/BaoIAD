# PaDiM strict-alignment evidence

- **Method slug**: `padim`
- **Family**: Feature-memory / density
- **Method README**: [`configs/padim/README.md`](../../configs/padim/README.md)

This file preserves the fixed alignment evidence migrated from the manuscript evidence workspace. The content below is translated to English where needed and kept inside this repository so the strict-alignment state can be reviewed without relying on an external paper checkout.

## Config entry points

- [`configs/padim/padim_wrn50_256_mvtec_strict.py`](../../configs/padim/padim_wrn50_256_mvtec_strict.py)
- [`configs/padim/padim_wrn50_256_visa.py`](../../configs/padim/padim_wrn50_256_visa.py)

## Detailed alignment report

**Status**: `aligned`
**Date**: `2026-03-27`

## 1. Reference freezing

- Reference warehouse:
  - Primary reference: local `.refs/anomalib-worktrees/0ef8ab1e`
- Reference commit:
  - `0ef8ab1e`
- Refer to config/checkpoint:
  - `.refs/anomalib-worktrees/0ef8ab1e/anomalib/models/padim/torch_model.py`
  - `.refs/anomalib-worktrees/0ef8ab1e/anomalib/models/padim/anomaly_map.py`
  - `.refs/anomalib-worktrees/0ef8ab1e/anomalib/models/padim/config.yaml`
  - MVTec indicators for WRN-50-2 in `.refs/anomalib-worktrees/0ef8ab1e/docs/source/research/benchmark.md`
- Dataset/Category: `MVTec AD`, standard 15-class single-class training/testing
- Input resolution: `256x256`
- seed: `42`
- Indicator definition:
  - image score: `max(smoothed full-res anomaly_map)`
  - pixel map: Mahalanobis distance -> bilinear upsample -> Gaussian smoothing
- intentional diff:
  - BaoIAD maintains explicit `TIMMBackbone` configuration instead of directly reusing anomalib's Lightning wrapper
  - In order to allow `MemoryBankHook` / `alignment_probe` to follow a unified path, `build_memory_bank(dataloader)` now supports supplementary collection when there is only dataloader; the scoring caliber under the frozen main configuration will not be changed.

## 2. Code path comparison conclusion

See [`padim_checklist.md`](padim_checklist.md) for the control matrix.

### Consistency confirmed

- `configs/padim/padim_wrn50_256_mvtec_strict.py` has been frozen to `wide_resnet50_2 + out_indices=(1,2,3) + sigma=4.0 + max_epochs=1`
- `extract_features()` is consistent with the reference implementation, still press `layer1/layer2/layer3 -> nearest upsample -> concat -> random feature subset`
- `predict` paths still generate pixel / image scores as in `Mahalanobis -> bilinear upsample -> Gaussian blur -> max(map)`

### Fixed inconsistencies

- `PaDiMDetector.build_memory_bank()` now supports `dataloader=None | train_loader` two entrances, probe/hook no longer need to rely on external scripts for manual warmup
- `PaDiMDetector.forward(mode='predict')` now explicitly reports an error when Gaussian stats is not completed; the probe will capture this precondition first, then automatically warmup and retry
- `tests/test_models/test_detectors/test_padim.py` complemented `predict precondition` and `build_memory_bank(dataloader)` guard
- `tests/test_utils/test_benchmark_config_detection.py` Fixed WRN-50 main configuration freeze guard

### Items that are still open

- No algorithm level `open` items

## 3. Behavior Probe

Order:

```bash
python tools/alignment_probe.py configs/padim/padim_wrn50_256_mvtec_strict.py \
    --splits train test \
    --device cpu \
    --max-batch-size 2 \
    --cfg-options model.backbone.pretrained=False \
    --output runs/alignment/padim_probe.json
```
in conclusion:

- `runs/alignment/padim_probe.json` passed
- The batch, loss, and predict structures of train/test are all normal.
- The first time predict in test triggers `PaDiM Gaussian statistics are not built`, then the probe automatically warms up and completes the retry, which proves that the current precondition guard and warmup paths have been opened.

Key statistics:

- dataset sample:
  - train sample: `zipper/train/good/065.png`
  - test sample: `bottle/test/broken_large/000.png`
  - Input shape: `2 x 3 x 256 x 256`
- loss path:
  - `keys=['loss']`
  - `loss=0.0`
  -finite
- predict path:
  - `memory_bank_warmup.used = true`
  - `memory_bank_warmup.num_batches = 1`
  - `pred_score mean = 42.2859`
  - `pred_anomaly_map shape = (1, 256, 256)`
  - `pred_anomaly_map` finite

## 4. Small-scale controlled experiment

Experimental setup:

- Category: `bottle`
- Training budget: `1 epoch`
- seed: `42`
- Comparison object: current strict main configuration `configs/padim/padim_wrn50_256_mvtec_strict.py`

Order:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/benchmark.py \
    --data_root data/mvtec_ad \
    --methods padim \
    --categories bottle \
    --device cuda \
    --epochs 1 \
    --timeout 3600 \
    --output runs/alignment/padim_bottle_smoke.json
```
observe:

- `bottle` smoke ends normally and takes `66.8s`
- The indicators are:
  - `image_auroc = 1.0000`
  - `pixel_auroc = 0.9862`
  - `image_f1max = 1.0000`
  - `aupro = 0.9615`
- There is no score collapse, pixel map full brightness or shutdown line close to `0.5`.

Supplementary single category rerun:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/benchmark.py \
    --data_root data/mvtec_ad \
    --methods padim \
    --categories cable \
    --device cuda \
    --timeout 3600 \
    --output runs/alignment/padim_cable_smoke.json
```
- `cable` The single-category rerun also ends normally and takes `134s`
- Indicators are `image_auroc = 0.9089`, `pixel_auroc = 0.9741`, `image_f1max = 0.8821`

determination:

- `pass`
- Reason: Both smoke and supplementary single-category reruns can stably produce reasonable results, and there is no algorithm-level stop-line signal.

## 5. Full Benchmark

The fresh `15/15` benchmark is completed in a sharding manner, and the products used are:

- `runs/alignment/padim_seq3.json`
- `runs/alignment/padim_split_a.json`
- `runs/alignment/padim_split_b.json`
- `runs/alignment/padim_screw_retry.json`
- Merged result: `runs/alignment/padim_full_merged.json`

Sharding command:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/benchmark.py --data_root data/mvtec_ad --methods padim --categories bottle cable zipper --device cuda --timeout 3600 --output runs/alignment/padim_seq3.json
CUDA_VISIBLE_DEVICES=1 python tools/benchmark.py --data_root data/mvtec_ad --methods padim --categories capsule carpet grid hazelnut leather metal_nut --device cuda --timeout 3600 --output runs/alignment/padim_split_a.json
CUDA_VISIBLE_DEVICES=2 python tools/benchmark.py --data_root data/mvtec_ad --methods padim --categories pill screw tile toothbrush transistor wood --device cuda --timeout 3600 --output runs/alignment/padim_split_b.json
CUDA_VISIBLE_DEVICES=2 python tools/benchmark.py --data_root data/mvtec_ad --methods padim --categories screw --device cuda --timeout 3600 --output runs/alignment/padim_screw_retry.json
```
Summary of results:

| Metric | Reference | Fresh BaoIAD | Gap |
|--------|-----------|----------------|-----|
| image_auroc | `0.9500` | `0.9589` | `+0.0089` |
| pixel_auroc | `0.9790` | `0.9773` | `-0.0017` |
| image_f1max | `—` | `0.9594` | `—` |

Key categories:

- `bottle`: `img=1.0000`, `pxl=0.9862`
- `cable`: `img=0.9089`, `pxl=0.9741`
- `capsule`: `img=0.9266`, `pxl=0.9881`
- `screw`: `img=0.8526`, `pxl=0.9885`
- `zipper`: `img=0.9070`, `pxl=0.9870`

illustrate:

- The intermittent subprocess/import failure that occurred in the single-process `all` run earlier has not been reproduced as an algorithm-level problem in the current stable working tree.
- The first time split_b encountered on `screw` was a memory contention failure, not numerical collapse; subsequent single-category reruns produced normal results.

Shutdown line inspection:

- [x] No large area image AUROC near `0.5` appears
- [x] Multiple categories did not collapse to similar platform values.
- [x] score / pixel The result does not show abnormal collapse
- [x] gap from frozen reference still within acceptable range

## 6. Guard

- New/enhanced tests:
  - `tests/test_models/test_detectors/test_padim.py`
  - `tests/test_utils/test_benchmark_config_detection.py`
  - `tests/test_utils/test_alignment_probe.py`
- New/enhanced tool chain:
  - `tools/alignment_probe.py`
  - `baoiad/utils/alignment_probe.py`
- Added new anti-regression points:
  - Gaussian stats must exist before `predict`
  - `build_memory_bank(dataloader)` must be able to complete statistics from the train loader of probe / hook
  - WRN-50 master configuration must remain frozen to `TIMMBackbone + out_indices=(1,2,3) + sigma=4.0 + batch_size=32 + max_epochs=1`
- If you change these paths later, you must rerun:
  - `baoiad/models/detectors/padim.py`
  - `configs/padim/padim_wrn50_256_mvtec_strict.py`
  -`python tools/alignment_probe.py configs/padim/padim_wrn50_256_mvtec_strict.py --splits train test --device cpu --max-batch-size 2 --cfg-options model.backbone.pretrained=False --output runs/alignment/padim_probe.json`
  - `CUDA_VISIBLE_DEVICES=0 python tools/benchmark.py --data_root data/mvtec_ad --methods padim --categories bottle --device cuda --epochs 1 --timeout 3600 --output runs/alignment/padim_bottle_smoke.json`

## 7. Residual Risk

- This time fresh `15/15` is segmented and merged by category rather than running through a single command at once; but all categories use the same main configuration and the same frozen reference caliber
- `screw` requires a separate retry to complete, indicating that resource competition may still be encountered in a shared GPU environment; this is a running environment risk and is not the current PaDiM caliber blocker

## 8. Conclusion

- Final decision: `aligned`
- Allowed to proceed to next stage: `allowed`
- If the follow-up continues, the next action will be:
  - Synchronize the current results of PaDiM in the paper side or summary table to the fresh `15/15` merged caliber

## Alignment checklist

Status only allowed: `matched | mismatch-fixed | intentional-diff | open`

## 1. Enter path

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| train color channel | `.refs/anomalib-worktrees/0ef8ab1e/anomalib/models/padim/config.yaml` MVTec train loader | `configs/_base_/datasets/mvtec_ad.py` | train input into backbone in RGB | `runs/alignment/padim_probe.json` train `inputs.shape=[2,3,256,256]` | `matched` |
| test color channel | Same as above | Same as above | test is consistent with train | `runs/alignment/padim_probe.json` test `inputs.shape=[2,3,256,256]` | `matched` |
| resize/crop | anomalib `image_size=256`, no extra crop | `ResizeAD(size=256)` | input unified to `256x256` | `configs/_base_/datasets/mvtec_ad.py` | `matched` |
| normalization / value range | ImageNet pre-training backbone caliber | `NormalizeAD` | The input backbone is a finite floating point tensor | `runs/alignment/padim_probe.json` train/test `inputs.finite=true` | `matched` |
| DTD / external texture source | PaDiM does not use anomaly synthesis | None | No additional texture paths should be introduced | PaDiM is a training-free memory-bank method | `intentional-diff` |

## 2. Backbone / Embedding

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| Backbone and layer selection | `.refs/anomalib-worktrees/0ef8ab1e/anomalib/models/padim/torch_model.py` `layers=['layer1','layer2','layer3']` | `configs/padim/padim_wrn50_256_mvtec_strict.py` + `baoiad/models/detectors/padim.py` | WRN-50-2 + `out_indices=(1,2,3)`, corresponding to three layers of features | `tests/test_utils/test_benchmark_config_detection.py` | `matched` |
| Multi-scale spatial alignment | Same as above `F.interpolate(..., mode='nearest')` | `extract_features()` | The last two layers are aligned to the spatial resolution of the first layer and spliced | `baoiad/models/detectors/padim.py` | `matched` |
| Random dimensionality reduction | Same as above `sample(range(...), n_features)` | `idx` buffer | WRN-50-2 uses `550` dimension random subset, fixed seed=42 | Code freeze + probe / benchmark configuration uses `seed=42` | `matched` |

## 3. Gaussian Fit / Memory Bank

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| train feature collection | `.refs/anomalib-worktrees/0ef8ab1e/anomalib/models/padim/lightning_model.py` `self.embeddings.append(...)` | `forward(mode='loss')` | Only normal samples are cached in the training phase embedding | `runs/alignment/padim_probe.json` train loss path | `matched` |
| Gaussian statistical fitting | `.refs/anomalib-worktrees/0ef8ab1e/anomalib/models/padim/torch_model.py` | `fit()` | Calculate mean / covariance / inverse covariance by spatial location | `baoiad/models/detectors/padim.py` | `matched` |
| `build_memory_bank(dataloader)` | The reference implementation only relies on cached embedding; probe requires explicit warmup entry | `build_memory_bank(dataloader=None)` | Allow probe / hook to complete collection and complete fitting when there is only dataloader | `tests/test_models/test_detectors/test_padim.py` | `mismatch-fixed` |
| predict preconditions | Fit Gaussian must be used before reference implementation verification | `forward(mode='predict')` | An error must be reported explicitly when the fitting is not completed, instead of silently returning false results | `tests/test_models/test_detectors/test_padim.py` + `runs/alignment/padim_probe.json` | `mismatch-fixed` |

## 4. Predict / Scoring

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| anomaly map source | `.refs/anomalib-worktrees/0ef8ab1e/anomalib/models/padim/anomaly_map.py` `compute_distance()` | `forward(mode='predict')` | Generate patch score map based on Mahalanobis distance | `baoiad/models/detectors/padim.py` | `matched` |
| Upsampling | Same as above `F.interpolate(..., mode='bilinear', align_corners=False)` | `forward(mode='predict')` | Score map upsampled to input resolution | `runs/alignment/padim_probe.json` `map_shapes=[1,256,256]` | `matched` |
| smoothing | Same as above `sigma=4` | `_GaussianBlur2d(sigma=4)` | full-res anomaly map for Gaussian smoothing | `configs/padim/padim_wrn50_256_mvtec_strict.py` | `matched` |
| image score aggregation | history README and anomalib benchmark caliber | `score_map.view(B,-1).max(...)` | image score = `max(postprocessed anomaly_map)` | `runs/alignment/padim_probe.json` predict path | `matched` |

## 5. Behavior verification conclusion

- [x] After fixing the seed, the dataset sample structure is as expected
- [x] `loss` path output as finite scalar
- [x] `predict` path’s `pred_score / pred_anomaly_map` both exist and are limited
- [x] `bottle` smoke is not triggered image/pixel stop-line
- [x] fresh `15/15` full benchmark completed; see `runs/alignment/padim_full_merged.json` for merged results
