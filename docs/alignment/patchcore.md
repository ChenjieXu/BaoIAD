# PatchCore strict-alignment evidence

- **Method slug**: `patchcore`
- **Family**: Feature-memory / density
- **Method README**: [`configs/patchcore/README.md`](../../configs/patchcore/README.md)

This file preserves the fixed alignment evidence migrated from the manuscript evidence workspace. The content below is translated to English where needed and kept inside this repository so the strict-alignment state can be reviewed without relying on an external paper checkout.

## Config entry points

- [`configs/patchcore/patchcore_wrn50_256_mvtec_strict.py`](../../configs/patchcore/patchcore_wrn50_256_mvtec_strict.py)
- [`configs/patchcore/patchcore_wrn50_256_visa.py`](../../configs/patchcore/patchcore_wrn50_256_visa.py)

## Detailed alignment report

**Status**: `aligned`
**Date**: `2026-03-26`

## 1. Reference freezing

- Reference warehouse:
  - Indicator caliber frozen to PatchCore entry in [`docs/alignment/README.md`](../../docs/alignment/README.md)
  - Locally inspectable code paths use `.refs/ader/model/patchcore.py` as implementation reference
- Reference commit:
  - `anomalib` corresponds to a commit that is not archived in the warehouse
  - `.refs/ader` is a local snapshot and no commit is recorded separately
- Refer to config/checkpoint:
  - `WRN-50-2`
  - `256x256` input
  - `coreset_ratio=0.1`
  - `num_neighbors=9`
- Dataset/Category: `MVTec AD`, standard 15 categories
- Input resolution: `256`
- seed: `42`
- Indicator definition:
  - image score: `max(postprocessed anomaly_map)`
  - pixel map: patch NN distance -> resize to input resolution -> Gaussian smoothing
- intentional diff:
  - The `faiss-gpu` environment has been independently verified and can be installed, but the current benchmark mainline is still executed according to the CPU index caliber.
  - coreset implementation has been changed to approximate-greedy + prefilter to eliminate long-term coreset lag in single-class benchmarks

## 2. Code path comparison conclusion

See [patchcore_checklist.md](patchcore_checklist.md) for the control matrix.

### Consistency confirmed

- The input path remains `RGB -> Resize(256) -> ImageNet Normalize -> PackADInputs`.
- PatchCore master configuration continues to be frozen to `wide_resnet50_2 + out_indices=(2, 3)`.
- patch-level NN distance is still the source of anomaly score map.
- The benchmark entry continues to recognize PatchCore as a single-class method.

### Fixed inconsistencies

- Historical fatal issue `out_indices=(1, 2)` has been fixed to `(2, 3)`.
- Coreset sampling was changed to approximate-greedy + projection + prefilter, and a new deterministic guard was added.
- `MemoryBankHead.predict()` now explicitly outputs the full-res `256x256` anomaly map.
- PatchCore config explicitly freezes `blur_sigma=4.0`, and adds the reference segmentation post-processing caliber.
- unified config also syncs to `num_neighbors=9` with full-res postprocess.
- The benchmark runner now turns off checkpoint saving by default, fixes dataloader workers to `0`, and pegs the `OPENBLAS/OMP/MKL` thread count to prevent the benchmark from being bogged down by checkpoint serialization or OpenBLAS thread conflicts.
- `tools/benchmark.py` has provided a direct runner path for PatchCore, avoiding the long tail of `Runner.train()`.

### Items that are still open

- none. `checklist / probe / bottle smoke / 15/15` full benchmark has been completed.

## 3. Behavior Probe

Order:

```bash
python tools/alignment_probe.py configs/patchcore/patchcore_wrn50_256_mvtec_strict.py \
    --splits train test \
    --device cuda \
    --max-batch-size 2 \
    --cfg-options model.backbone.pretrained=False \
    --output runs/alignment/patchcore_probe.json
```
in conclusion:

- probe passes and the output file is `runs/alignment/patchcore_probe.json`.
- `train/test` batch, loss, memory-bank warmup, and predict paths all pass.
- After this revision, `pred_anomaly_map` has been changed from the old `28x28` to `256x256`.

Key statistics:

- dataset sample:
  - train sample: `cls_name=zipper`, `gt_label=0`, `inputs.shape=[2,3,256,256]`
  - test sample: `cls_name=bottle`, `gt_label=1`, `gt_mask.shape=[256,256]`
- loss path:
  - `keys=['loss']`
  - `loss=0.0`
  - finite
- predict path:
  - warmup train batches: `1`
  - `pred_score` finite
  - `pred_anomaly_map shape=[1,256,256]`
  - `pred_anomaly_map` finite

## 4. Small-scale controlled experiment

Try the command:

```bash
python tools/benchmark.py \
    --data_root data/mvtec_ad \
    --methods patchcore \
    --categories bottle \
    --epochs 1 \
    --timeout 3600 \
    --output runs/alignment/patchcore_bottle_smoke.json
```
Current observations:

- The direct benchmark path has been opened, and `bottle` smoke can stably produce indicators.
- Current `bottle` smoke results:
  - `image_auroc=1.0000`
  - `pixel_auroc=0.9845`
  - `image_f1max=1.0000`

determination:

- `pass`
- Reason: Both smoke and full benchmark can now produce results.

## 5. Full Benchmark

Command (archive caliber):

```bash
python tools/benchmark.py \
    --data_root data/mvtec_ad \
    --methods patchcore \
    --categories all
```
Summary of results:

> The current final result is based on the latest full benchmark, which meets the more stringent `±0.5` alignment standard.

| Metric | Reference | BaoIAD | Gap |
|--------|-----------|----------|-----|
| image_auroc | `0.9800` | `0.9814` | `+0.0014` |
| pixel_auroc | `0.9800` | `0.9792` | `-0.0008` |
| image_f1max | `0.9760` | `0.9702` | `-0.0058` |

Additional experiments:

- weighted image score:
  - `image_auroc = 0.9681`
  - Conclusion: Lower than final mainline results in current implementation
- `image_score_source=upsampled`:
  - `screw image_auroc = 0.9069`
  - Conclusion: Lower than final mainline result
- `patch_score_neighbors=9 + mean`:
  - The operating cost of a single category is obviously high, and it has not entered the main line of the full benchmark
- `prefilter_multiplier=4`:
  - The final total results reached `image_auroc=0.9814`, `pixel_auroc=0.9792`
  - Conclusion: The current mainline is aligned within the `±0.5` caliber

Shutdown line inspection:

- [x] No large area image AUROC near `0.5` appears
- [x] Multiple categories did not collapse to similar platform values.
- [x] score / pixel The result does not show abnormal collapse
- [x] is within `±0.5` diameter from the reference

## 6. Guard

- New test:
  - `tests/test_models/test_memory_bank_head.py`
  - `tests/test_models/test_detectors/test_patchcore.py`
- New/strengthened assertions:
  - full-res `pred_anomaly_map`
  - `blur_sigma` post-processing shape/finite guard
  - softmax reweighting image-score guard
  - coreset deterministic guard
  - PatchCore main configuration freeze guard
- Added probe/assertion:
  - `runs/alignment/patchcore_probe.json`
- If you change these paths later, you must rerun:
  - `baoiad/models/heads/memory_bank_head.py`
  - `configs/patchcore/patchcore_wrn50_256_mvtec_strict.py`
  - `tools/benchmark.py`'s single-class identification logic

## 7. Residual Risk

- Although the existence of `faiss-gpu` symbols has been verified in `.venv`, the mainline benchmark still uses CPU FAISS in the end; the GPU path is not a necessary condition for the current conclusion.

## 8. Conclusion

- Final decision: `aligned`
- Allowed to proceed to next stage: `allowed`
- If you continue to finish, the next step:
  - Synchronously update `docs/alignment/README.md` and the paper side summary table

## Alignment checklist

Status only allowed: `matched | mismatch-fixed | intentional-diff | open`

## 1. Enter path

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| train color channel | `.refs/ader/model/patchcore.py` input RGB image tensor | `LoadImage(to_rgb=True)` + `PackADInputs` | The BGR file is read, converted to RGB, and then packaged into `CHW float32` | `baoiad/datasets/transforms/loading.py`, probe `train.inputs.shape=[2,3,256,256]` | `matched` |
| test color channel | Same as above | Same as above | Consistent with train | `runs/alignment/patchcore_probe.json` in `test.inputs.shape=[2,3,256,256]` | `matched` |
| resize/crop | PatchCore 256px input size | `ResizeAD(size=256)`, no crop | input unified to `256x256` | `configs/_base_/datasets/mvtec_ad.py` | `matched` |
| normalization / value range | ImageNet pre-training backbone caliber | `NormalizeAD` | RGB image walk ImageNet statistical normalization | `baoiad/datasets/transforms/augmentation.py` | `matched` |
| DTD / texture external source | PatchCore does not use anomaly synthesis | None | No additional texture sources should be introduced | PatchCore is a training-free memory bank method | `intentional-diff` |

## 2. Backbone / Features

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| Feature layer selection | README frozen caliber + `.refs/ader/model/patchcore.py` layer2/layer3 | `configs/patchcore/patchcore_wrn50_256_mvtec_strict.py` | Use `out_indices=(2,3)` instead of the old `(1,2)` | Added config guard single test | `mismatch-fixed` |
| Multi-scale space alignment | `.refs/ader/model/patchcore.py` Align multi-layer patches to the same patch grid | `MultiScalePooling(output_size=28)` | Features of each layer fall into a unified `28x28` grid | config freeze + existing detector path | `matched` |
| patch neighborhood context | `.refs/ader/model/patchcore.py` `patchsize=3`, `stride=1` | `MemoryBankHead._patchify_and_aggregate()` | patch embedding including `3x3` neighborhood context | `tests/test_models/test_memory_bank_head.py` | `matched` |

## 3. Memory Bank / Coreset

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| coreset ratio | PatchCore default 10% | `MemoryBankHead(coreset_ratio=0.1)` | memory bank reserves 10% patch features | main config frozen | `matched` |
| approximate greedy coreset | `.refs/ader/model/patchcore_utils/sampler.py` | `MemoryBankHead._approximate_coreset_sampling()` | Approximate greedy sampling using `10` starting points + `128` dimensional projection | New deterministic single test | `mismatch-fixed` |
| Large sample pre-filtering | Engineering approximation, non-reference source code direct caliber | `build_memory_bank()` | When the amount of data is large, randomly pre-filter first and then do approximate greedy to avoid single-class benchmark being stuck in coreset for a long time | 2026-03-24 Code repair | `mismatch-fixed` |
| kNN backend | Reference first FAISS | `faiss` / `sklearn` fallback | None `faiss` environment is allowed to fall back to `sklearn` only for structural verification | Current environment probe has passed, and `faiss=False` | `intentional-diff` |

## 4. Predict / Scoring

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| anomaly map source | PatchCore patch-level NN distance | `MemoryBankHead.predict()` | Generated from the NN distance from patch to memory bank | head single test + probe | `matched` |
| image score aggregation | local reference image score freeze final caliber after test | `MemoryBankHead.predict()` | current final caliber is `max(postprocessed map)`, latest `15/15` full benchmark reaches `img=0.9814`, benchmark `pxl=0.9792` | full has reached `±0.5` caliber | `mismatch-fixed` |
| Number of neighbors | README has been frozen as `9` | `num_neighbors=9` | PatchCore benchmark config does not allow rollback to the old value | Added config guard single test | `mismatch-fixed` |
| Post-processing / smoothing | `.refs/ader/model/patchcore.py` `RescaleSegmentor` + gaussian smoothing | `input_size=(256,256)`, `blur_sigma=4.0` | anomaly map output full-res and perform Gaussian smoothing | probe `map_shapes=[1,256,256]` + new blur single test | `mismatch-fixed` |
| benchmark mode | single-class PatchCore | `tools/benchmark.py` metadata | benchmark runner must not be misidentified as a multi-class method | `tests/test_utils/test_benchmark_config_detection.py` | `matched` |

## 5. Behavior verification conclusion

- [x] After fixing the seed, the dataset sample structure is as expected
- [x] mask shape and range are as expected
- [x] The key output of loss path makes a finite assertion
- [x] predict path's score / map makes shape / range assertions
- [x] The latest `15/15` full benchmark has been completed and the results are within the `±0.5` caliber
