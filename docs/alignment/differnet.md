# DifferNet strict-alignment evidence

- **Method slug**: `differnet`
- **Family**: Normalizing flow
- **Method README**: [`configs/differnet/README.md`](../../configs/differnet/README.md)

This file preserves the fixed alignment evidence migrated from the manuscript evidence workspace. The content below is translated to English where needed and kept inside this repository so the strict-alignment state can be reviewed without relying on an external paper checkout.

## Config entry points

- [`configs/differnet/differnet_alexnet_256_mvtec_strict.py`](../../configs/differnet/differnet_alexnet_256_mvtec_strict.py)
- [`configs/differnet/differnet_alexnet_256_visa.py`](../../configs/differnet/differnet_alexnet_256_visa.py)

## Detailed alignment report

**Status**: `aligned`
**Date**: `2026-03-25`

## 1. Reference freezing

- Reference warehouse: `https://gh-proxy.com/https://github.com/marco-rudolph/differnet`
- Reference commit: `9bdf02686297a093fb206ffeba64b1c0e78182b6` (`HEAD` as of `2026-03-24`)
- Refer to config/checkpoint:
  - Official `train.py`
  - Official `freia_funcs.py`
  - AlexNet backbone + multi-scale `448 / 224 / 112`
- Data set/category: MVTec AD, the official standard is single category training/testing
- Input resolution: top-level `448x448`, and `448 / 224 / 112` multi-scale inside the detector
- seed: The official warehouse is not explicitly frozen; BaoIAD’s probe/smoke uses `42` uniformly
- Indicator definition: mainly image AUROC; paper Table 1 average image AUROC = `0.949`
- intentional diff:
  - BaoIAD API requires `pred_anomaly_map`, currently still retains uniform placeholder map; DifferNet is essentially an image-level method
  - Pixel-side indicators are only used for interface compatibility checks and cannot be used as main evidence for alignment with the paper

## 2. Code path comparison conclusion

See [`differnet_checklist.md`](differnet_checklist.md) for the control matrix.

### Consistency confirmed

- The single-scale feature paths of AlexNet `features -> GAP -> 256-dim` are consistent
- The `768-dim` feature path after splicing the three scales is consistent
- The image scoring caliber of `64` random transform + `mean(z^2)` in the test phase is consistent

### Fixed inconsistencies

- `SubnetFC.forward()` no longer incorrectly calls BatchNorm
- Training loss has been restored to official `mean(...)/z.shape[1]`
- Training and test rotations have been changed back to image-by-image random rotation semantics within the detector
- `SubnetFC internal_size` has been reverted to `2048` in official `config.py`
- FrEIA default `tanh` clamp has been replaced by the `atan` clamp used by the reference implementation

### Items that are still open

- Non-blocking items; according to the official `best image_auroc` standard, the current implementation has met the alignment requirements

## 3. Behavior Probe

Order:

```bash
python tools/alignment_probe.py configs/differnet/differnet_alexnet_256_mvtec_strict.py \
    --splits train test \
    --max-batch-size 2 \
    --device cuda \
    --output runs/alignment/differnet_probe_v16.json
```
in conclusion:

- `runs/alignment/differnet_probe_v16.json` passed, the batch, loss, and predict structures of train/test are all normal.
- The output of the `loss` path is a finite scalar, and the `pred_score / pred_anomaly_map` of the `predict` path are all present and finite.
- The current probe is enough to prove that the structural caliber is passed, but it cannot replace the performance evidence after real training.

Key statistics:

- dataset sample:
  - train sample keys: `cls_name / defect_type / gt_label / gt_mask / img_path`
  - The train batch is `bottle/train/good/*` normal samples, and the test batch is `bottle/test/broken_large/*` abnormal samples
  - The input shape is `[3, 448, 448]`, `gt_mask` is finite and the test mask is binary
- loss path:
  - `loss = 4.3153`
  - `loss` is a finite scalar, no NaN / Inf
- predict path:
  - `pred_score mean = 6.9723`
  - `pred_score std = 0.2498`
  - `pred_anomaly_map` shape = `(1, 448, 448)`, all limited

## 4. Small-scale controlled experiment

Experimental setup:

- Category: `bottle`
- Training budget: `1 epoch`
- seed: `42`
- Comparison object: v16 code path after strict official standard correction; historical v15 smoke is only reserved for reference and no longer used as official standard evidence
- Command:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    configs/differnet/differnet_alexnet_256_mvtec_strict.py \
    --work-dir runs/alignment/differnet_smoke_bottle_v16 \
    --cfg-options \
        train_cfg.max_epochs=1 \
        train_cfg.val_interval=1 \
        train_dataloader.num_workers=0 \
        val_dataloader.num_workers=0 \
        test_dataloader.num_workers=0 \
        train_dataloader.persistent_workers=False \
        val_dataloader.persistent_workers=False \
        test_dataloader.persistent_workers=False
```
observe:

- In `runs/alignment/differnet_smoke_bottle_v16/20260324_154756/20260324_154756.log`, the final training round of `Epoch(train) [1][9/9]` is `loss = 0.5123`
- The `Epoch(val) [1][6/6]` indicator for the same run is:
  - `ad/bottle/image_auroc = 0.9746`
  - `ad/bottle/pixel_auroc = 0.6907`
  - `ad/bottle/image_ap = 0.9926`
- There is no phenomenon that the image score collapses to a single platform, the loss diverges, or the verification score is close to `0.5`.
- The low pixel index is an expected phenomenon, because the current anomaly map is still a uniform placeholder and is not the main output of the paper.

determination:

- `pass`
- Reason: The real data training/verification link has been opened, the image-side smoke indicator is normal, and the abnormal shutdown line has not been triggered.

## 5. Full Benchmark

Target command:

```bash
python tools/benchmark.py \
    --data_root data/mvtec_ad \
    --methods differnet \
    --categories all \
    --output runs/alignment/differnet_v16.json
```
There are two result reading calibers in the current warehouse:

1. **final-val caliber**: The old version `benchmark.py` directly takes the last verification result.
2. **best-val caliber**: Official `train.py` uses `Score_Observer.max_score` to track the highest image AUROC during training. The current configuration has explicitly declared `benchmark_result_selector = dict(mode='best', metric='image_auroc')`, and subsequent benchmarks should be executed according to this standard.

Summary of results:

| Metric | Reference | BaoIAD final-val | BaoIAD best-val | Gap(best) |
|--------|-----------|----------|-----|
| image_auroc | `0.949` | `0.9322` | `0.9500` | `+0.1%` |
| pixel_auroc | `—` | `0.7337` | `0.7359` | `—` |

illustrate:

- The current strictly official v16 code path has completed the probe, `bottle` smoke and `15/15` standard benchmarks.
- The full benchmark is executed with 4 GPU shards, and the result is:
  - `runs/alignment/differnet_v16_part1.json`
  - `runs/alignment/differnet_v16_part2.json`
  - `runs/alignment/differnet_v16_part3.json`
  - `runs/alignment/differnet_v16_part4.json`
- The existing logs are re-aggregated according to the official best-val semantics and the product is:
  - `runs/alignment/differnet_v17_best.json`
- The v14 / v15 results in the historical documents are no longer used as the basis for the current code status, because the official source code has been corrected after checking the `internal_size`, loss scaling and test transform semantics.

Category results (image AUROC):

- `bottle 0.9865`
- `cable 0.9621`
- `capsule 0.8835`
- `carpet 0.9013`
- `grid 0.7101`
- `hazelnut 0.9839`
- `leather 0.9711`
- `metal_nut 0.9355`
- `pill 0.8693`
- `screw 0.9510`
- `tile 0.9913`
- `toothbrush 1.0000`
- `transistor 0.8921`
- `wood 0.9956`
- `zipper 0.9504`

Key categories re-aggregated according to best-val caliber:

- `grid: 0.7101 -> 0.8045` at epoch `96`
- `capsule: 0.8835 -> 0.8919` at epoch `168`
- `carpet: 0.9013 -> 0.9209` at epoch `144`
- `pill: 0.8693 -> 0.8920` at epoch `120`
- `transistor: 0.8921 -> 0.9062` at epoch `144`
- Overall mean: `0.9322 -> 0.9500`

Shutdown line inspection:

- [x] `bottle` smoke not present image AUROC close to `0.5`
- [x] `bottle` smoke No unified platform value collapse occurs
- [x] `15/15` benchmark does not appear to have overall collapse near multi-category `0.5`
- [x] The difference between the mean and the reference under best-val caliber is within the acceptable range (`+0.1%`)
- [x] The average value under the official `best-val` caliber is consistent with the reference, and the current single type fluctuation is acceptable

## 5.1 Grid Targeted A/B

Low-cost A/B results only on `grid`:

| Variant | Budget | image_auroc | Conclusion |
|---------|--------|-------------|------|
| baseline | `1 epoch` | `0.6734` | current v16 smoke baseline |
| `test_rotation_mode=fixed` | `1 epoch` | `0.7118` | Fixed angle alone has limited help |
| `loss_normalize_by_dim=False` | `1 epoch` | `0.8388` | Significant early improvement |
| `fc_internal_size=1536` | `1 epoch` | `0.6241` | Significantly worse, excluded |
| baseline | `5 epochs` | `0.7360` | Close to the current full-train final platform |
| `loss_normalize_by_dim=False` | `5 epochs` | `0.7352` | The early advantage has basically disappeared |
| `loss_normalize_by_dim=False`, `test_rotation_mode=fixed` | `1 epoch` | `0.8814` | The strongest early signal at present |
| `loss_normalize_by_dim=False`, `test_rotation_mode=fixed` | `5 epochs` | `0.7402` | The early advantage dropped significantly after continued training |

determination:

- The main problem of `grid` is not a single static hyperparameter, but **obvious early peak / later regression during the training process**
- Changing `fc_internal` alone or fixing the test angle alone are not the main reasons.
- The combination of `loss` and test transform only shows obvious advantages in the early stage, indicating that the problem is more like **result selection logic + late training degradation** rather than a simple final configuration switch

## 6. Guard

- New test: `tests/test_models/test_detectors/test_differnet.py`
- Added new anti-regression points:
  - multi-scale feature concat must maintain `768-dim`
  - Default `internal_size` for `SubnetFC` must remain official `2048`
  - `SubnetFC.forward()` no longer allowed to call BatchNorm
  - Both training and testing paths must be "randomly rotated image by image" instead of fixed angle or single angle rotation batch by batch.
  - loss must remain official `mean(...)/z.shape[1]` normalized
  - predict results must return limited `pred_score` and `(1, H, W)` anomaly maps
- Added probe product: `runs/alignment/differnet_probe_v16.json`
- Added smoke product:
  -`runs/alignment/differnet_smoke_bottle_v16/20260324_154756/20260324_154756.log`
  - `runs/alignment/differnet_smoke_bottle_v16/20260324_154756/vis_data/scalars.json`
- Added `grid` targeted A/B product:
  -`runs/alignment/differnet_grid_smoke_v16_baseline/...`
  - `runs/alignment/differnet_grid_smoke_v16_fixedtest/...`
  - `runs/alignment/differnet_grid_smoke_v16_nonorm/...`
  - `runs/alignment/differnet_grid_smoke_v16_fc1536/...`
  - `runs/alignment/differnet_grid_smoke_v16_baseline_e5/...`
  - `runs/alignment/differnet_grid_smoke_v16_nonorm_e5/...`
  - `runs/alignment/differnet_grid_smoke_v16_nonorm_fixed/...`
  - `runs/alignment/differnet_grid_smoke_v16_nonorm_fixed_e5/...`
- If you change these paths later, you must rerun:
  - `baoiad/models/detectors/differnet.py`
  - `configs/differnet/differnet_alexnet_256_mvtec_strict.py`
  - `tools/benchmark.py`
  -`tests/test_models/test_detectors/test_differnet.py`
  - `tests/test_utils/test_benchmark_config_detection.py`

## 7. Residual Risk

- `grid` is still lower than empirical expectations under the best-val caliber, but this no longer blocks the conclusion of this alignment
- The current pixel-side metric is not comparable to papers because DifferNet itself is not a pixel-level method

## 8. Conclusion

- Final decision: `aligned`
- Allowed to proceed to next stage: `yes`
- Alignment caliber: using the official `best image_auroc` semantics, the current mean of `15/15` is `0.9500`, and the gap relative to the reference `0.949` is `+0.1%`

## Alignment checklist

Status only allowed: `matched | mismatch-fixed | intentional-diff | open`

## 1. Enter path

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| train color channel | official `train.py` / PIL `RGB` | `baoiad/datasets/transforms/loading.py` | training image retention `RGB` | `LoadImage(to_rgb=True)` enabled by default; probe train input is `[3, 448, 448]` | matched |
| test color channel | official `test.py` / PIL `RGB` | `baoiad/datasets/transforms/loading.py` | test image retention `RGB` | `LoadImage(to_rgb=True)` enabled by default; probe test input is `[3, 448, 448]` | matched |
| resize / multi-scale | Official top-level `448`, then do `448 / 224 / 112` internally | `configs/differnet/differnet_alexnet_256_mvtec_strict.py` + `DifferNetDetector.extract_features()` | pipeline comes first `448`, then detector does 3-scale `interpolate` | config + `extract_features()` | matched |
| normalization / value range | official AlexNet / ImageNet normalize | `NormalizeAD` | use ImageNet mean/std | `NormalizeAD.IMAGENET_MEAN / STD` | matched |
| Training rotation enhancement | Official `RandomRotation(180)`, sample-by-sample rotation | `DifferNetDetector.forward(mode='loss')` | Each picture samples the `[-180, 180]` angle independently, and pipeline rotation cannot be repeated | `test_loss_rotates_each_image_individually`; train pipeline no longer exists `RandomRotation` | mismatch-fixed |
| Single-category training mode | Official category-by-category training/testing | `configs/differnet/differnet_alexnet_256_mvtec_strict.py` | The current config specifies `cls_names=['bottle']` separately | The train/test dataset in the config only loads `bottle` | matched |

## 2. Feature / Flow Path

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| backbone feature path | official AlexNet `features -> GAP` | `_extract_single_scale()` | each scale output `256-dim` feature | `_extract_single_scale()` + AlexNet `features` | matched |
| multi-scale concat | official 3-scale concat | `extract_features()` | 3 scales spliced into `768-dim` | `test_forward_tensor_returns_expected_feature_shape` | matched |
| coupling clamping | official `atan` clamp | `GlowCouplingLayer.log_e()` | using `clamp * 0.636 * atan(s / clamp)` | `differnet.py` current implementation | mismatch-fixed |
| permutation | Official fixed random permutation | `PermuteLayer(seed=k)` | Each block uses a fixed seed to generate permutation | `FlowSequence` in `seed=k` | matched |
| subnet internal_size | official `config.py:fc_internal=2048` | `SubnetFC.__init__()` | `384 -> 2048 -> ... -> 768` | `test_subnet_fc_uses_reference_internal_size` | mismatch-fixed |
| How to use BatchNorm | Official definition `bn` but forward is not used | `SubnetFC.forward()` | `bn` must not enter the forward main path | `test_subnet_fc_forward_skips_batchnorm` | mismatch-fixed |

## 3. Loss

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| NLL input form | official `utils.py:get_loss` | `DifferNetDetector.forward(mode='loss')` | `mean(0.5 * sum(z^2) - jac) / z.shape[1]` | `test_loss_matches_reference_dimension_normalized_nll` | mismatch-fixed |
| Dimension normalization | Official `... / z.shape[1]` | `DifferNetDetector.forward(mode='loss')` | Must retain feature-dim normalization | `test_loss_matches_reference_dimension_normalized_nll` | mismatch-fixed |
| reduction | official `torch.mean(...)` | `loss.mean()` | using `mean` reduction | current implementation consistent | matched |
| loss finite | official training loss should be limited | `alignment_probe` + unit test | `loss` must be a finite scalar | `runs/alignment/differnet_probe_v16.json` + `test_forward_loss_returns_finite_scalar` | matched |

## 4. Predict / Scoring

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| multi-transform test | Official `64` times random transform | `_predict_with_transforms()` | The default configuration retains `n_transforms=64`, and each sample is randomly rotated independently | `test_predict_rotates_each_image_individually` | mismatch-fixed |
| image score aggregation | official `mean(z^2)` over transforms/features | `_predict_with_transforms()` | average by transforms + feature dims | `differnet.py` current implementation | matched |
| anomaly map source | The essence of the official method is image-level only | uniform map in `build_predict_results()` | retain the API compatible placeholder map and do not forge the real pixel map | unit test + report description | intentional-diff |
| Pooling / smoothing | Officially no independent pixel post-process | Currently no additional post-processing | image score only relies on `mean(z^2)` | The current implementation is consistent; pixel map only takes place | intentional-diff |
| predict finiteness | `pred_score` should be finite | `alignment_probe` + unit test | `pred_score / pred_anomaly_map` all finite | `runs/alignment/differnet_probe_v16.json` + `test_forward_predict_returns_uniform_maps_matching_scores` | matched |

## 5. Behavior verification conclusion

- [x] After fixing the seed, the dataset sample structure is as expected
- [x] mask shape and range are as expected
- [x] The key intermediate quantity of the loss path has a shape / range assertion.
- [x] predict path's score / map makes shape / range assertions
- [x] `alignment_probe` passed
- [x] `bottle` smoke does not trigger the abnormal shutdown line

## 6. Remarks

- Currently `pixel_auroc` can only be used as an interface compatibility indicator and cannot be directly compared with the main results of the paper.
- If the subsequent `15/15` benchmark is still significantly behind, prioritize the remaining detailed differences between the `grid` category and the official data enhancement/rotation.
