# AnomalyDINO strict-alignment evidence

- **Method slug**: `anomalydino`
- **Family**: Few-shot / registration
- **Method README**: [`configs/anomalydino/README.md`](../../configs/anomalydino/README.md)

This file preserves the fixed alignment evidence migrated from the manuscript evidence workspace. The content below is translated to English where needed and kept inside this repository so the strict-alignment state can be reviewed without relying on an external paper checkout.

## Config entry points

- [`configs/anomalydino/anomalydino_vitb14_448_mvtec_strict.py`](../../configs/anomalydino/anomalydino_vitb14_448_mvtec_strict.py)
- [`configs/anomalydino/anomalydino_vitb14_448_visa.py`](../../configs/anomalydino/anomalydino_vitb14_448_visa.py)

## Detailed alignment report

**Status**: `playbook-complete`
**Date**: `2026-03-28`

## 1. Reference freezing

- Reference warehouse: `https://github.com/dammsi/AnomalyDINO`
- Reference commit: `b9d1c2648e3a5247437d4d953d907a8f3d994457` (`2025-12-17`, `main`)
- Reference entrance:
  - `run_anomalydino.py`
  - `src/detection.py`
  - `src/utils.py`
  - `src/backbones.py`
- Dataset/Category: MVTec AD, single-class, fresh strict `15/15` Completed
- Input resolution: short-edge `448`, keep-ratio, then crop to an integral multiple of `14`
- seed: strict mainline fixed `few_shot_seed=0`
- Indicator definition: image score=`mean_top1p`; pixel map=`resize + gaussian sigma=4`
- intentional diff:
  - Officially uses FAISS `L2_normalized / 2`; BaoIAD strict uses normalized dot-product to accurately achieve the same `1 - cosine` distance
  - The current warehouse does not yet provide an independent `tools/alignment_probe.py` CLI; the current probe is first archived in the form of ad-hoc artifact

## 2. Code path comparison conclusion

See [`anomalydino_checklist.md`](anomalydino_checklist.md) for the control matrix.

### Consistency confirmed

- DINOv2 patch token backbone, ImageNet normalization, `mean_top1p` image score, `sigma=4` smoothing have been frozen to official standards
- strict configuration has fallen separately in `configs/anomalydino/anomalydino_vitb14_448_mvtec_strict.py`

### Fixed inconsistencies

- few-shot picture selection changed from random `randperm` to official `sorted(paths)[seed*k:(seed+1)*k]`
- support rotation changed from historical `0/90/180/270` to official `0/45/90/135/180/225/270/315`
- The masking semantics has been changed from the historical "memory bank global PCA filter patch" to the official "test-side PCA mask enabled by category, support does not mask by default"
- `_get_patch_tokens()` changed from nearest-multiple resize to official patch-size crop
- predict anomaly map changed from wrong square resize to `inputs.shape[2:]`
- The default entrance of benchmark is switched to strict config, and config-selection guard is added.

### Items that are still open

- The independent probe entry has not been completed in the form of unified CLI, but the ad-hoc probe artifact has been archived

## 3. Behavioral evidence

### Targeted Tests

Added/updated:

- `tests/test_models/test_detectors/test_anomalydino.py`
- `tests/test_utils/test_benchmark_config_detection.py`

Coverage points:

- strict `agnostic` masking category selection
- strict `informed` rotate category selection
- PCA mask output shape / dtype / non-empty
- official sorted-slice few-shot picture selection
- strict 8-angle support rotation
- limited predict score / map after build-memory-bank
- benchmark default entry priority strict config

### Lightweight Probe

- Product: `runs/alignment/anomalydino_strict_bottle_probe.json`
- train batch:
  - `inputs.shape=[1,3,448,448]`
  - `memory_bank.shape=[8192,768]`
- test batch:
  - `pred_anomaly_map.shape=[1,448,448]`
  - `pred_score=0.4157`
- Conclusion:
  - strict `loss -> build_memory_bank -> predict` structure link has completed a fixed seed verification on the real `bottle` data
  - Score / map / memory bank statistics are all finite

### Smoke Run

Order:

```bash
python tools/benchmark.py \
    --data_root data/mvtec_ad \
    --methods anomalydino \
    --categories bottle \
    --batch_size 1 \
    --timeout 3600 \
    --output runs/alignment/anomalydino_strict_bottle_smoke.json
```
result:

- `bottle img=1.0000`
- `bottle pxl=0.9873`
- `bottle AUPRO=0.9628`
- Output file: `runs/alignment/anomalydino_strict_bottle_smoke.json`

determination:

- strict config, DINOv2 backbone, few-shot support build, and predict path can already run on real data
- Currently `bottle` smoke does not trigger stop-line

### Fresh Strict `15/15`

Order:

```bash
python tools/benchmark.py \
    --data_root data/mvtec_ad \
    --methods anomalydino \
    --categories bottle cable capsule carpet grid \
    --batch_size 8 \
    --output runs/alignment/anomalydino_strict_full_bs8_part1.json

python tools/benchmark.py \
    --data_root data/mvtec_ad \
    --methods anomalydino \
    --categories tile toothbrush transistor wood zipper \
    --batch_size 8 \
    --output runs/alignment/anomalydino_strict_full_bs8_part3.json

python tools/benchmark.py \
    --data_root data/mvtec_ad \
    --methods anomalydino \
    --categories hazelnut \
    --batch_size 8 \
    --output runs/alignment/anomalydino_strict_hazelnut_bs8.json

python tools/benchmark.py \
    --data_root data/mvtec_ad \
    --methods anomalydino \
    --categories leather metal_nut \
    --batch_size 8 \
    --output runs/alignment/anomalydino_strict_full_bs8_part2e.json

python tools/benchmark.py \
    --data_root data/mvtec_ad \
    --methods anomalydino \
    --categories pill screw \
    --batch_size 8 \
    --output runs/alignment/anomalydino_strict_full_bs8_part2f.json
```
Merged product:

- `runs/alignment/anomalydino_strict_full_15of15.json`

result:

- fresh strict `15/15 image_auroc = 0.9721`
- fresh strict `15/15 pixel_auroc = 0.9570`
- Relative Paper `1-shot img=0.966`: `+0.61%`

Main weak categories:

- image: `screw=0.8809`, `capsule=0.9039`, `transistor=0.9329`, `cable=0.9423`, `pill=0.9654`
- pixel: `transistor=0.8232`, `cable=0.9059`, `zipper=0.9296`, `wood=0.9317`, `pill=0.9541`

Stop line judgment:

- Multiple categories not present image AUROC close to `0.5`
- There is no uniform collapse to similar platform values.
- The current weak classes are mainly concentrated in `screw / capsule / transistor / cable`
- The conclusion is imperfect but does not trigger the stop-line and can be archived as an active strict conclusion

### Post-Alignment Targeted Ablations

Auxiliary products:

- `runs/alignment/anomalydino_informed_cable_transistor.json`
- `runs/alignment/anomalydino_agnostic_no_mask_capsule_screw.json`
- `runs/alignment/anomalydino_variant_compare.json`
- `tools/anomalydino_compare_variants.py`

Comparison completed:

- `informed` does not bring image improvements to `cable / transistor`:
  - `cable image -0.0090`, `pixel +0.0045`
  - `transistor image -0.0092`, `pixel +0.0475`
- `agnostic_no_mask` does not constitute a superior auxiliary line to `capsule / screw`:
  - `capsule image +0.0028`, but `pixel -0.0110`
  - `screw image -0.0209`, `pixel -0.0247`

determination:

- Currently completed targeted ablation has found no alternative preprocessing that is more stable than strict `agnostic`
- So no new `aligned-plus` or auxiliary mainline

## 4. Current main line and historical auxiliary line

### Strict main line

- Configuration: `configs/anomalydino/anomalydino_vitb14_448_mvtec_strict.py`
- Caliber: official `main@b9d1c264` + `preprocess='agnostic'` + `few_shot_seed=0`

### Legacy auxiliary line

- Configuration: `configs/anomalydino/anomalydino_vitb14_448_mvtec.py`
- Meaning: historical best-reproducible path, fixed square `448`, `pca_foreground=False`, 4-corner rotation
- Historical results: `img=0.9553`, `pxl=0.9498`

legacy results remain, but no longer represent strict official alignment conclusions.

## 5. Guard

- New test:
  - `tests/test_models/test_detectors/test_anomalydino.py`
  - `tests/test_utils/test_benchmark_config_detection.py`
- Added strict configuration:
  - `configs/anomalydino/anomalydino_vitb14_448_mvtec_strict.py`
- benchmark entry guard:
  - `tools/benchmark.py` now prefers strict config
  - `AnomalyDINODetector` has been added to the direct-runner training-free path

## 6. Residual Risk

- The main image weak classes under strict `agnostic` are still concentrated in `screw / capsule / transistor / cable`
- The current conclusion is based on fresh strict `15/15`; if the official commit or DINO weight source is subsequently switched, the entire group will need to be rerun.
- The independent probe CLI is still missing, but the `bottle` probe artifact has been completed

## 7. Conclusion

- Final judgment: strict official The main line has been completed and filed as current `playbook-complete`
- Allow to proceed to next stage: Yes
- Next action:
  - If you need to get closer to the official upper limit, give priority to `screw / capsule / transistor / cable` targeted diagnosis

## Alignment checklist

Status only allowed: `matched | mismatch-fixed | intentional-diff | open`

## 1. Enter path

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| train/test color channel | `src/detection.py` + `cv2.cvtColor(..., COLOR_BGR2RGB)` | `baoiad/datasets/transforms/loading.py` | Convert BGR files to RGB after reading | `LoadImage` clear `cv2.COLOR_BGR2RGB` | `matched` |
| resize / crop | `src/backbones.py:DINOv2Wrapper.prepare_image()` | `configs/anomalydino/anomalydino_vitb14_448_mvtec_strict.py` + `baoiad/models/detectors/anomalydino.py:_get_patch_tokens()` | short-edge `448` keep-ratio, then crop to an integral multiple of patch-size | strict config uses `ResizeAD(size=448, keep_ratio=True)`; detector is changed to crop instead of round-resize | `mismatch-fixed` |
| normalization / value range | ImageNet normalization in `src/backbones.py` | `NormalizeAD` | RGB float32 input imageNet mean/std | `baoiad/datasets/transforms/augmentation.py` | `matched` |

## 2. Support Bank / Few-shot

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| few-shot image selection | `src/detection.py:img_ref_samples = sorted(...)[seed*k:(seed+1)*k]` | `AnomalyDINODetector._selected_reference_paths()` | Make deterministic slice based on sorted file names | Added `test_few_shot_selection_matches_official_sorted_slice` | `mismatch-fixed` |
| support rotation angle | `src/utils.py:augment_image(... angles=[0,45,...,315])` | `AnomalyDINODetector._rotation_angles_for_category()` | strict `agnostic` uses 8-angle rotation | new `test_agnostic_rotation_augments_support_bank_with_eight_angles` | `mismatch-fixed` |
| informed rotation category | `src/utils.py:rotation_default` | `AnomalyDINODetector._should_apply_rotation()` | `informed` rotation only `hazelnut/screw` | new `test_informed_preprocess_rotates_only_selected_categories` | `mismatch-fixed` |
| support masking default value | `run_anomalydino.py --mask_ref_images=False` | strict config `mask_ref_images=False` | default support bank does not do PCA mask | strict config freeze + detector press switch execution | `mismatch-fixed` |

## 3. Predict / Scoring

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| PCA foreground mask category | `src/utils.py:masking_default` | `AnomalyDINODetector._should_apply_masking()` | strict `agnostic` Only `capsule/hazelnut/pill/screw/toothbrush` enables mask | New `test_agnostic_preprocess_uses_official_masking_categories` | `mismatch-fixed` |
| PCA mask construction | `src/backbones.py:compute_background_mask()` | `AnomalyDINODetector._build_background_mask()` | first-PC threshold + center check + morphology | New `test_background_mask_has_expected_shape` | `mismatch-fixed` |
| Distance definition | `src/detection.py` after normalization `faiss L2 / 2` | `AnomalyDINODetector._knn_distances()` | Calculate `1 - cosine` nearest neighbor distance | torch normalization dot-product is mathematically equivalent to the reference | `intentional-diff` |
| image score aggregation | `src/post_eval.py:mean_top1p()` | `forward(..., mode='predict')` | `top 1%` patch distance mean | detector strict main line freeze `top_ratio=0.01` | `matched` |
| Post-processing / smoothing | `src/utils.py:dists2map()` `gaussian sigma=4` | `_gaussian_blur(sigma=4)` | `sigma=4` smoothing after upsampling | strict config freeze `gaussian_sigma=4.0` | `matched` |
| Non-square output | Official resize back to anomaly map according to the original image size | `predict` in `F.interpolate(..., size=inputs.shape[2:])` | Keep-ratio input cannot be forced into a square | This time the bug of only using `inputs.shape[2]` has been fixed | `mismatch-fixed` |

## 4. Behavior verification conclusion

- [x] fixed-seed few-shot image selection rules have been locked by targeted test
- [x] PCA mask shape / dtype / non-empty covered by targeted test
- [x] build-memory-bank predict score / map finite is covered by detector test
- [x] `runs/alignment/anomalydino_strict_bottle_probe.json` has archived the real data structure product once
- [x] strict `bottle` smoke has been run through, and the result is `img=1.0000`, `pxl=0.9873`
- [x] fresh strict `15/15` full benchmark has been completed, the results can be found in `runs/alignment/anomalydino_strict_full_15of15.json`
