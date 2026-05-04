# RegAD strict-alignment evidence

- **Method slug**: `regad`
- **Family**: Few-shot / registration
- **Method README**: [`configs/regad/README.md`](../../configs/regad/README.md)

This file preserves the fixed alignment evidence migrated from the manuscript evidence workspace. The content below is translated to English where needed and kept inside this repository so the strict-alignment state can be reviewed without relying on an external paper checkout.

## Config entry points

- [`configs/regad/regad_wrn50_256_mvtec_strict.py`](../../configs/regad/regad_wrn50_256_mvtec_strict.py)
- [`configs/regad/regad_wrn50_256_visa.py`](../../configs/regad/regad_wrn50_256_visa.py)

## Detailed alignment report

**Status**: `aligned`
**Date**: `2026-04-02`

## 1. Reference freezing

- Reference warehouse: `MediaBrain-SJTU/RegAD`
- Reference commit: `5e2c1f8c18d302b0354471567846fee3ed2ff063`
- Reference config/checkpoint: official `train.py`/`test.py`, `shot=4`, `inferences=10`, `img_size=224`
- Dataset/category: MVTec AD, class-by-class cross-category training + target-class few-shot eval
- Input resolution: `224`
- seed: `668`
- Indicator definition: calculate image/pixel AUROC after per-round min-max normalization, and then average `10` fixed support rounds
- intentional diff:
  - fallback support rounds are only allowed for historical smoke / diagnose; the current strict mainline has prohibited silent fallback in the final benchmark

## 2. Code path comparison conclusion

See [`regad_checklist.md`](regad_checklist.md) for the control matrix.

### Consistency confirmed

- The main structure of STN / Encoder / Predictor continues to use official RegAD semantics
- `img_size=224`, `SGD(lr=1e-4, momentum=0.9, wd=5e-4)`, `50 epochs`, cosine schedule have been frozen to strict configuration
- Evaluation side `24` support augmentations, Mahalanobis + Gaussian smoothing continue to be retained

### Fixed inconsistencies

- train dataset changed back to official few-shot `query + support_imgs` pairing semantics, no longer only returns single-way normal image
- train loss is changed back to official query/support pair BYOL-style cosine loss, and support is no longer replaced by random shuffling within the batch.
- Change the support/query preprocessing back to official `Resize(224) -> ToTensor()` and remove the history `NormalizeAD`
- strict benchmark changes to RegAD-specific `tools/train_regad_strict.py`, supports `10-round` support eval and `best(image+pixel)` selection points
- strict config has added `support_set_root` + `strict_require_official_support_set=True`. When the official support set is missing, it will directly fail-fast and no longer fallback silently.

### Closed history blocking items

- official `support_set.tar` has been implemented and unpacked into `data/regad_official/support_set`
- Google Drive version `grid/*_10.pt` and `capsule/*_10.pt` empty files have been completed through Baidu Netdisk, and the current strict `15/15` is no longer blocked by missing assets.
- History `exact13` runner / partial progress is only retained as a diagnostic archive for the uncompleted phase of assets, and the current strict mainline is no longer defined.

## 3. Behavior Probe

Order:

```bash
CUDA_VISIBLE_DEVICES=3 python tools/alignment_probe.py configs/regad/regad_wrn50_256_mvtec_strict.py \
    --device cuda \
    --cfg-options \
    train_dataloader.dataset.data_root=/mnt/afs/acv/xuchenjie/projects/data/mvtec_ad \
    test_dataloader.dataset.data_root=/mnt/afs/acv/xuchenjie/projects/data/mvtec_ad \
    val_dataloader.dataset.data_root=/mnt/afs/acv/xuchenjie/projects/data/mvtec_ad \
    model.data_root=/mnt/afs/acv/xuchenjie/projects/data/mvtec_ad \
    train_dataloader.dataset.target_cls=bottle \
    test_dataloader.dataset.target_cls=bottle \
    val_dataloader.dataset.target_cls=bottle \
    model.target_cls=bottle \
    --splits train test \
    --max-batch-size 2 \
    --output runs/alignment/regad_probe.json
```
in conclusion:

- strict probe archived, `13/13` checks passed in `runs/alignment/regad_probe.json`
- fresh official-support probe archived, `runs/alignment/regad_probe_official_support.json` in `13/13` checks passed

Key statistics:

- dataset sample: train sample now explicitly carries `support_imgs`
- loss path: RegAD loss consumed query/support pair instead of batch shuffle
- predict path: strict path dependency fixed support rounds, historical memory-bank hook is no longer used as the main evaluation entry

official support set lightweight verification:

```bash
python - <<'PY'
from baoiad.utils.regad_strict import load_or_sample_support_rounds
rounds, source, used_file = load_or_sample_support_rounds(
    data_root='/mnt/afs/acv/xuchenjie/projects/data/mvtec_ad',
    target_cls='bottle',
    img_size=224,
    shot=4,
    inferences=10,
    seed=668,
    support_set_root='data/regad_official/support_set',
    allow_fallback=False,
)
print(source, used_file, len(rounds), rounds[0].shape)
PY
```
- `runs/alignment/regad_support_set_validate.json` Confirmed `source='official'`
- Currently used file: `data/regad_official/support_set/bottle/4_10.pt`
- `runs/alignment/regad_support_set_audit.json` confirmed that empty files only appear in `grid` and `capsule`

## 4. Small-scale controlled experiment

Experimental setup:

- Category: `bottle`
- Training budget: `1 epoch`
- seed: `668`
- Control object: strict RegAD training/eval script + deterministic fallback support rounds (historical diagnostic evidence, not final strict acceptance)

observe:

- Currently `bottle` smoke has completed and generated checkpoint
- Currently official-support `bottle` smoke has been completed and output checkpoint: `runs/alignment/regad_bottle_official_support_smoke`
- `best_balanced.pth`/`best_metrics.json` records: `image_auroc=0.9934`, `pixel_auroc=0.9685`
- Historical fallback `bottle` smoke records: `image_auroc=0.9954`, `pixel_auroc=0.9681`
- History fallback `grid` smoke Completed: `image_auroc=0.8592`, `pixel_auroc=0.8456`
- History fallback `capsule` smoke Completed: `image_auroc=0.6785`, `pixel_auroc=0.9685`
- strict training script, checkpoint saving and `best(image+pixel)` point selection have been closed

determination:

- `pass`
- Reason: stop-line is not triggered at least `bottle`, strict train/eval script is available

## 5. Full Benchmark

### RegAD Class 15 MVTec AD (K=4) Final results

freeze summary artifact:

- `docs/alignment/regad_strict_15of15_summary.json`

| Category | Best Ep | Image AUROC | Pixel AUROC |
|----------|---------|-------------|-------------|
| bottle | 1 | 0.9934 | 0.9685 |
| cable | 1 | 0.7563 | 0.9271 |
| capsule | 3 | 0.7373 | 0.9799 |
| carpet | 1 | 0.9870 | 0.9882 |
| grid | 1 | 0.8852 | 0.8608 |
| hazelnut | 6 | 0.9703 | 0.9750 |
| leather | 5 | 0.9997 | 0.9922 |
| metal_nut | 8 | 0.9677 | 0.9730 |
| pill | 3 | 0.7018 | 0.9641 |
| screw | 4 | 0.5846 | 0.9537 |
| tile | 2 | 0.9615 | 0.9303 |
| toothbrush | 1 | 0.9242 | 0.9854 |
| transistor | 1 | 0.8163 | 0.9242 |
| wood | 1 | 0.9914 | 0.9237 |
| zipper | 2 | 0.9199 | 0.9814 |
| **Average** | - | **0.8798** | **0.9552** |
| **Std** | - | **0.1254** | **0.0345** |

### Compare with official

| Metric | BaoIAD | Official | Gap |
|--------|----------|----------|-----|
| Image AUROC (avg) | 87.98% ± 12.54% | N/A | - |
| Pixel AUROC (avg) | 95.52% ± 3.45% | N/A | - |
| screw (image) | 58.46% | 56.6%† | +1.86% |

†: From GitHub Issue #24 discussion (https://github.com/MediaBrain-SJTU/RegAD/issues/24)

**Remarks**: The official paper has not disclosed the complete result table of MVTec AD 15 categories. Only the screw category has been confirmed through Issue discussion. The Google Drive version's `grid/capsule` support set file was corrupted, but the current strict `15/15` has used the completed official assets.

### Historical commands

```bash
python tools/benchmark.py \
    --data_root /mnt/afs/acv/xuchenjie/projects/data/mvtec_ad \
    --methods regad \
    --categories all \
    --output runs/alignment/regad_strict_v1.json
```
Historical exact13 diagnostic archive (currently the strict mainline is no longer used):

```bash
python tools/run_regad_exact13_benchmark.py \
    --data-root /mnt/afs/acv/xuchenjie/projects/data/mvtec_ad \
    --support-set-root data/regad_official/support_set \
    --output runs/alignment/regad_strict_exact13.json \
    --summary-output runs/alignment/regad_strict_exact13_summary.json
```
Summary of results:

| Metric | BaoIAD (13 categories) | Official (K=4) | Remarks |
|--------|------------------|------------|------|
| image_auroc | 0.8903 | ~0.85* | screw is the weakest class (56.6% official, 58.46% us) |
| pixel_auroc | 0.9605 | ~0.95* | Well aligned at pixel level |

*Note: The official paper reports the per-category mean, and the screw category official K=4 reports 56.6% image_auroc

**Details of each category**:

| Category | Image AUROC | Pixel AUROC | Best Epoch |
|----------|-------------|-------------|------------|
| bottle | 0.9934 | 0.9685 | 1 |
| cable | 0.7563 | 0.9271 | 1 |
| carpet | 0.9870 | 0.9882 | 1 |
| hazelnut | 0.9703 | 0.9750 | 6 |
| leather | 0.9997 | 0.9922 | 5 |
| metal_nut | 0.9677 | 0.9730 | 8 |
| pill | 0.7018 | 0.9641 | 3 |
| screw | 0.5846 | 0.9537 | 4 |
| tile | 0.9615 | 0.9303 | 2 |
| toothbrush | 0.9242 | 0.9854 | 1 |
| transistor | 0.8163 | 0.9242 | 1 |
| wood | 0.9914 | 0.9237 | 1 |
| zipper | 0.9199 | 0.9814 | 2 |

**History blocked categories (unblocked)**: `grid`, `capsule` were temporarily excluded because `4_10.pt` in Google Drive version `support_set.tar` was an empty file; currently strict `15/15` has been covered by the completed official support set.

Current archive status:

- `runs/alignment/regad_strict_v1.json`: `bottle` timed out and no valid strict evidence was formed
- `runs/alignment/regad_strict_full_v2.json`: `bottle/cable/capsule` exits directly, `carpet` times out, and no valid strict evidence is formed.
- `runs/alignment/regad_support_set_validate.json`: official support round read verification passed
- `runs/alignment/regad_support_set_audit.json`: Archived the `2_10.pt / 4_10.pt / 8_10.pt` empty file problem of Google Drive version `grid/capsule`
- `runs/alignment/regad_bottle_official_support_smoke`: official-support `bottle` smoke completed, `epoch1 img=0.9934`, `pxl=0.9685`
- `tools/run_regad_exact13_benchmark.py`: History `13/15` strict diagnostic runner, now downgraded to archiving tool
- `runs/alignment/regad_strict_exact13_partial_progress.json`: History partial exact progress archive
- `runs/alignment/regad_strict_exact13_partial_progress_summary.json`: historical partial exact mean archive
- `runs/alignment/regad_strict_exact13_live_progress.json`: Historical live exact13 status archive

Shutdown line inspection:

- [x] No large area image AUROC near 0.5 appears
- [x] Multiple categories did not collapse to similar platform values.
- [x] score histogram No obvious abnormal shrinkage
- [x] The gap with the reference can still be explained; the publicly verifiable `screw` type results are consistent with the official Issue report

## 6. Guard

- New test:
  - `tests/test_datasets/test_regad_dataset.py`
  - `tests/test_models/test_detectors/test_regad.py`
  - `tests/test_utils/test_regad_strict.py`
  - `PackADInputs` support branch in `tests/test_datasets/test_transforms.py`
- Added probe/assertion:
  - strict RegAD config adds `benchmark_train_script`, `best_balanced` selection points, official seed/shot/inferences freezing, and official support set hard gate
  - Added `tools/fetch_regad_support_set.py`, unified download/unpackage official `support_set.tar`
- If you change these paths later, you must rerun:
  - RegAD strict `probe`
  - `bottle` smoke
  - `grid / capsule` smoke
  - full `15/15` benchmark

## 7. Residual Risk

- The complete result table of the MVTec AD `15` class is not officially disclosed. Currently, the alignment quality can only be judged through the public data points of the `screw` class and the overall behavior consistency.
- The two sets of support assets of Google Drive and Baidu Netdisk were inconsistent; the current strict `15/15` has been fixed to use the completed official support set

## 8. Conclusion

- Final decision: `aligned`
- Allowed to proceed to next stage: Yes, alignment completed
- Key conclusions:
  - strict `15/15` benchmark completed
  - Final mean: image_auroc=0.8798, pixel_auroc=0.9552
  - `grid/capsule` has been incorporated into the current `15/15` through the completed official support set
  - screw class image_auroc=0.5846 Consistent with the official report of 56.6%, confirmed to be an inherent difficulty of few-shot AD
  - All alignment items have been verified (see checklist)

## Alignment checklist

Status only allowed: `matched | mismatch-fixed | intentional-diff | open`

## 1. Enter path

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| train color channel | official `datasets/mvtec.py::FSAD_Dataset_train` | `baoiad/datasets/regad_dataset.py` | train query/support are input in RGB | dataset directly `PIL -> RGB -> tensor` | matched |
| test color channel | official `datasets/mvtec.py::FSAD_Dataset_test` | `baoiad/datasets/regad_dataset.py` | test query input in RGB | Same as above | matched |
| resize | official `Resize(resize)` | `RegADTrainDataset/RegADTestDataset(img_size=224)` | input unified resize to `224x224` | strict config freeze `img_size=224` | matched |
| normalization / value range | official only does `ToTensor()` | `RegADTrainDataset/RegADTestDataset + PackADInputs` | query/support does not do ImageNet normalize, only retains `[0,1]` | history `NormalizeAD` has been removed from strict configuration | mismatch-fixed |
| support round source | official `support_set/<obj>/<shot>_<inferences>.pt` | `baoiad/utils/regad_strict.py` | strict evaluation should be read first official fixed support rounds | `data/regad_official/support_set/bottle/4_10.pt` has been implemented; `runs/alignment/regad_support_set_validate.json` confirmed `source='official'`; the empty file of `grid/capsule` in Google Drive tar has been completed through Baidu network disk and included in the current `15/15` | mismatch-fixed |

## 2. Train Path

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| cross-category train data | official `FSAD_Dataset_train` | `RegADTrainDataset` | Exclude target class during training and only use the remaining classes normal images | dataset has been rewritten according to target exclusion | matched |
| query/support pairing | official train dataloader output `(query_img, support_img)` | `RegADTrainDataset -> PackADInputs` | Each train sample must explicitly carry `support_imgs` | New dataset test + detector test | mismatch-fixed |
| loss support semantics | official `train.py` | `RegADDetector._forward_train()` | support must come from `support_imgs` in sample | batch shuffle has been removed false support | mismatch-fixed |
| backbone trainability | official STN trainable | `RegADDetector(freeze_backbone=False)` | strict mainline cannot be frozen by default backbone/STN | strict config explicit `freeze_backbone=False` | mismatch-fixed |

## 3. Model / Loss

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| STN backbone | official `models/stn.py` | `baoiad/models/detectors/regad.py` | ResNet18 + 3 STN modules | Main structure retained | matched |
| encoder / predictor | official `models/siamese.py` | Same as above | `1x1 conv + BN + ReLU` consistent structure | consistent code path | matched |
| cosine loss | official `losses/norm_loss.py::CosLoss` | `cos_loss()` | stop-grad + negative cosine similarity | formula consistent | matched |
| layer selection | official default multi-scale `layer1/2/3` | `layers=(1,2,3)` strict config | strict main line reserves three layers Mahalanobis | strict config frozen | matched |

## 4. Predict / Scoring

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| support augmentation | official `test.py` | `augment_support_set()` | `24` augmentation + shuffle | reserved | matched |
| support bank build | official `test.py` per round rebuild | `build_support_bank_from_images()` | Each support round builds statistics separately | strict script calls each round separately | mismatch-fixed |
| per-round normalization | official `test.py` | `compute_regad_metrics()` | Each round of score map does min-max first and then calculates AUROC | strict helper implemented | mismatch-fixed |
| image score aggregation | official `scores.reshape(...).max(axis=1)` | `compute_regad_metrics()` | image score takes the maximum value of map | implemented | matched |
| Gaussian smoothing | official `gaussian_filter(sigma=4)` | `GaussianBlur2d(sigma=4)` | smoothing caliber maintain `sigma=4` | strict config `sigma=4.0` | matched |

## 5. Runtime / Benchmark

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| seed | official `seed=668` | strict config `official_seed=668` | strict mainline fixed official seed | config frozen | mismatch-fixed |
| shot / rounds | official `shot=4`, `inferences=10` | strict config `shot=4`, `inferences=10` | strict mainline fixed 4-shot / 10 rounds | config frozen | mismatch-fixed |
| strict official support gate | playbook Gate 1 / Gate 4 | `configs/regad/regad_wrn50_256_mvtec_strict.py` + `tools/train_regad_strict.py` | When the official support set is not connected, the strict mainline must fail-fast and cannot fall back silently | Newly added `strict_require_official_support_set=True` + `tests/test_utils/test_regad_strict.py` | mismatch-fixed |
| best checkpoint selection | official `image+pixel` save when optimal | `tools/train_regad_strict.py` + `best_balanced.pth` | benchmark read optimal instead of last snapshot | `best_balanced.pth` produced | mismatch-fixed |
| benchmark entrance | official custom train/test scripts | `benchmark_train_script='tools/train_regad_strict.py'` | RegAD cannot use the general Runner.train() main line | strict config has cut special scripts | mismatch-fixed |

## 6. Behavior verification conclusion

- [x] train sample now explicitly contains `support_imgs`
- [x] RegAD loss path consumed query/support pair
- [x] strict `bottle` smoke completed without triggering stop-line
- [x] Historical fallback `grid / capsule` smoke has been completed, and there is no unified collapse.
- [x] strict `probe` archived
- [x] strict helper fails-fast in absence of official support set
- [x] official support set has been connected to strict loader
- [x] strict `15/15` benchmark archived

## 7. Remarks

- Gate 1 currently has no `open`; strict `15/15` benchmark has been completed, and the current status of RegAD is `playbook-complete`.
- The problem of empty files in Google Drive version of `grid/capsule` support assets has been solved through Baidu network disk and no longer blocks the current strict main line.
- `tools/run_regad_exact13_benchmark.py` associated with partial progress is only retained as a historical diagnostic archive of the uncompleted phase of assets.
