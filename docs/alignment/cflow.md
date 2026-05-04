# CFlow strict-alignment evidence

- **Method slug**: `cflow`
- **Family**: Normalizing flow
- **Method README**: [`configs/cflow/README.md`](../../configs/cflow/README.md)

This file preserves the fixed alignment evidence migrated from the manuscript evidence workspace. The content below is translated to English where needed and kept inside this repository so the strict-alignment state can be reviewed without relying on an external paper checkout.

## Config entry points

- [`configs/cflow/cflow_mvtec_strict.py`](../../configs/cflow/cflow_mvtec_strict.py)
- [`configs/cflow/cflow_visa.py`](../../configs/cflow/cflow_visa.py)

## Detailed alignment report

**Status**: `playbook-complete`
**Date**: `2026-03-29`

## 1. Reference freezing

- Reference warehouse: `.refs/cflow-ad`
- Reference commit: `b2ebf9e673a0aa46992a3b18367ec066a57bba89`
- Refer to config/checkpoint:
  - Training entrance: `.refs/cflow-ad/main.py`
  - Training logic: `.refs/cflow-ad/train.py`
  - Model definition: `.refs/cflow-ad/model.py`
  - MVTec reference command: `.refs/cflow-ad/README.md`
- Dataset/Category: MVTec AD, single-class training/single-class testing, `15/15`
- Input resolution:
  - `512`: `bottle / carpet / grid / leather / screw / tile / toothbrush / wood / zipper`
  - `256`: `cable / capsule / hazelnut / metal_nut / pill`
  - `128`: `transistor`
- seed:
  - Official warehouse default `time.time()` dynamic seed
  - BaoIAD strict mainline fixed `seed=42`, as the only `intentional-diff`, used for reviewable benchmark
- Indicator definition:
  - official training logs are recorded separately according to the historical optimal `DET_AUROC / SEG_AUROC / SEG_AUPRO`
  - BaoIAD strict benchmark is aligned to `benchmark_result_selector = best_per_metric`
- intentional diff:
  - Fixed `seed=42`
  - Still output indicators through MMEngine / BaoIAD evaluator, and do not directly reuse the text result file of the official script

## 2. Code path comparison conclusion

See [`cflow_checklist.md`](cflow_checklist.md) for the control matrix.

### Consistency confirmed

- The detector's per-fiber train loss, test-time remainder fiber processing, multi-layer anomaly-map aggregation and image score aggregation are consistent with the official main logic.
- The strict mainline has been switched back to the `layer2/layer3/layer4` path of torchvision `wide_resnet50_2`.

### Fixed inconsistencies

- Added strict configuration [`configs/cflow/cflow_mvtec_strict.py`](../../configs/cflow/cflow_mvtec_strict.py), and no longer regard the old unified `256` configuration as the official mainline.
- Added `CFlowOfficialTransform` to solidify the official README's category-level input size and `Resize -> RandomRotation(5) -> CenterCrop` sequence into the pipeline.
- Added `CFlowOfficialTrainLoop` to reproduce the official `25 meta-epochs x 8 sub-epochs`, cosine + warmup training rhythm.
- The strict mainline changes `permute_soft` back to official `True`, and changes `fiber_batch_size` back to official `256`.
- The strict path adds `.refs/cflow-ad` asset guard, which will directly fail when the reference warehouse is missing.

### Items that are still open

- No new blocker; strict `15/15` full benchmark completed

## 3. Behavior Probe

Order:

```bash
python tools/alignment_probe.py configs/cflow/cflow_mvtec_strict.py \
    --splits train test \
    --max-batch-size 2 \
    --output runs/alignment/cflow_probe_strict.json
```
in conclusion:

- strict `alignment_probe` passed and filed into `runs/alignment/cflow_probe_strict.json`.

Key statistics:

- dataset sample:
  - train/test `bottle` sample shapes are all `[3, 512, 512]`
  - train is a normal sample, test is an abnormal sample, and the mask shape is `[512, 512]`
- loss path:
  - `train.loss = 26.7411`
  - loss is limited, and `keys=['loss']`
- predict path:
  - `pred_score mean = 0.8024`
  - `pred_anomaly_map mean = 0.2169`
  - score/map both exist and are limited

## 4. Small-scale controlled experiment

Experimental setup:

- Category: `bottle`
- Training budget:
  - smoke main run: `1 meta-epoch`, `sub_epochs=1`
  - Evaluation items: `image_auroc`, `pixel_auroc`
- seed: `42`
- Comparison objects:
  - strict `configs/cflow/cflow_mvtec_strict.py`
  - Low-cost smoke is only used for Gate 3 direction verification and does not replace the full benchmark

observe:

- work dir: `runs/alignment/cflow_strict_bottle_smoke_v4`
- summary: `runs/alignment/cflow_strict_bottle_smoke_v4_summary.json`
- checkpoint: `runs/alignment/cflow_strict_bottle_smoke_v4/epoch_1.pth`
- loss curve:
  - `0.7894 -> 0.4080` (steady decline within a single sub-epoch)
- image score distribution:
  - `image_auroc = 0.9817`
- pixel map:
  - `pixel_auroc = 0.9760`

determination:

- `pass`
- Reason: The strict mainline has been able to complete low-cost bottle training and verification, and the image/pixel indicators are at reasonably high values, with no sign of stop-line.

## 5. Full Benchmark

**Completed** (2026-03-29)

Result file: `runs/alignment/cflow_v2.json`

| Metric | BaoIAD | Official | Gap |
|--------|----------|----------|-----|
| image_auroc | **0.9716** | 0.976 | -0.0044 |
| pixel_auroc | **0.9827** | 0.980 | +0.0027 |

### Detailed results for each category

| category | image_auroc | pixel_auroc |
|------|-------------|-------------|
| bottle | 1.0000 | 0.9886 |
| cable | 0.9584 | 0.9718 |
| capsule | 0.9270 | 0.9888 |
| carpet | 0.9805 | 0.9759 |
| grid | 0.9713 | 0.9665 |
| hazelnut | 0.9855 | 0.9842 |
| leather | 0.9975 | 0.9920 |
| metal_nut | 0.9744 | 0.9825 |
| pill | 0.9658 | 0.9803 |
| screw | 0.9548 | 0.9863 |
| tile | 0.9894 | 0.9652 |
| toothbrush | 0.9597 | 0.9871 |
| transistor | 0.9652 | 0.9810 |
| wood | 0.9908 | 0.9872 |
| zipper | 0.9915 | 0.9886 |

Shutdown line inspection:

- [x] No large area image AUROC near 0.5 appears
- [x] Multiple categories did not collapse to similar platform values.
- [x] score histogram No obvious abnormal shrinkage
- [x] is within ±0.01 of the reference

## 6. Guard

- New test:
  - `tests/test_models/test_detectors/test_cflow.py`
  - `tests/test_datasets/test_cflow_transform.py`
  - `tests/test_utils/test_cflow_benchmark.py`
- Added strict code path:
  - `baoiad/datasets/transforms/cflow.py`
  - `baoiad/engine/loops/cflow_train_loop.py`
  - `configs/cflow/cflow_mvtec_strict.py`
- If you change these paths later, you must rerun:
  - strict `alignment_probe`
  - strict `bottle` smoke
  - strict `15/15` benchmark

## 7. Residual Risk

- The old legacy configuration remains in the repository; if the benchmark configuration priority is changed back, the strict conclusion will be polluted again.
- The official warehouse defaults to dynamic seed, while the BaoIAD strict mainline is fixed to `seed=42`; this is the currently reserved intentional diff.

## 8. Conclusion

- Final decision: `playbook-complete`
- Alignment results: image_auroc=0.9716 (official 0.976, difference -0.0044), pixel_auroc=0.9827 (official 0.980, difference +0.0027)
- The differences are all within ±0.01 and are considered aligned.

## Alignment checklist

Status only allowed: `matched | mismatch-fixed | intentional-diff | open`

## 1. Enter path

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| train / test color channel | `.refs/cflow-ad/custom_datasets/loader.py:MVTecDataset.__getitem__` | `LoadImage(color_type='color')` | 3-channel RGB before input backbone | OpenCV `IMREAD_COLOR` + `BGR->RGB`; grayscale class will also be expanded into 3-channel | `matched` |
| Category level input size | `.refs/cflow-ad/README.md` + `.refs/cflow-ad/main.py` | `configs/cflow/cflow_mvtec_strict.py` + `CFlowOfficialTransform` | `bottle=512`, `cable/capsule/hazelnut/metal_nut/pill=256`, `transistor=128`, the rest `512` | strict config solidification `cflow_input_size_map` | `mismatch-fixed` |
| resize / crop order | `.refs/cflow-ad/custom_datasets/loader.py` | `CFlowOfficialTransform(train={True,False})` | train: `Resize -> RandomRotation(5) -> CenterCrop`; test: `Resize -> CenterCrop` | New strict special transform | `mismatch-fixed` |
| normalization / value range | `.refs/cflow-ad/main.py` + `.refs/cflow-ad/custom_datasets/loader.py` | `NormalizeAD` | MVTec uses ImageNet mean/std | strict config continues to use `NormalizeAD` default statistics | `matched` |

## 2. Model and training

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| backbone source | `.refs/cflow-ad/model.py:load_encoder_arch` | strict config `FeatureExtractor(backbone_name='wide_resnet50_2')` | Use torchvision `wide_resnet50_2` to pre-train backbone | strict config no longer goes `TIMMBackbone` | `mismatch-fixed` |
| feature layers | `.refs/cflow-ad/model.py:load_encoder_arch` | `FeatureExtractor(out_indices=(2,3,4))` | take `layer2/layer3/layer4` | strict config solidify `out_indices=(2,3,4)` | `matched` |
| conditional flow blocks | `.refs/cflow-ad/model.py:freia_cflow_head` | `build_cflow_head()` | 8 `AllInOneBlock`, `SOFTPLUS` affine | detector consistent with official structure | `matched` |
| `permute_soft` | `.refs/cflow-ad/model.py:freia_cflow_head` | strict config `permute_soft=True` | official uses `permute_soft=True` | old legacy config is `False`; strict mainline has been corrected | `mismatch-fixed` |
| fiber batch size | `.refs/cflow-ad/train.py` | strict config `fiber_batch_size=256` | official `N=256` | old legacy config is `64`; strict mainline has been corrected | `mismatch-fixed` |
| optimizer | `.refs/cflow-ad/train.py` | `optim_wrapper.optimizer` | Adam, `lr=2e-4`, `weight_decay=0` | strict config solidification | `matched` |
| Training rhythm | `.refs/cflow-ad/train.py:train_meta_epoch` | `CFlowOfficialTrainLoop` | `25 meta-epochs x 8 sub-epochs`, verified once for each meta-epoch | New strict dedicated train loop | `mismatch-fixed` |
| cosine + warmup lr | `.refs/cflow-ad/main.py` + `.refs/cflow-ad/custom_models/utils.py` | `CFlowOfficialTrainLoop` | `warmup_ratio=0.1`, `warmup_meta_epochs=2`, cosine over `25` meta-epochs | strict loop reproduces the official formula | `mismatch-fixed` |

## 3. Loss and Predict / Scoring

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| per-fiber optimization | `.refs/cflow-ad/train.py` | `CFlowDetector.train_step()` | Each fiber batch alone `zero_grad -> backward -> step` | The detector still keeps per-fiber updates | `matched` |
| train-time remainder handling | `.refs/cflow-ad/train.py` | `CFlowDetector.train_step()` | Discard the remainder during training fiber | `FIB = E // N` Consistent with the official | `matched` |
| test-time remainder handling | `.refs/cflow-ad/train.py:test_meta_epoch` | `CFlowDetector.forward(mode='predict')` | Keep the last batch of fiber during inference | `FIB = E // N + int(E % N > 0)` | `matched` |
| anomaly map aggregation | `.refs/cflow-ad/train.py` | `CFlowDetector.forward(mode='predict')` | `exp(log_prob - max)` -> upsample -> multi-layer sum -> invert | The existing scoring path of the detector is consistent with the official one | `matched` |
| image score aggregation | `.refs/cflow-ad/train.py` | `pred_score = max(super_mask)` | image score from anomaly map maximum | detector using `score_map.view(...).max()` | `matched` |

## 4. Guard

- [x] backbone / schedule / scorer default values of strict config have been fixed by single test
- [x] strict config fails explicitly when assets are missing
- [x] strict `alignment_probe` archived
- [x] strict `bottle` smoke archived
- [ ] strict `15/15` benchmark archived
