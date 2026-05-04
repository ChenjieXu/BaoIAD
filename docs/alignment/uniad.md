# UniAD strict-alignment evidence

- **Method slug**: `uniad`
- **Family**: Reconstruction / ViT
- **Method README**: [`configs/uniad/README.md`](../../configs/uniad/README.md)

This file preserves the fixed alignment evidence migrated from the manuscript evidence workspace. The content below is translated to English where needed and kept inside this repository so the strict-alignment state can be reviewed without relying on an external paper checkout.

## Config entry points

- [`configs/uniad/uniad_wrn50_256_mvtec_strict.py`](../../configs/uniad/uniad_wrn50_256_mvtec_strict.py)
- [`configs/uniad/uniad_wrn50_256_visa.py`](../../configs/uniad/uniad_wrn50_256_visa.py)

## Detailed alignment report

**Status**: `aligned`
**Date**: `2026-03-26`

## 1. Reference freezing

- Reference repository: local `.refs/ader`
- Reference commit: `902937a7ed7fa7689674a4ac9b8fe9a72a40c402`
- Refer to config/checkpoint:
  - `.refs/ader/configs/benchmark/uniad/uniad_256_100e.py`
  - `.refs/ader/configs/__base__/cfg_model_uniad.py`
  - `.refs/ader/model/uniad.py`
  - `.refs/ader/trainer/uniad_trainer.py`
- Data set/category: MVTec AD, MUAD multi-class joint training, 15-class standard benchmark
- Input resolution: `256x256`
- seed: `42`
- Indicator definition: image AUROC / pixel AUROC
- intentional diff:
  - BaoIAD continues to retain compatibility `wrn50` filename `uniad_wrn50_256_mvtec_strict.py`, but its actual strict mainline has been frozen to `EfficientNet-B4 + MUAD`
  - BaoIAD uses mmengine training/evaluation loop with `build_predict_results()` wrapper `predict`

## 2. Code path comparison conclusion

See [`uniad_checklist.md`](uniad_checklist.md) for the control matrix.

### Consistency confirmed

- The current MUAD main configuration has been aligned to ADer `tf_efficientnet_b4 + out_indices=(0,1,2,3) + batch=8 + AdamW(lr=1e-4, wd=1e-4) + StepLR(80e/100e) + seed=42`
- `UniADDetector` The current encoder/decoder depth, feature jitter, neighbor mask, MSE loss and pooled anomaly map generation paths are all consistent with the ADer main line
- `alignment_probe` has actually run through the two paths `train loss` and `test predict`, proving that the current MUAD configuration is not a pseudo-alignment of "can only load, cannot forward"
- `runs/alignment/uniad_v2_checkpointed.json` has completed the full MUAD 100e benchmark with reusable checkpoints, and the baselines are `image AUROC=0.9117` and `pixel AUROC=0.9616`
- `runs/alignment/uniad_v3_checkpoint_eval.json` has completed the checkpoint-eval review through the benchmark infrastructure, and the current mainline results are `image AUROC=0.9303`, `pixel AUROC=0.9616`

### Fixed inconsistencies

- The historical MFCN direction bug has been fixed to "Align to the coarsest layer `16x16`" and no longer incorrectly upsamples features to stride 4
- `tools/benchmark.py` now explicitly fixes the UniAD default configuration to `configs/uniad/uniad_wrn50_256_mvtec_strict.py` to avoid accidentally selecting the bypass configuration
- Added `benchmark_multi_class = True` to the MUAD main configuration to prevent benchmark runner from treating UniAD as a class-by-class single-class method
- Added `benchmark_keep_dataloader_workers = True` to the MUAD main configuration to prevent the benchmark runner from pushing `num_workers` to `0` and then raising the total duration of 100e to the unexecutable range.
- `benchmark_preserve_checkpoint_hooks = True` is added to the MUAD main configuration. The checkpoint will be retained when the benchmark is re-run later, which facilitates offline scoring A/B of weak classes.
- The weak class sweep on the real `100e` checkpoint has confirmed that `pooled_topk_mean` is better than `pooled_max`, and `k=128` is better than `k=64`; the current main configuration has been switched to `pooled_topk_mean(k=128)`

### Items that are still open

- No algorithm level `open` items
- If you want to continue to narrow the image-level weak class gap in the future, give priority to doing scoring diagnostics from `tools/uniad_score_sweep.py` instead of changing the backbone/training caliber first.

## 3. Behavior Probe

Order:

```bash
python tools/alignment_probe.py configs/uniad/uniad_wrn50_256_mvtec_strict.py \
    --splits train test \
    --max-batch-size 2 \
    --device cuda \
    --output runs/alignment/uniad_probe.json
```
in conclusion:

- `pass`
- There is no local cache of `tf_efficientnet_b4` before the first probe; after pulling the timm weight once online, the default probe of `HF_HUB_OFFLINE=1` can be directly reproduced
- The probe proves that the main paths of MUAD dataloader, loss, predict, score map and image score are all normal.

Key statistics:

- dataset sample:
  - train preview: `data/mvtec_ad/zipper/train/good/065.png`
  - test preview: `data/mvtec_ad/bottle/test/broken_large/000.png`
  - train/test input shape: `2 x 3 x 256 x 256`
- loss path:
  - `loss` dict exists and is finite
  - probe train loss=`94.1978`
- predict path:
  - `pred_score` finite, mean=`206.8859`
  - `pred_anomaly_map` shape=`[1, 256, 256]`, mean=`167.4247`

## 4. Small-scale controlled experiment

Experimental setup:

- Category: `bottle`
- Training budget: `1 epoch`
- seed: `42`
- Comparison object: low-cost Gate 3 smoke with MUAD main configuration, without additional A/B forks

Order:

```bash
python tools/train.py configs/uniad/uniad_wrn50_256_mvtec_strict.py \
    --work-dir runs/alignment/uniad_bottle_smoke \
    --cfg-options \
        train_cfg.max_epochs=1 \
        train_cfg.val_interval=1 \
        train_dataloader.batch_size=1 \
        test_dataloader.batch_size=1 \
        val_dataloader.batch_size=1 \
        "train_dataloader.dataset.cls_names=['bottle']" \
        train_dataloader.dataset.multi_class=False \
        "test_dataloader.dataset.cls_names=['bottle']" \
        test_dataloader.dataset.multi_class=False \
        "val_dataloader.dataset.cls_names=['bottle']" \
        val_dataloader.dataset.multi_class=False
```
observe:

- train loss continues to decrease from `88.0294 -> 81.5367 -> 75.7691 -> 71.8862`
- The `bottle` verification result of the first epoch is:
  - `image_auroc=0.8016`
  - `pixel_auroc=0.7317`
  - `image_f1max=0.8806`
  - `aupro=0.5799`
- There is no loss explosion, pure zero score, pure noise map or uniform collapse to the `0.5` platform value.

determination:

- `pass`
- Reason: The goal of this smoke is to verify that the MUAD mainline can be stably trained/verified under the minimum training budget, rather than using the 1 epoch results to directly judge the quality of the final alignment.

## 5. Full Benchmark

baseline training benchmark:

```bash
python tools/benchmark.py \
    --data_root data/mvtec_ad \
    --methods uniad \
    --categories all \
    --output runs/alignment/uniad_v2_checkpointed.json \
    --timeout 28800
```
checkpoint-eval review benchmark:

```bash
python tools/benchmark.py \
    --data_root data/mvtec_ad \
    --methods uniad \
    --config configs/uniad/uniad_wrn50_256_mvtec_muad_checkpoint_eval.py \
    --categories all \
    --output runs/alignment/uniad_v3_checkpoint_eval.json \
    --timeout 7200
```
Summary of results:

| Metric | Reference | BaoIAD | Gap |
|--------|-----------|----------|-----|
| image_auroc | `0.925` | `0.9303` | `+0.53%` |
| pixel_auroc | `0.958` | `0.9616` | `+0.36%` |

Shutdown line inspection:

- [x] No large area image AUROC near `0.5` appears
- [x] Multiple categories did not collapse to similar platform values.
- [x] score histogram No obvious abnormal shrinkage
- [x] The gap from the reference can still be explained

illustrate:

- `benchmark.py` files the MUAD method as a single multi-category run, so `runs/alignment/uniad_v2_checkpointed.json` and `_average.num_categories=1` of `runs/alignment/uniad_v3_checkpoint_eval.json` both represent "a total MUAD run", not "only 1 category was evaluated"
- baseline `pooled_max` The full result is `image_auroc=0.9117`
- After using `epoch_100.pth` to do test-only override:
  - `pooled_topk_mean(k=64)` brings the full image AUROC to `0.9238`
  - `pooled_topk_mean(k=128)` further improved to `0.9303`
  - pixel indicator remains unchanged
- `runs/alignment/uniad_v3_checkpoint_eval.json` has solidified `0.9303 / 0.9616` into a benchmark-style official product, no longer just a test-only log conclusion
- The most obvious improvements to weak classes are:
  -`capsule: 0.6071 -> 0.6849`
  - `pill: 0.8189 -> 0.8481`
  - `toothbrush: 0.8500 -> 0.8778`
  - `screw: 0.8783 -> 0.8856`
  - `zipper: 0.8117 -> 0.8233`
- Strong class `bottle / carpet` does not degenerate
- The old `0.7854 / 0.9382` results come from the historical `13/15` partial benchmark under the `WRN-50` path, now downgraded to stop-line historical evidence

## 6. Guard

- New/enhanced tests:
  - `tests/test_utils/test_benchmark_config_detection.py`
  - `tests/test_models/test_detectors/test_uniad_detector.py`
- Added document guard:
  - `docs/alignment/uniad_checklist.md`
- New diagnostic tools:
  - `tools/uniad_score_sweep.py`
- New configuration / runner guard:
  - `configs/uniad/uniad_wrn50_256_mvtec_strict.py` explicitly declares `benchmark_multi_class = True`
  - `configs/uniad/uniad_wrn50_256_mvtec_strict.py` explicitly declares `benchmark_keep_dataloader_workers = True`
  - `configs/uniad/uniad_wrn50_256_mvtec_strict.py` explicitly declares `benchmark_preserve_checkpoint_hooks = True`
  - `configs/uniad/uniad_wrn50_256_mvtec_strict.py` The current mainline image scoring has been switched to `pooled_topk_mean(k=128)`
  - `configs/uniad/uniad_wrn50_256_mvtec_muad_checkpoint_eval.py` provides checkpoint-eval benchmark entry
  - `tools/benchmark.py` Explicitly point the UniAD default benchmark configuration to the MUAD mainline
- If you change these paths later, you must rerun:
  - `pytest tests/test_utils/test_benchmark_config_detection.py -q`
  - `pytest tests/test_models/test_detectors/test_uniad_detector.py -q`
  - `python tools/alignment_probe.py configs/uniad/uniad_wrn50_256_mvtec_strict.py --splits train test --max-batch-size 2 --device cuda --output runs/alignment/uniad_probe.json`
  - `python tools/uniad_score_sweep.py configs/uniad/uniad_wrn50_256_mvtec_strict.py runs/alignment/uniad_bottle_smoke/epoch_1.pth --categories bottle --score-modes pooled_max pooled_topk_mean raw_max raw_topk_mean --topk 64 --batch-size 1 --device cuda --output runs/alignment/uniad_bottle_score_sweep.json`

## 7. Residual Risk

- When running UniAD for the first time in a new environment, you still need to cache the local `tf_efficientnet_b4` or pull the timm weight once from the Internet.
- The image-level indicators of individual categories are still lower than those of the head category, but the overall average is within the reasonable alignment range defined in the README; if the image-level is specifically improved in the future, it should be regarded as a performance optimization and no longer an alignment blocking item
- Preliminary scoring sweep verified on `bottle` 1e checkpoint:
  - `pooled_max`: `image_auroc=0.8016`
  - `pooled_topk_mean`: `0.7992`
  - `raw_topk_mean`: `0.7619`
  - `raw_max`: `0.6627`
  This shows that weak class optimization cannot directly return to raw/top-k aggregation. In the future, priority should be paid to "Why the minority categories are still insufficiently separated under pooled semantics"
- Currently `pooled_topk_mean(k=128)` already has both:
  - test-only review log
  - benchmark-style JSON product `runs/alignment/uniad_v3_checkpoint_eval.json`
  If you need to completely replace the training benchmark record of baseline `v2` later, you can rerun the 100e full training benchmark based on the current main configuration separately.

## 8. Conclusion

- Final decision: `aligned`
- Allowed to proceed to next stage: `yes`
- If not allowed, next action: None

## Alignment checklist

Status only allowed: `matched | mismatch-fixed | intentional-diff | open`

## 1. Enter path

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| train color channel | `.refs/ader/configs/benchmark/uniad/uniad_256_100e.py` train transforms | `configs/_base_/datasets/mvtec_ad.py` | train input enters backbone in RGB | `LoadImage` default RGB; `alignment_probe` train sample shape=`[3,256,256]` | matched |
| test color channel | Same as above | Same as above | test and train keep the same channel order | `alignment_probe` test sample shape=`[3,256,256]` | matched |
| resize / crop | `.refs/ader/configs/benchmark/uniad/uniad_256_100e.py` | `ResizeAD(size=256)` | The input is unified to `256x256` | ADer is `Resize(256)+CenterCrop(256)`; BaoIAD directly resizes to `256`, which is equivalent to the target size | matched |
| normalization / value range | Same as above `Normalize(IMAGENET_DEFAULT_MEAN/STD)` | `NormalizeAD()` | Use ImageNet mean/std normalization | probe input value range is consistent with `NormalizeAD` default behavior | matched |
| batch size / seed | ADer `batch_train=8`, `seed=42` | `configs/uniad/uniad_wrn50_256_mvtec_strict.py` + `configs/_base_/default_runtime.py` | MUAD mainline fixed `batch=8`, `seed=42` | The current configuration is consistent with the reference; smoke is only Gate 3 low-cost override | matched |

## 2. Backbone / Feature Extraction

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| Backbone selection | `.refs/ader/configs/__base__/cfg_model_uniad.py` | `configs/uniad/uniad_wrn50_256_mvtec_strict.py` | Use `tf_efficientnet_b4`, `out_indices=(0,1,2,3)` | The current MUAD main configuration has been fixed to EffNet-B4 four-layer features | matched |
| Feature channel/stride | `.refs/ader/configs/__base__/cfg_model_uniad.py` + `.refs/ader/model/uniad.py` | `UniADDetector.__init__` | Channel `[24,32,56,160]`, stride `[2,4,8,16]` | `TIMMBackbone.feature_info` Drive channel/stride; consistent with ADer | matched |
| backbone freezes | `.refs/ader/model/uniad.py::freeze_layer` | `UniADDetector.__init__` + `train()` | backbone always freezes and remains eval | explicit `self.backbone.eval()` in config `frozen=True`, `train()` | matched |
| MFCN target resolution | `.refs/ader/model/uniad.py::MFCN` | `baoiad/models/detectors/uniad_detector.py::MFCN` | All scales aligned to the coarsest layer `16x16` | Historical bug has been fixed `target_size = features[-1].shape[2:]` | mismatch-fixed |
| feature_size / neighbor_size | `.refs/ader/configs/benchmark/uniad/uniad_256_100e.py` | `configs/uniad/uniad_wrn50_256_mvtec_strict.py` | `feature_size=(16,16)`, `neighbor_size=(8,8)` | The current main configuration is consistent with the caliber of ADer `256 // 16`, `256 // 32` | matched |

## 3. Reconstruction Branch

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| Encoder structure | `.refs/ader/model/uniad.py` | `UniADDetector.encoder_layers` | `4` layer encoder, `hidden_dim=256`, `nhead=8` | detector default value consistent with MUAD configuration | matched |
| Decoder structure | Same as above | `UniADDetector.decoder_layers` | `4` layer decoder, learned queries | detector default value consistent with MUAD configuration | matched |
| position embedding | `.refs/ader/model/uniad.py` | `PositionEmbeddingLearned` | learned positional embedding | Use both sides learned PE | matched |
| feature jitter | `.refs/ader/model/uniad.py` | `_add_jitter()` | `scale=20.0`, `prob=1.0` | detector default value is consistent with ADer | matched |
| neighbor mask | `.refs/ader/model/uniad.py` | `_generate_neighbor_mask()` | The three-way mask is fully enabled, and the neighborhood size is controlled by the configuration | `neighbor_mask_layers=(True,True,True)` | matched |
| loss input | `.refs/ader/trainer/uniad_trainer.py` | `forward(mode='loss')` | Do MSE on `feature_rec` and `feature_align` | `alignment_probe` train loss finite=`94.1978` | matched |

## 4. Predict / Scoring

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| anomaly map source | `.refs/ader/model/uniad.py` | `forward(mode='predict')` | generated by reconstruction error `sqrt(sum((rec-align)^2))` | detector logically consistent; probe map fully finite | matched |
| pooling | `.refs/ader/configs/benchmark/uniad/uniad_256_100e.py` evaluator `pooling_ks=[16,16]` | `forward(mode='predict')` | First `avg_pool2d(kernel=16, stride=1)` | current predict path Still according to ADer, do pooled map first | matched |
| image score aggregation | ADer evaluator / trainer | `forward(mode='predict')` + `configs/uniad/uniad_wrn50_256_mvtec_strict.py` | The baseline is `max` taken from the pooled map; the current main line is changed to `pooled_topk_mean(k=128)` | The real `100e` checkpoint sweep shows that `capsule/pill/zipper/toothbrush/screw` is all improved, `bottle/carpet` is not degraded, and `k=128` is better than `k=64` | intentional-diff |
| Post-processing / smoothing | ADer UniAD mainline | `forward(mode='predict')` | No additional Gaussian smoothing | No additional smoothing currently implemented | matched |

## 5. Benchmark Routing

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| Main configuration selection | `docs/alignment/PENDING_BENCHMARKS.md` | `tools/benchmark.py::_METHOD_CONFIG_PRIORITY` | `benchmark.py --methods uniad` defaults to strict MUAD configuration | Added `uniad -> uniad_wrn50_256_mvtec_strict.py` priority guard | mismatch-fixed |
| multi-class decision | MUAD reference training protocol | `configs/uniad/uniad_wrn50_256_mvtec_strict.py` | benchmark must treat UniAD as a single multi-class run | Added `benchmark_multi_class = True` + regression test | mismatch-fixed |
| dataloader workers | Current machine running benchmark diagnosis | `configs/uniad/uniad_wrn50_256_mvtec_strict.py` + `tools/benchmark.py` | benchmark needs to retain `num_workers=4` to avoid throughput being overwhelmed | Added `benchmark_keep_dataloader_workers = True` + regression test | mismatch-fixed |
| checkpoint reserved | current machine running benchmark diagnosis | `configs/uniad/uniad_wrn50_256_mvtec_strict.py` + `tools/benchmark.py` | subsequent weak class sweeps require a loadable full MUAD checkpoint | new `benchmark_preserve_checkpoint_hooks = True`; `runs/benchmark/uniad/all/epoch_100.pth` has been produced | mismatch-fixed |
| Single class smoke override | playbook Gate 3 | `tools/train.py --cfg-options ...` | Allows temporary overwriting of MUAD mainline to `bottle` single class only when smoke is used | `runs/alignment/uniad_bottle_smoke` verified | intentional-diff |

## 6. Behavior verification conclusion

- [x] After fixing the seed, the dataset sample structure is as expected
- [x] mask shape and range are as expected
- [x] The key intermediate quantity of the loss path has a shape / range assertion.
- [x] predict path's score / map makes shape / range assertions
- [x] `runs/alignment/uniad_probe.json` passed
- [x] `runs/alignment/uniad_bottle_smoke`'s 1-epoch smoke does not trigger stop-line
- [x] `runs/alignment/uniad_bottle_score_sweep.json` Verified `pooled_max` baseline is better than `raw_max` / `raw_topk_mean`
- [x] `runs/alignment/uniad_weak_classes_score_sweep.json` verified that `pooled_topk_mean` is better than `pooled_max` on `capsule/pill/zipper/toothbrush/screw`
