# Dinomaly strict-alignment evidence

- **Method slug**: `dinomaly`
- **Family**: Reconstruction / ViT
- **Method README**: [`configs/dinomaly/README.md`](../../configs/dinomaly/README.md)

This file preserves the fixed alignment evidence migrated from the manuscript evidence workspace. The content below is translated to English where needed and kept inside this repository so the strict-alignment state can be reviewed without relying on an external paper checkout.

## Config entry points

- [`configs/dinomaly/dinomaly_392_mvtec_strict.py`](../../configs/dinomaly/dinomaly_392_mvtec_strict.py)
- [`configs/dinomaly/dinomaly_392_visa.py`](../../configs/dinomaly/dinomaly_392_visa.py)

## Detailed alignment report

**Status**: `aligned`
**Date**: `2026-04-06`

## 1. Reference freezing

- Reference repository: `.refs/Dinomaly_official` (`https://gh-proxy.com/https://github.com/guojiajeremy/Dinomaly`)
- Reference commit: `c5c76d01a2bd7212f1c4b7dfdad14902d0f48cfe`
- Reference entry: `dinomaly_mvtec_uni.py`
- Dataset/Category: MVTec AD, 15 classes unified/multi-class training
- Input resolution: `Resize(448) -> CenterCrop(392)`
- seed: `1`
- Indicator definition:
  - train-time eval uses `evaluation_batch(..., max_ratio=0.01, resize_mask=256)`
  - anomaly map: multi-layer cosine map mean
  - image score: Gaussian(5, sigma=4) and then take top-1% mean
- intentional diff:
  - Currently smoke is used for resource verification in a shared GPU environment, using `batch_size=4` to run the `bottle` single-class subset; the strict main configuration itself is still frozen in the official unified `batch_size=16`

### Freeze parameters

- backbone: `dinov2reg_vit_base_14`
- target layers: `[2, 3, 4, 5, 6, 7, 8, 9]`
- fuse layers: `[[0,1,2,3], [4,5,6,7]]` for encoder / decoder
- bottleneck / decoder: dropout `0.2`, decoder depth `8`
- optimizer: `StableAdamW(lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)`
- scheduler: warmup + cosine，`warmup_iters=100`, `total_iters=10000`, `final_lr=2e-4`
- grad clip: `max_norm=0.1`
- loss: `global_cosine_hm_percent`, `p=min(0.9 * it / 1000, 0.9)`, `factor=0.1`

## 2. Code path comparison conclusion

See [`dinomaly_checklist.md`](dinomaly_checklist.md) for the control matrix.

### Consistency confirmed

- The backbone structure, target layer, and fuse layer are consistent with the official `ViTill`
- Input preprocessing has been frozen to `448 -> 392`, ImageNet normalize, RGB
- The unified training protocol has been switched to the `multi_class=True` category 15 mainline
- image score is now top-1% mean and calculated on `256x256` smoothed plots

### Fixed inconsistencies

- The old `dinomaly_256_mvtec.py` caliber is `256`, `5000 iters`, a single class path that favors anomalib; strict has been changed to the official MUAD unified
- `DinomalyDetector.predict` previously returned an anomaly map without post-processing the original image size; now it has been changed to return a `resize_mask=256 + Gaussian` map consistent with official eval
- Official `StableAdamW` / `WarmCosineScheduler` are registered to BaoIAD as `StableAdamW` / `WarmCosineLR`

### Items that are still open

- none

## 3. Behavior Probe

Order:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/alignment_probe.py configs/dinomaly/dinomaly_392_mvtec_strict.py \
    --splits train test \
    --max-batch-size 2 \
    --device cuda \
    --output runs/alignment/dinomaly_probe.json
```
in conclusion:

- `passed=true`
- The train/test dataloader, loss, and predict paths can all be constructed normally.
- predict output map shape has been aligned to `256x256`

Key statistics:

- dataset sample: train/test are both `392x392` input; train preview is `zipper/good`, test preview is `carpet/color`
- loss path: single batch `loss=1.0056`, limited value
- predict path: `pred_score` finite; `pred_anomaly_map` shapes `[1, 256, 256]`, `min=0.9349`, `max=1.0437`, `std=0.0145`

## 4. Small-scale controlled experiment

Experimental setup:

- Category: `bottle`
- Training budget: `53` iters, about 1 data pass; `val_begin=53`, `val_interval=53`
- Equipment: `GPU2`
- Resource adjustment: Temporarily reduce smoke `batch_size` to `4` in a shared GPU environment

Order:

```bash
CUDA_VISIBLE_DEVICES=2 python tools/train.py configs/dinomaly/dinomaly_392_mvtec_strict.py \
    --work-dir runs/alignment/dinomaly_bottle_smoke_gpu2_b4 \
    --cfg-options \
        train_cfg.max_iters=53 \
        train_cfg.val_begin=53 \
        train_cfg.val_interval=53 \
        default_hooks.logger.interval=10 \
        train_dataloader.batch_size=4 \
        test_dataloader.batch_size=4 \
        val_dataloader.batch_size=4 \
        train_dataloader.dataset.cls_names="['bottle']" \
        test_dataloader.dataset.cls_names="['bottle']" \
        val_dataloader.dataset.cls_names="['bottle']" \
        train_dataloader.dataset.multi_class=False \
        test_dataloader.dataset.multi_class=False \
        val_dataloader.dataset.multi_class=False
```
observe:

- loss curve: `0.6389 -> 0.3169 -> 0.2058 -> 0.1507 -> 0.1193`, steadily decreasing, no NaN / divergence
- image score / indicator: val `image_auroc=1.0000`, `pixel_auroc=0.9851`, `aupro=0.9646`
- anomaly map: smoke checkpoint probe shows `pred_anomaly_map` is still `[1,256,256]`, `min=0.9332`, `max=1.0752`, `std=0.0204`, not all zeros or all bright

determination:

- `pass`
- Reason: Both training and prediction links are normal, and the score does not appear 0.5 platform or collapse

## 5. Full Benchmark

`2026-04-06` fresh unified `15/15` full benchmark completed:

```bash
CUDA_VISIBLE_DEVICES=3 python tools/benchmark.py \
    --data_root data/mvtec_ad \
    --categories all \
    --methods dinomaly \
    --device cuda \
    --output runs/alignment/dinomaly_392_muad_strict_fresh.json \
    --work-dir-root runs/benchmark_dinomaly_392_muad_strict_fresh
```
Current conclusion:

- The code and strict configuration have been switched to the official MUAD unified mainline
- `probe / bottle smoke` passed
- fresh unified `15/15` benchmark has been completed and the result file is `runs/alignment/dinomaly_392_muad_strict_fresh.json`

Summary of results:

| Metric | Archived official unified | Fresh strict rerun | Gap |
|--------|---------------------------|--------------------|-----|
| image_auroc | `0.9962` | `0.9960` | `-0.0002` |
| pixel_auroc | `0.9826` | `0.9830` | `+0.0004` |
| image_ap | `-` | `0.9979` | `-` |
| pixel_ap | `-` | `0.6522` | `-` |
| aupro | `-` | `0.9484` | `-` |

in conclusion:

- fresh strict rerun vs. archived official unified comparison falls within `±0.001` on both `image/pixel`
- Dinomaly strict392 can now be considered as closed

## 6. Guard

- New test:
  - `tests/test_utils/test_benchmark_config_detection.py::test_dinomaly_strict_benchmark_is_multi_class`
  - `tests/test_utils/test_benchmark_config_detection.py::test_dinomaly_strict_config_freezes_official_muad_hparams`
- Added assertion:
  - strict config must point to `dinomaly_392_mvtec_strict.py`
  - strict config must be `benchmark_multi_class=True`
  - strict config must freeze `448 -> 392`, `StableAdamW`, `WarmCosineLR`, `max_iters=10000`

## 7. Residual Risk

- The official `batch_size=16` smoke in the shared GPU environment encountered competition for external video memory for the first time; however, the fresh unified full rerun has been completed normally under the official batch, so this is only retained as an environment note and will no longer block the closure.

## 8. Conclusion

- Final judgment: `configs/dinomaly/dinomaly_392_mvtec_strict.py` has been strictly aligned according to the official MUAD unified mainline
- Allow to proceed to next stage: Yes
- Next action: None; subsequent reruns will only occur when the implementation path changes again `probe / bottle smoke / fresh full benchmark`

## Alignment checklist

Status only allowed: `matched | mismatch-fixed | intentional-diff | open`

## 1. Enter path

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| train color channel | `dataset.py:get_data_transforms` | `configs/dinomaly/dinomaly_392_mvtec_strict.py` + `LoadImage` | RGB + ImageNet normalize | official `Image.open(...).convert('RGB')`; strict pipeline frozen `NormalizeAD` | matched |
| test color channel | `dataset.py:get_data_transforms` | Same as above | RGB + ImageNet normalize | Same as above | matched |
| resize/crop | `Resize(448) -> CenterCrop(392)` | `ResizeAD(size=448, official_pil=True) -> CenterCrop(392)` | full alignment | strict config explicitly frozen | mismatch-fixed |
| normalization / value range | torchvision `ToTensor() + Normalize(ImageNet)` | `NormalizeAD` | `[0,255] -> [0,1] -> ImageNet normalize` | probe input statistics are normal | matched |

## 2. Backbone / Decoder

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| encoder structure | `dinomaly_mvtec_uni.py` + `models/uad.py` | `baoiad/models/backbones/dinomaly_backbone.py` | `dinov2reg_vit_base_14` | strict config / detector alignment | matched |
| target layers | `target_layers=[2..9]` | `configs/dinomaly/dinomaly_392_mvtec_strict.py` | complete and consistent | config assertion complemented | matched |
| fuse layers | `[[0,1,2,3],[4,5,6,7]]` | `DinomalyDetector` | complete consistency | detector default/strict explicit consistency | matched |
| bottleneck / decoder | `dropout=0.2`, decoder depth `8` | `DinomalyDetector` | complete and consistent | strict config explicit freeze | matched |

## 3. Loss

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| loss formula | `utils.py:global_cosine_hm_percent` | `CosineHardMiningLoss` | global cosine + hard mining grad mask | consistent with the comparison formula | matched |
| p scheduling | `p=min(0.9*it/1000,0.9)` | `loss_p_final=0.9`, `loss_schedule_steps=1000` | complete consistency | strict config explicit freeze | mismatch-fixed |
| hard-mining factor | `factor=0.1` | `loss_factor=0.1` | complete consistency | strict config explicit freeze | mismatch-fixed |
| reduction | level mean | `loss / len(encoder_features)` | complete and consistent | code comparison | matched |

## 4. Optimizer / Scheduler

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| optimizer type | `optimizers/StableAdamW.py` | `baoiad/engine/optimizers/stable_adamw.py` | Official StableAdamW | New registration implementation | mismatch-fixed |
| optimizer parameters | `lr=2e-3`, `wd=1e-4`, `amsgrad=True`, `eps=1e-10` | strict config | complete and consistent | config assertions complemented | mismatch-fixed |
| scheduler | `WarmCosineScheduler` | `baoiad/engine/schedulers/warm_cosine_scheduler.py` | warmup 100 + cosine to `2e-4` | New registration implementation | mismatch-fixed |
| grad clip | `clip_grad_norm_(..., 0.1)` | `optim_wrapper.clip_grad.max_norm=0.1` | complete and consistent | strict config | matched |

## 5. Predict / Scoring

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| anomaly map source | `cal_anomaly_maps` | `DinomalyDetector._calculate_anomaly_maps` | multi-layer cosine map mean | consistent formula | matched |
| resize_mask | `evaluation_batch(..., resize_mask=256)` | `predict_map_size=256` + evaluator `resize_mask=256` | official MVTec caliber | strict config / probe map shape `[1,256,256]` | mismatch-fixed |
| smoothing | Gaussian kernel `5`, sigma `4` | `GaussianBlur2d(5, 4)` | complete and consistent | strict config / probe map stats | matched |
| image score aggregation | top-1% mean | `image_score_max_ratio=0.01` | complete and consistent | smoke / probe output is normal | mismatch-fixed |

## 6. Training protocol

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| mainline protocol | `dinomaly_mvtec_uni.py` | `dinomaly_392_mvtec_strict.py` | class 15 unified / MUAD | `benchmark_multi_class=True` | mismatch-fixed |
| training budget | `10000` iters | `train_cfg.max_iters=10000` | complete and consistent | config assertions complemented | mismatch-fixed |
| batch size | `16` | `train_dataloader.batch_size=16` | Complete and consistent | config assertion complemented | mismatch-fixed |
| few-shot / multi-stage | None | None | No additional protocols introduced | Reference code comparison | matched |

## 7. Official vs Anomalib deviation

| Projects | Official Dinomaly | Anomalib / Old BaoIAD Mainline | Notes | Status |
|------|---------------|--------------------------------|------|------|
| Main settings | MUAD unified | Single class / 256 / 5000 iters | strict must be based on the official version | mismatch-fixed |
| optimizer / scheduler | StableAdamW + WarmCosine | The historical path does not explicitly freeze the official implementation | This round has been compensated | mismatch-fixed |
| predict caliber | resize_mask=256 post-smoothing and aggregation | old detector returns unpost-processed map | fixed this round | mismatch-fixed |

## 8. Behavior verification conclusion

- [x] After fixing the seed, the dataset sample structure is as expected
- [x] The key intermediate quantity of the loss path is a finite value
- [x] predict path's score / map makes shape / range assertions
- [x] smoke loss is not divergent, and `bottle` image AUROC is not close to 0.5
- [x] anomaly map is not all zeros or all brights
