# RD++ strict-alignment evidence

- **Method slug**: `rdpp`
- **Family**: Knowledge distillation
- **Method README**: [`configs/rdpp/README.md`](../../configs/rdpp/README.md)

This file preserves the fixed alignment evidence migrated from the manuscript evidence workspace. The content below is translated to English where needed and kept inside this repository so the strict-alignment state can be reviewed without relying on an external paper checkout.

## Config entry points

- [`configs/rdpp/rdpp_wrn50_256_mvtec_strict.py`](../../configs/rdpp/rdpp_wrn50_256_mvtec_strict.py)
- [`configs/rdpp/rdpp_wrn50_256_visa.py`](../../configs/rdpp/rdpp_wrn50_256_visa.py)

## Detailed alignment report

**Status**: `aligned`
**Date**: `2026-04-02`

## 1. Reference freezing

- Reference warehouse: `https://gh-proxy.com/https://github.com/tientrandinh/Revisiting-Reverse-Distillation`
- Reference commit: `7f2ceb7c87e602617b8600e1a498f7ef7f5247d6`
- Reference entrance:
  - `main.py`
  - `dataset/dataset.py`
  - `dataset/noise.py`
  - `model/resnet.py`
  - `model/de_resnet.py`
  - `utils/utils_train.py`
  - `utils/utils_test.py`
- Reference configuration caliber:
  - `image_size=256`
  -`batch_size=16`
  - `seed=111`
  -`proj_lr=1e-3`
  -`distill_lr=5e-3`
  - `weight_proj=0.2`
  - Each category is trained separately, and the epoch budget changes with the category
- BaoIAD strict configuration: `configs/rdpp/rdpp_wrn50_256_mvtec_strict.py`
- Historical legacy configuration: `configs/rdpp/rdpp_wrn50_256_mvtec.py`
- intentional diff:
  - The data root directory continues to use `data/mvtec_ad`, only path mapping is done, and the sample content is not changed.
  - strict config continues to use MMEngine runner/metric output, but the training logic and scoring semantics are aligned with the official

## 2. Code path comparison conclusion

See [`rdpp_checklist.md`](rdpp_checklist.md) for the control matrix.

### Consistency confirmed

- teacher feature extraction level is still WRN50 `layer1/2/3`
- `OCBE` / student decoder structure corresponds to the official implementation
- The image-level score still takes the `max` of the final anomaly map

### Inconsistencies fixed this round

- The strict path no longer temporarily generates approximate noise inside the detector and is changed to dataset-side official simplex noise
- The strict path is changed to the official `align_corners=True + scipy gaussian_filter(sigma=4)` anomaly map
- strict training is changed to projection / distillation two optimizers, and updated according to `accumulation_steps=2`
- Added official per-class epoch schedule, corresponding to `bottle=200`, `carpet=10`, `capsule=300`, etc.
- benchmark main configuration switches to `rdpp_wrn50_256_mvtec_strict.py`

### Closed and added

- `geomloss` dependency issues have been closed; strict training and full benchmark have all been run
- `runs/alignment/rdpp_strict_merged_progress.json` now covers `15/15`
- The current mainline conclusion is based on the complete strict `15/15` results, and the "sharding in progress" caliber is no longer retained

## 3. Behavior Probe

Plan command:

```bash
python tools/alignment_probe.py configs/rdpp/rdpp_wrn50_256_mvtec_strict.py \
    --splits train test \
    --output runs/alignment/rdpp_probe.json
```
Current conclusion:

- `geomloss` has been completed
- strict `alignment_probe` passed, output file: `runs/alignment/rdpp_probe.json`
- probe key statistics:
  - train sample contains `img_noise`, `shape=[3,256,256]`, with limited value range
  - strict loss output `loss=1.8878`, `loss_distill=1.8877`, `loss_proj=4.78e-4`
  - test predict output is limited, `pred_score=2.0443`

## 4. Small-scale controlled experiment

Plan an experiment:

- Category: `bottle`
- Training budget: strict official `1-5 epoch` smoke
- Comparison object: historical legacy path vs strict official path

Current decision:

- `pass`
- Actual test commands:

```bash
python tools/train.py configs/rdpp/rdpp_wrn50_256_mvtec_strict.py \
    --work-dir runs/alignment/rdpp_bottle_smoke_fast_e1_bs2 \
    --cfg-options \
        train_cfg.max_epochs=1 \
        train_cfg.category_epochs.bottle=1 \
        train_dataloader.batch_size=2 \
        optim_wrapper.projection.accumulative_counts=1 \
        optim_wrapper.distillation.accumulative_counts=1 \
        default_hooks.logger.interval=1 \
        default_hooks.checkpoint.interval=1
```
- Observation:
  - Training loss continued to decrease from `1.8884 -> 0.1896` without any shock or collapse.
  - `bottle` `1 epoch` cheap smoke Verification results:
    - `image_auroc=0.9976`
    - `pixel_auroc=0.9846`
    - `aupro=0.9583`
  - This smoke uses the cheaper `batch_size=2, accumulative_counts=1`, which is only used for Gate 3 direction verification and is not used as the final strict benchmark result.

## 5. Full Benchmark

Plan command:

```bash
python tools/benchmark.py \
    --data_root data/mvtec_ad \
    --methods rdpp \
    --categories all
```
**Final results (15/15 categories completed)**:

| Metric | Value |
|--------|-------|
| image_auroc | **0.9931** |
| pixel_auroc | **0.9821** |
| image_f1max | 0.9863 |
| image_ap | 0.9969 |
| aupro | 0.9452 |

**Per-category results**:

| Category | image_auroc | pixel_auroc | aupro |
|----------|-------------|-------------|-------|
| bottle | 1.0000 | 0.9876 | 0.9645 |
| cable | 0.9818 | 0.9826 | 0.9292 |
| capsule | 0.9828 | 0.9874 | 0.9225 |
| carpet | 1.0000 | 0.9922 | 0.9700 |
| grid | 1.0000 | 0.9931 | 0.9759 |
| hazelnut | 1.0000 | 0.9915 | 0.9270 |
| leather | 1.0000 | 0.9946 | 0.9853 |
| metal_nut | 1.0000 | 0.9801 | 0.9505 |
| pill | 0.9787 | 0.9834 | 0.9665 |
| screw | 0.9924 | 0.9964 | 0.9763 |
| tile | 0.9982 | 0.9639 | 0.8971 |
| toothbrush | 1.0000 | 0.9910 | 0.9317 |
| transistor | 0.9808 | 0.9490 | 0.8958 |
| wood | 0.9930 | 0.9559 | 0.9327 |
| zipper | 0.9895 | 0.9835 | 0.9527 |

Result file: `runs/alignment/rdpp_strict_merged_progress.json`

## 6. Guard

- Added strict transform / packing guard:
  - `GenerateRDPPNoise`
  - `PackRDPPInputs`
- Added strict runtime guard:
  - `RDPPOptimWrapperConstructor`
  - `RDPPTrainLoop`
  - `RDPPDetector.train_step()` multi-optimizer constraints
- Added targeted tests:
  - `tests/test_datasets/test_transforms.py`
  - `tests/test_models/test_detectors/test_rdpp.py`
  - `tests/test_engine/test_rdpp_optim_wrapper_constructor.py`
  - `tests/test_engine/test_rdpp_train_loop.py`

## 7. Residual Risk

- vendor's simplex noise currently comes with a pure Python fallback when `numba` is not available, the functionality is available but will be slower
- strict `15/15` completed; subsequent remaining risks are mainly operating environment and non-primary indicator warnings, rather than benchmark coverage gaps
- `image_ece / pixel_ece` will prompt that the input is not a probability in the current metric suite and will fall back to `0.0`. It will not affect the official alignment of the main indicators, but a warning will appear in the log.

## 8. Conclusion

- **Final decision**: strict official code path has been aligned, full benchmark 15/15 category has been completed
- **Alignment Status**: `aligned`
- **Main indicators**: image_auroc=0.9931, pixel_auroc=0.9821
- **Conclusion**: RD++ is strictly aligned and consistent with the official implementation

## Alignment checklist

Status only allowed: `matched | mismatch-fixed | intentional-diff | open`

## 1. Enter path

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| train color channel | official `dataset/dataset.py` | strict train pipeline `LoadImage(to_rgb=True)` | RGB input | `LoadImage` default BGR->RGB | matched |
| test color channel | official `dataset/dataset.py` | strict test pipeline `LoadImage(to_rgb=True)` | RGB input | same as above | matched |
| resize | official `cv2.resize(img/255., (256,256))` | `ResizeAD(size=256, backend='cv2')` + `ScaleNormalizeAD` | resize first, then normalize to `[0,1]` | strict config fixed | mismatch-fixed |
| normalization / value range | official `Normalize([0.485...], [0.229...])` | `NormalizeAD(mean=0.485..., std=0.229..., keys=...)` | official ImageNet normalize | strict config fixed | mismatch-fixed |

## 2. Noise Synthesis

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| simplex noise implementation | official `dataset/noise.py` | `baoiad/utils/rdpp_noise.py` | using official simplex / octave noise | vendored to the warehouse | mismatch-fixed |
| noise patch sampling | official `MVTecDataset_train.__getitem__` | `GenerateRDPPNoise` | `h/w ~ randint(10, size//8)`, patch injected into a single area | strict transform has been implemented according to the official formula | mismatch-fixed |
| noise injection position | official train dataset | `GenerateRDPPNoise` | generated after resize and before normalize `img_noise` | strict train pipeline fixed | mismatch-fixed |
| noisy branch transfer | official dataloader returns `(img, img_noise, ...)` | `PackRDPPInputs -> data_samples.img_noise` | noisy view explicitly enters batch | strict path single test coverage | mismatch-fixed |

## 3. Teacher / Student

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| teacher feature layer | official `model/resnet.py` | `extract_teacher_feats()` | using WRN50 `layer1/2/3` | detector code comparison | matched |
| BN / MFF_OCE | Official `BN_layer` | `OCBE` | Enter bottleneck block after three-scale fusion | detector code comparison | matched |
| decoder | official `de_resnet.py` | `StudentDecoder` | reversed WRN50 decoder | detector code comparison | matched |
| teacher freeze | official `encoder.eval()` | `train()` mandatory teacher eval | teacher does not participate in training | detector code comparison | matched |

## 4. Loss / Training

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| distill loss | official `loss_fucntion` | `_official_rdpp_loss()` | flattened cosine sum | detector code comparison | mismatch-fixed |
| projection loss | official `Revisit_RDLoss` | `Revisit_RDLoss` | sinkhorn + reconstruct + contrast | detector code comparison | matched |
| split optimizers | official `main.py` | `RDPPOptimWrapperConstructor` + `train_step()` | `proj Adam(1e-3)`, `distill Adam(5e-3)` | constructor / detector single test | mismatch-fixed |
| accumulation cadence | official `accumulation_steps=2` | strict `train_step()` + `accumulative_counts=2` | every 2 iter updates | detector single test | mismatch-fixed |
| per-class epochs | official `main.py` | `RDPPTrainLoop` | class-specific epoch budget | loop single test | mismatch-fixed |

## 5. Predict / Scoring

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| anomaly map source | official `utils_test.cal_anomaly_map` | `_strict_predict()` | multi-scale cosine map addition | detector code comparison | mismatch-fixed |
| interpolation | official `F.interpolate(..., align_corners=True)` | `_strict_predict()` | `align_corners=True` | detector code comparison | mismatch-fixed |
| smoothing | official `gaussian_filter(sigma=4)` | `_gaussian_blur_bchw()` | scipy gaussian sigma=4 | detector code comparison | mismatch-fixed |
| image score aggregation | official `max(anomaly_map)` | `_strict_predict()` | final map `max` | detector code comparison | matched |

## 6. Behavior verification conclusion

- [x] After fixing the seed, the dataset sample structure is as expected
- [x] noisy branch shape / value range verified by real probe
- [x] strict loss path verified by real probe
- [x] strict predict path verified by real probe
- [x] smoke / full benchmark does not trigger the shutdown line

## 7. Final alignment result

**Status**: `aligned`

| Metric | BaoIAD (15/15) | Official Reference |
|--------|------------------|-------------------|
| image_auroc | 0.9931 | ~0.99+ ✓ |
| pixel_auroc | 0.9821 | ~0.98+ ✓ |

**Completion date**: 2026-04-02
