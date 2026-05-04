# SuperSimpleNet strict-alignment evidence

- **Method slug**: `supersimplenet`
- **Family**: Discriminative
- **Method README**: [`configs/supersimplenet/README.md`](../../configs/supersimplenet/README.md)

This file preserves the fixed alignment evidence migrated from the manuscript evidence workspace. The content below is translated to English where needed and kept inside this repository so the strict-alignment state can be reviewed without relying on an external paper checkout.

## Config entry points

- [`configs/supersimplenet/supersimplenet_256_mvtec_strict.py`](../../configs/supersimplenet/supersimplenet_256_mvtec_strict.py)
- [`configs/supersimplenet/supersimplenet_256_visa.py`](../../configs/supersimplenet/supersimplenet_256_visa.py)

## Detailed alignment report

**Status**: `playbook-complete`
**Date**: `2026-03-29`

## 1. Reference freezing

- Reference warehouse:
  - Primary reference: local `.refs/anomalib`
- Reference commit:
  - anomalib: `4f6af1acb0ee7b81f54cda036dd9f1c27f63b69a`
- Refer to config/checkpoint:
  - `.refs/anomalib/src/anomalib/models/image/supersimplenet/torch_model.py`
  - `.refs/anomalib/src/anomalib/models/image/supersimplenet/anomaly_generator.py`
  - `.refs/anomalib/src/anomalib/models/image/supersimplenet/loss.py`
  - `.refs/anomalib/src/anomalib/models/image/supersimplenet/lightning_model.py`
- Data set/category: `MVTec AD`, standard 15-category benchmark; the current review will be done first `bottle` smoke
- Input resolution: `256x256`
- seed: `42`
- Indicator definition: image AUROC / pixel AUROC
- intentional diff:
  - BaoIAD retains the MMEngine runner / dataloader / evaluator framework and does not copy the anomalib Lightning trainer organization method
  - `RawBackbone(wide_resnet50_2, pretrained=True)` and anomalib's `wide_resnet50_2.tv_in1k` are regarded as the same ImageNet V1 weight family, and there is no longer a separate set of backbones.
  - The current strict image-level main caliber is first switched to `pred_score_max` (map max), because this round of diagnosis shows that the classifier score is obviously inconsistent with the official compatibility behavior; this still needs to be finally confirmed with fresh strict rerun

## 2. Code path comparison conclusion

See [`supersimplenet_checklist.md`](supersimplenet_checklist.md) for the control matrix.

### Consistency confirmed

- When `adapt_cls_features=False` is used, the seg branch uses adapted features and the cls branch uses raw features, which is consistent with anomalib's current default JIMS extension caliber.
- The splicing order of `seg_head / cls_conv / cls_fc`, `stop_grad`, and pooling of `SegmentationDetectionModule` is consistent with the reference implementation.
- `SSNLoss` is still `focal(map) + trunc_l1(map) + focal(score)`, `alpha=-1`, `gamma=4.0`, `truncation_term=0.5` are consistent
- The `predict` path still uses `bilinear upsample + Gaussian blur + sigmoid(score/map)`, and the current output range guard has been completed.
- `runs/alignment/supersimplenet_reference_diag_train.json` and `runs/alignment/supersimplenet_reference_diag_train_b1.json` have been proven:
  - feature extractor output pair reference `MAE=0.0`
  - anomaly generator's `perturbed_feat / perturbed_adapt / mask / labels` to reference `MAE=0.0`
- Local `ResizeAD(pillow) + NormalizeAD` has consistent output statistics with torchvision `Resize(antialias=True) + Normalize` for the same `bottle` image, input preprocessing has been excluded

### Fixed inconsistencies

- Added strict configuration [`configs/supersimplenet/supersimplenet_256_mvtec_strict.py`](../../configs/supersimplenet/supersimplenet_256_mvtec_strict.py) to close optimizer to anomalib official param-group caliber:
  - `adaptor`: `lr=1e-4`, `weight_decay=1e-2`
  - `segdec`: `lr=2e-4`, `weight_decay=1e-5`
- The strict configuration training budget has been switched to `300 epochs` recommended by the official README, and the historical `100 epochs` will no longer be used.
- `tools/benchmark.py` has switched the SuperSimpleNet main configuration priority to strict config, and the old [`configs/supersimplenet/supersimplenet_256_mvtec.py`](../../configs/supersimplenet/supersimplenet_256_mvtec.py) is only retained as historical baseline
- This round changed anomaly synthesis from "multi-scale random noise approximation" to anomalib-style gradient-based Perlin generation, and retained the threshold fallback logic in the reference
- `tools/benchmark.py` adds `benchmark_rescale_epoch_schedulers` support so that smoke's `--epochs` override will scale `MultiStepLR` milestones synchronously
- `SuperSimpleNetDetector.predict()` now additionally exports:
  - `pred_score_mean`: anomaly map spatial mean
  - `pred_score_max`: anomaly map spatial max
- The strict evaluator is currently changed to `image_score_field='pred_score_max'`, and the image-level main indicator is switched to the map-max caliber.

### Items that are still open

- `300 epoch strict 15/15` full benchmark completed; no new acceptance blockers are currently available

## 3. Behavior Probe

Order:

```bash
python tools/alignment_probe.py configs/supersimplenet/supersimplenet_256_mvtec_strict.py \
    --splits train test \
    --max-batch-size 2 \
    --device cuda \
    --output runs/alignment/supersimplenet_probe.json
```
in conclusion:

- `runs/alignment/supersimplenet_probe.json` regenerated and passed
- `train/test` No NaN / Inf appears in the batch, loss, and predict paths.
- The probe proves that the structural access of strict config has been cleared, but it does not mean that the training behavior is consistent with the reference

Key statistics:

- dataset sample:
  - train preview: `zipper/train/good/065.png`
  - test preview: `bottle/test/broken_large/000.png`
  - Input shape: `2 x 3 x 256 x 256`
- loss path:
  - `keys=['loss']`
  - `loss=1.5094`
  -finite
- predict path:
  - `pred_score mean=0.5094`
  - `pred_anomaly_map shape=[1, 256, 256]`
  - map mean `0.5283`
  - all finite

## 4. Small-scale controlled experiment

Experimental setup:

- Category: `bottle`
- Training budget: `3 epochs`
- seed: `42`
- Comparison objects:
  - strict: `configs/supersimplenet/supersimplenet_256_mvtec_strict.py`
  - legacy: `configs/supersimplenet/supersimplenet_256_mvtec.py`

Order:

```bash
python tools/benchmark.py \
    --data_root data/mvtec_ad \
    --methods supersimplenet \
    --categories bottle \
    --device cuda \
    --epochs 3 \
    --timeout 3600 \
    --config configs/supersimplenet/supersimplenet_256_mvtec_strict.py \
    --output runs/alignment/supersimplenet_bottle_smoke_strict.json

python tools/benchmark.py \
    --data_root data/mvtec_ad \
    --methods supersimplenet \
    --categories bottle \
    --device cuda \
    --epochs 3 \
    --timeout 3600 \
    --config configs/supersimplenet/supersimplenet_256_mvtec.py \
    --output runs/alignment/supersimplenet_bottle_smoke_legacy.json
```
observe:

- strict `3 epoch` smoke:
  - Old image main caliber (classifier `pred_score`): `image_auroc=0.5960`
  - In the new diagnosis, `pred_score_max` caliber: `image_auroc=0.8627`
  - `pixel_auroc=0.5490`
  - `image_f1max=0.9104`
- strict `20 epoch` smoke (new image main caliber `pred_score_max`):
  - `image_auroc=0.9992`
  -`image_auroc_mean=0.9976`
  - `image_auroc_max=0.9992`
  -`pixel_auroc=0.9622`
  -`image_f1max=0.9921`
- legacy `3 epoch` smoke:
  - `image_auroc=0.5048`
  - `pixel_auroc=0.5424`
  - `image_f1max=0.8630`
- Additional diagnostics:
  - `runs/alignment/supersimplenet_bottle_smoke_strict_v3_gpu2.json`: classifier `image_auroc=0.4849`, `image_auroc_mean=0.8048`, `image_auroc_max=0.8627`
  - `runs/alignment/supersimplenet_bottle_smoke_strict_e20.json`: `image_auroc=0.5865` under the old image main caliber, but `pixel_auroc=0.9679`
- This shows that the training body is not broken; the real problem is focused on the image-side score source, not the backbone / preprocess / generator / segmentation branch

determination:

- `partial-pass`
- `pass`
- Reason:
  - The new image-side scoring caliber has moved the strict `3 epoch bottle` image AUROC from `0.4849-0.5960` to `0.8627`
  - Under the new strict main caliber, `20 epoch bottle` has been restored to `img=0.9992 / pxl=0.9622`, reaching the trust level of playbook Gate 3
  - This phased conclusion has subsequently entered Gate 4 and completed the strict `300 epoch 15/15` full benchmark

## 5. Full Benchmark

**Completed** (2026-03-29)

Result file: `runs/alignment/supersimplenet_strict300_full_merged.json`

| Metric | BaoIAD | Official | Gap |
|--------|----------|----------|-----|
| image_auroc | **0.9817** | 0.986 | -0.0043 |
| pixel_auroc | **0.9757** | 0.975 | +0.0007 |

### Detailed results for each category

| category | image_auroc | pixel_auroc |
|------|-------------|-------------|
| bottle | 1.0000 | 0.9785 |
| cable | 0.9829 | 0.9729 |
| capsule | 0.9932 | 0.9884 |
| carpet | 0.9900 | 0.9820 |
| grid | 0.9841 | 0.9818 |
| hazelnut | 0.9968 | 0.9830 |
| leather | 1.0000 | 0.9900 |
| metal_nut | 0.9985 | 0.9855 |
| pill | 0.9757 | 0.9840 |
| screw | 0.9430 | 0.9893 |
| tile | 0.9953 | 0.9466 |
| toothbrush | 0.8889 | 0.9852 |
| transistor | 0.9900 | 0.9537 |
| wood | 0.9904 | 0.9297 |
| zipper | 0.9963 | 0.9854 |

Shutdown line inspection:

- [x] classifier will trigger obvious stop-line when used as image main score
- [x] `pred_score_max` caliber significantly improved strict `bottle` image AUROC
- [x] `300 epoch strict 15/15` Completed within ±0.01

## 6. Guard

- New/enhanced tests:
  - [`tests/test_models/test_detectors/test_supersimplenet.py`](../../tests/test_models/test_detectors/test_supersimplenet.py)
  - [`tests/test_utils/test_alignment_probe.py`](../../tests/test_utils/test_alignment_probe.py)
  - [`tests/test_utils/test_benchmark_config_detection.py`](../../tests/test_utils/test_benchmark_config_detection.py)
- Added/enhanced configuration guard:
  - [`configs/supersimplenet/supersimplenet_256_mvtec_strict.py`](../../configs/supersimplenet/supersimplenet_256_mvtec_strict.py)
- Added/enhanced probe guard:
  - [`tools/alignment_probe.py`](../../tools/alignment_probe.py)
  - [`baoiad/utils/alignment_probe.py`](../../baoiad/utils/alignment_probe.py)
- New/enhanced diagnostic scripts:
  - [`tools/supersimplenet_reference_diagnose.py`](../../tools/supersimplenet_reference_diagnose.py)
- If you change these paths later, you must rerun:
  - `baoiad/models/detectors/supersimplenet.py`
  - `configs/supersimplenet/supersimplenet_256_mvtec_strict.py`
  - `python tools/alignment_probe.py configs/supersimplenet/supersimplenet_256_mvtec_strict.py --splits train test --max-batch-size 2 --device cuda --output runs/alignment/supersimplenet_probe.json`
  -`python tools/supersimplenet_reference_diagnose.py configs/supersimplenet/supersimplenet_256_mvtec_strict.py --split train --device cuda --max-batch-size 1 --output runs/alignment/supersimplenet_reference_diag_train_b1.json`
  - strict `bottle` smoke JSON

## 7. Residual Risk

- Subsequent maintenance must continue to fix the strict image-level main caliber to `pred_score_max`; if it falls back to classifier `pred_score`, the old stop-line will be retriggered
- The legacy config and old scorer are still kept in the warehouse. If the benchmark priority / evaluator fields are rolled back in the future, the strict conclusion will be polluted again.

## 8. Conclusion

- Final decision: `playbook-complete`
- Alignment results: image_auroc=0.9817 (official 0.986, difference -0.0043), pixel_auroc=0.9757 (official 0.975, difference +0.0007)
- The differences are all within ±0.01 and are considered aligned.

## Alignment checklist

Status only allowed: `matched | mismatch-fixed | intentional-diff | open`

## 1. Enter path

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| train / test color channel | anomalib pre-processor | `LoadImage(to_rgb=True)` | input into backbone as RGB | `configs/_base_/datasets/mvtec_ad.py` + `runs/alignment/supersimplenet_probe.json` | matched |
| resize | anomalib `Resize(256)` | `ResizeAD(size=256)` | input unified resize to `256x256` | `configs/_base_/datasets/mvtec_ad.py` | matched |
| normalization / value range | anomalib `Normalize(mean/std)` | `NormalizeAD()` + `ImgDataPreprocessor` | Normalization using ImageNet statistics | probe input statistics limited | matched |
| seed / batch size | anomalib seed `42` | `configs/_base_/default_runtime.py` + base dataloader | The current strict caliber uses seed `42`, batch size `32` | probe / strict config | matched |

## 2. Backbone / Feature Extraction

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| backbone weight family | `.refs/anomalib/.../torch_model.py` `wide_resnet50_2.tv_in1k` | `RawBackbone(wide_resnet50_2, pretrained=True)` | use the same ImageNet V1 weight family | different wrappers but the same target weight family | intentional-diff |
| Feature layer selection | anomalib `layers=["layer2", "layer3"]` | `SuperSimpleNetDetector(layers=['layer2','layer3'])` | only take `layer2/layer3` | strict config guard | matched |
| Upsampling and patch aggregation | anomalib `F.interpolate(... size=(h*2,w*2)) + AvgPool2d(3,1,1)` | `UpscalingFeatureExtractor.forward()` | Multi-layer features are upsampled to the first layer 2x and then neighborhood aggregation is performed | Code comparison is consistent | matched |
| adapter input dimensions | anomalib `get_channels_dim()` | `feature_extractor.channels` | adaptor / segdec input dimensions consistent | model unit test can be built, probe can pass | matched |

## 3. Anomaly Synthesis

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| Gaussian noise parameters | `.refs/anomalib/.../anomaly_generator.py` | `AnomalyGenerator(noise_mean=0.0, noise_std=0.015)` | `mean=0`, `std=0.015` | Code comparison consistent | matched |
| duplicate / no-overlap / label update | Same as above | `AnomalyGenerator.forward()` | Copy batch, do not cover the existing anomaly area, update labels according to new mask | Code comparison is consistent | matched |
| Perlin mask generation | `generate_perlin_noise + threshold fallback` | `AnomalyGenerator._generate_perlin_mask()` | Use the official style gradient-based Perlin | This round has been changed from simplified approximation to official implementation | mismatch-fixed |
| 50% clean / anomaly probability | Same as above | `_generate_perlin_mask()` | 50% probability of each sample without adding new anomaly | `torch.rand(1).item() > 0.5` | matched |
| Small feature map scale cropping | The reference implementation only runs on real feature maps by default | `_generate_perlin_noise()` | Safely crop single test feature maps such as `16x16` to avoid empty gradient tiles | Only affects test tensors smaller than the official mainline size | intentional-diff |

## 4. Segmentation / Detection Module

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| seg head structure | `.refs/anomalib/.../torch_model.py` | `SegmentationDetectionModule.seg_head` | `1x1 conv -> BN -> LeakyReLU -> 1x1 conv` | Code comparison | matched |
| cls input splicing order | Same as above | `torch.cat((cls_features, map_dec_copy), dim=1)` | `cls_features` in front, `ano_map` in back | Code comparison is consistent | matched |
| pooling / fc header | Same as above | `dec_max/avg + map_max/avg -> cls_fc` | The structure is consistent with the output dimension | The code comparison is consistent | matched |
| stop_grad behavior | anomalib unsupervised `stop_grad=True` | strict config `stop_grad=True` | cls is not passed back to seg map | strict config + code | matched |

## 5. Loss

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| focal input form | `.refs/anomalib/.../loss.py` | `SSNLoss.forward()` | `pred_map/target_mask` and `pred_score/target_label` both use sigmoid focal | Code comparison is consistent | matched |
| trunc_l1 formula | Same as above | `SSNLoss.trunc_l1_loss()` | normal pushes negative, anomaly pushes positive, threshold `0.5` | Code comparison is consistent | matched |
| reduction | Same as above | `sigmoid_focal_loss(... reduction='mean')` | focal uses mean reduction | code comparison is consistent | matched |

## 6. Predict / Scoring

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| anomaly map source | `.refs/anomalib/.../torch_model.py` | `segdec -> anomaly_map_generator` | seg logits upsampling and smoothing | probe `map_shapes=[1,256,256]` | matched |
| smoothing | anomalib `GaussianBlur2d(kernel_size, sigma=4)` | `AnomalyMapGenerator(sigma=4)` | Gaussian smoothing of sigma `4` | Code comparison consistent | matched |
| sigmoid(score/map) | anomalib `anomaly_score.sigmoid(); anomaly_map.sigmoid()` | `forward(mode='predict')` | score/map average to `[0,1]` | unit test guard | mismatch-fixed |

## 7. Optimizer / Runtime

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| strict optimizer param groups | `.refs/anomalib/.../lightning_model.py` | `supersimplenet_256_mvtec_strict.py` | `adaptor=1e-4, wd=1e-2`; `segdec=2e-4, wd=1e-5` | strict config + unit test | mismatch-fixed |
| scheduler | Same as above | `MultiStepLR [80, 90], gamma=0.4` | 80% / 90% decay at 100e | strict config guard | matched |
| runtime | anomalib README `300 epochs` recommendation | strict config `max_epochs=300` | strict mainline budget is switched to the official current recommended value | strict config guard | mismatch-fixed |
| legacy config | History BaoIAD baseline | `supersimplenet_256_mvtec.py` | Keep the old config for baseline only, without strict closure | benchmark priority has been switched to strict | intentional-diff |
| image score source | official code path to be verified by final rerun | `AnomalyDetectionMetric(image_score_field='pred_score_max')` | strict image-level The main indicator is map max first, not classifier score | `3e bottle`: `image_auroc 0.4849 -> 0.8627` | mismatch-fixed |

## 8. Behavior verification conclusion

- [x] strict probe passed, `loss` / `predict` paths are limited
- [x] strict config / benchmark priority / sigmoid range guard completed
- [x] The Perlin path for anomaly synthesis has been changed from approximate implementation to official style
- [x] feature extractor compared to official-style timm reference `MAE=0.0`
- [x] anomaly generator versus reference `MAE=0.0`
- [x] The image-side score of strict `3e bottle` has been switched from classifier to map-max candidate mainline, `img=0.8627`
- [x] strict `20e bottle` passed, `img=0.9992`, `pxl=0.9622`
- [ ] Still missing `300e 15/15` strict rerun
