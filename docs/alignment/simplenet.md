# SimpleNet strict-alignment evidence

- **Method slug**: `simplenet`
- **Family**: Discriminative
- **Method README**: [`configs/simplenet/README.md`](../../configs/simplenet/README.md)

This file preserves the fixed alignment evidence migrated from the manuscript evidence workspace. The content below is translated to English where needed and kept inside this repository so the strict-alignment state can be reviewed without relying on an external paper checkout.

## Config entry points

- [`configs/simplenet/simplenet_wrn50_288_mvtec_strict.py`](../../configs/simplenet/simplenet_wrn50_288_mvtec_strict.py)
- [`configs/simplenet/simplenet_wrn50_288_visa.py`](../../configs/simplenet/simplenet_wrn50_288_visa.py)

## Detailed alignment report

**Status**: `aligned`
**Date**: `2026-03-28`

## 1. Reference freezing

- Reference repository: official `DonaldRR/SimpleNet`
- Reference commit: `351a2b8d4e8cfc944dbccbf9bc6ceda930c6f26b`
- Refer to config/checkpoint:
  - `run.sh`
  - `main.py::net`, `main.py::dataset`
  - `simplenet.py::load`, `_train_discriminator`, `_predict`, `_evaluate`
  - `common.py::RescaleSegmentor`
- Dataset/Category: MVTec AD, 15 categories single class training/single class evaluation
- Input resolution: `Resize(329) -> CenterCrop(288)`
- seed: `0`
- Indicator definition:
  - Official `_evaluate()` does min-max in the test set for image score, and then calculates image AUROC / AP
  - After the anomaly map is upsampled + `gaussian sigma=4`, the pixel AUROC / PRO is calculated according to the official normalized semantics
- intentional diff:
  - BaoIAD retains MMEngine training/evaluation framework
  - Official `meta_epochs=40`, `gan_epochs=4` mapped to `max_epochs=160`, `val_interval=4`
  - The best checkpoint caliber is reproduced by `benchmark_result_selector = dict(mode='best', metric='image_auroc')`
  - fresh strict `15/15` archive uses the local cache data root `/dev/shm/baoiad_data/mvtec_ad` when running to avoid AFS I/O becoming a non-algorithm bottleneck; the model caliber remains unchanged

> In terms of published proxy, the historical README / ADer archive still retains `image_auroc=0.954`, `pixel_auroc=0.968` as externally comparable MVTec reference numbers; however, the main frozen object of Gate 0 this time is the official upstream code path.

## 2. Code path comparison conclusion

See [`simplenet_checklist.md`](simplenet_checklist.md) for the control matrix.

### Consistency confirmed

- strict main configuration frozen to official `329 -> 288` preprocessing, `wide_resnet50_2 + layer2/layer3`, `pre_proj=1`, `noise_std=0.015`, `dsc_margin=0.5`
- patchify / preprocessing / aggregator / image score aggregation path has been aligned with upstream `PatchMaker + Preprocessing + Aggregator + score()`

### Fixed inconsistencies

- The old mainline only has ADer-style single optimizer / `StepLR`; the strict mainline has been changed to the official split optimizer, and `SimpleNetOptimWrapperConstructor` has been added
- The old `predict` directly outputs patch-grid map; the strict path has been supplemented with the official `RescaleSegmentor` upsampling + Gaussian smoothing
- The old evaluation uses raw image score / raw map by default; strict config has added image-score and map normalization semantics
- `benchmark.py` will compress workers to `0` by default; strict config has explicitly declared `benchmark_keep_dataloader_workers=True`
- `tools/benchmark.py --methods simplenet` The default entry has been switched to strict config to avoid continuing to run the legacy 256 caliber by mistake

### Items that are still open

- none

## 3. Behavior Probe

Order:

```bash
python tools/alignment_probe.py configs/simplenet/simplenet_wrn50_288_mvtec_strict.py \
    --splits train test \
    --max-batch-size 2 \
    --device cuda \
    --seed 0 \
    --output runs/alignment/simplenet_probe.json
```
in conclusion:

- `runs/alignment/simplenet_probe.json` passed, all structure checks are `ok`
- strict config stably produces limited train loss, test score and full-resolution anomaly map on real `bottle` samples

Key statistics:

- train/test input shapes are all `[3, 288, 288]`
- train single batch `loss = 1.4391`
- `pred_score` finite, mean=`0.3550`
- `pred_anomaly_map` shape is `[1, 288, 288]`

## 4. Small-scale controlled experiment

Experimental setup:

- Category: `bottle`
- Training budget: `4 epochs`
- seed: `0`
- Comparison object: strict main configuration `configs/simplenet/simplenet_wrn50_288_mvtec_strict.py`

Order:

```bash
python tools/train.py configs/simplenet/simplenet_wrn50_288_mvtec_strict.py \
    --work-dir runs/alignment/simplenet_bottle_smoke \
    --cfg-options \
        train_cfg.max_epochs=4 \
        train_cfg.val_interval=4 \
        train_dataloader.num_workers=0 \
        train_dataloader.persistent_workers=False \
        test_dataloader.num_workers=0 \
        test_dataloader.persistent_workers=False \
        val_dataloader.num_workers=0 \
        val_dataloader.persistent_workers=False
```
observe:

- loss continues to decrease: `0.9678 -> 0.8643 -> 0.7523 -> 0.6450`
- `bottle` verification result of the 4th epoch:
  - `image_auroc = 0.9968`
  - `pixel_auroc = 0.9806`
  - `image_ap = 0.9990`
  - `aupro = 0.9096`

determination:

- `pass`
- Reason: The loss trend is normal, single type smoke does not trigger stop-line

## 5. Full Benchmark

Archive command mainline:

```bash
python tools/benchmark.py \
    --data_root /dev/shm/baoiad_data/mvtec_ad \
    --methods simplenet \
    --categories all \
    --output runs/alignment/simplenet_strict_v1.json
```
illustrate:

- fresh strict `15/15` full benchmark is actually archived as 4 shards + 2 accelerated single-type tasks, and then merged into `runs/alignment/simplenet_strict_v1.json`
- Main archive uses 4 shard JSON:
  - `runs/alignment/simplenet_strict_v1_part0_local.json`
  - `runs/alignment/simplenet_strict_v1_part1_local.json`
  - `runs/alignment/simplenet_strict_v1_part2_local.json`
  - `runs/alignment/simplenet_strict_v1_part3_local.json`
- The extra single-type tasks of `metal_nut` and `toothbrush` are only used as acceleration redundancy and do not change the main archive value.

Summary of results:

| Metric | Published Proxy | BaoIAD Strict | Gap |
|--------|-----------------|-----------------|-----|
| image_auroc | `0.9540` | `0.9965` | `+4.25%` |
| pixel_auroc | `0.9680` | `0.9758` | `+0.78%` |
| image_ap | `-` | `0.9990` | `-` |
| aupro | `-` | `0.9035` | `-` |

Partial per-class results:

| Category | image_auroc | pixel_auroc | aupro |
|----------|-------------|-------------|-------|
| bottle | `0.9992` | `0.9801` | `0.9150` |
| cable | `0.9993` | `0.9730` | `0.8933` |
| capsule | `0.9828` | `0.9882` | `0.9429` |
| screw | `0.9910` | `0.9897` | `0.9380` |
| tile | `0.9989` | `0.9566` | `0.7836` |
| transistor | `1.0000` | `0.9455` | `0.8902` |
| wood | `1.0000` | `0.9394` | `0.8375` |
| zipper | `0.9989` | `0.9852` | `0.9439` |

Shutdown line inspection:

- [x] No large area appears near `0.5` image AUROC
- [x] No unified platform value collapse occurs
- [x] weak class still maintains normal separation without score collapse
- [x] fresh strict `15/15` full benchmark archived to `runs/alignment/simplenet_strict_v1.json`

## 6. Guard

- New/enhanced tests:
  - `tests/test_models/test_detectors/test_simplenet.py`
  - `tests/test_evaluation/test_ad_metric.py`
  - `tests/test_utils/test_benchmark_config_detection.py`
- New configuration/entrance guard:
  - `configs/simplenet/simplenet_wrn50_288_mvtec_strict.py`
  - `simplenet` config priority for `tools/benchmark.py`
  - `baoiad/engine/optimizers/simplenet_optim_wrapper_constructor.py`
- Key anti-regression points:
  - strict path must remain split optimizer
  - strict `predict` must output full-resolution smoothed anomaly map
  - strict evaluator must enable official image/map normalization semantics
  - The benchmark mainline must retain dataloader workers and cannot fall back to `0 workers`

## 7. Residual Risk

- published proxy still comes from ADer / historical external value, not the official fresh MVTec rerun of the same commit; therefore the value gap can only be regarded as "published proxy comparison" and cannot reversely overwrite this official-code freeze
- The current strict benchmark is local-cache archive; if it is re-run on the AFS direct reading path, the wall time will be significantly longer, but the algorithm conclusion should not be changed.

## 8. Conclusion

- Final decision: `aligned`
- Allowed to proceed to next stage: `yes`
- Current current conclusion: strict official `15/15` completed and filed in `runs/alignment/simplenet_strict_v1.json`

## Alignment checklist

Status only allowed: `matched | mismatch-fixed | intentional-diff | open`

## 1. Enter path

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| train color channel | `DonaldRR/SimpleNet/main.py::dataset` + `Image.convert('RGB')` | `LoadImage(to_rgb=True)` + strict config train pipeline | train enters backbone as RGB tensor | `runs/alignment/simplenet_probe.json` train sample shape=`[3,288,288]` | matched |
| test color channel | Same as above | Same as above | test and train maintain the same RGB path | `runs/alignment/simplenet_probe.json` test sample shape=`[3,288,288]` | matched |
| resize / crop | `run.sh`: `Resize(329) -> CenterCrop(288)` | `configs/simplenet/simplenet_wrn50_288_mvtec_strict.py` | The input space path is consistent with the official main line | strict config pipeline is clearly `329 -> 288`; the measured input of probe is `288x288` | matched |
| normalization / value range | `main.py::dataset` ImageNet normalize | `NormalizeAD` | Backbone is the limited floating point tensor after ImageNet normalization | probe in train/test `inputs_finite=true` | matched |

## 2. Features and training paths

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| backbone / layers | `run.sh` + `main.py::net` | `FeatureExtractor(backbone_name='wide_resnet50_2', out_indices=(2,3))` | WRN-50-2 extraction `layer2/layer3` | strict config frozen | matched |
| patch / preprocess / aggregator | `simplenet.py::PatchMaker`, `common.Preprocessing`, `common.Aggregator` | `baoiad/models/detectors/simplenet.py` | patchify + per-layer pooling + target embedding aggregation consistency | Current detector and upstream isomorphic implementations are compared item by item | matched |
| projection / discriminator superparameters | `run.sh` + `simplenet.py::load` | strict config + detector defaults | `pre_proj=1`, `noise_std=0.015`, `dsc_margin=0.5`, `dsc_layers=2`, `dsc_hidden=1024` | strict config frozen | matched |
| projection / discriminator optimizer split | `simplenet.py::load` + `_train_discriminator` | `SimpleNetOptimWrapperConstructor` + `SimpleNetDetector.train_step` | projection uses `AdamW(1e-4, 1e-2)`, discriminator uses `Adam(2e-4, 1e-5)`, and the same loss is back-transmitted separately step | Added strict `train_step` and optimizer constructor; `bottle` smoke has been run through | mismatch-fixed |
| Training budget / verification rhythm | `run.sh`: `meta_epochs=40`, `gan_epochs=4`, verified once per meta epoch | `train_cfg.max_epochs=160`, `val_interval=4`, `benchmark_result_selector=best(image_auroc)` | Use MMEngine to equivalently map the official training cycle and best-record selection | strict config frozen; `tools/benchmark.py` default entry has been switched to strict config | intentional-diff |

## 3. Predict / Scoring / Eval

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| image score aggregation | `simplenet.py::_predict` + `PatchMaker.score()` | `PatchMaker.unpatch_scores()` + `PatchMaker.score()` | image score takes patch score global max | strict predict path has been changed to official isomorphic aggregation | matched |
| anomaly map upsampling + smoothing | `common.RescaleSegmentor.convert_to_segmentation` | `RescaleSegmentor` in `simplenet.py` | patch grid upsampling to input resolution and do `gaussian sigma=4` | `runs/alignment/simplenet_probe.json` map shape=`[1,288,288]` | mismatch-fixed |
| image-score normalization | `simplenet.py::_evaluate` | `AnomalyDetectionMetric(normalize_image_scores=True)` | The image score in the test set is min-maxed and then calculated image AUROC / AP | strict config is turned on | mismatch-fixed |
| pixel-map normalization | `simplenet.py::_evaluate` | `AnomalyDetectionMetric(normalize_pred_maps='batch_broadcast')` | Normalize the anomaly map with the same caliber as the official `_evaluate` and then calculate the pixel index | strict config is turned on | mismatch-fixed |
| benchmark worker policy | Playbook Gate 4 runtime | `benchmark_keep_dataloader_workers=True` | `benchmark.py` Do not force strict mainline into `0 workers` | fresh `15/15` strict full archive generated by worker-preserve mainline | mismatch-fixed |
| benchmark default entrance | playbook Gate 4 | `tools/benchmark.py::_METHOD_CONFIG_PRIORITY['simplenet']` | `--methods simplenet` defaults to strict official mainline instead of legacy 256 config | New benchmark priority guard + single test | mismatch-fixed |

## 4. Behavior verification conclusion

- [x] After fixing `seed=0`, the dataset sample structure is as expected
- [x] `runs/alignment/simplenet_probe.json` It has been proved that train loss / test score / test map are all limited
- [x] strict `predict` measured output `(1,288,288)` anomaly map
- [x] `bottle` smoke loss continues to decrease from `0.9678 -> 0.6450` without triggering collapse
- [x] fresh strict `15/15` full benchmark archived to `runs/alignment/simplenet_strict_v1.json`
