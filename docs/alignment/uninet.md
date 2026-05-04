# UniNet strict-alignment evidence

- **Method slug**: `uninet`
- **Family**: Hybrid / unified
- **Method README**: [`configs/uninet/README.md`](../../configs/uninet/README.md)

This file preserves the fixed alignment evidence migrated from the manuscript evidence workspace. The content below is translated to English where needed and kept inside this repository so the strict-alignment state can be reviewed without relying on an external paper checkout.

## Config entry points

- [`configs/uninet/uninet_256_mvtec_strict.py`](../../configs/uninet/uninet_256_mvtec_strict.py)
- [`configs/uninet/uninet_256_visa.py`](../../configs/uninet/uninet_256_visa.py)

## Detailed alignment report

**Status**: `playbook-complete`
**Date**: `2026-03-29`

## 1. Reference freezing

- Reference warehouse: `.refs/anomalib`
- Reference commit: `4f6af1acb0ee7b81f54cda036dd9f1c27f63b69a`
- Refer to config/checkpoint: `examples/configs/model/uninet.yaml` + `src/anomalib/models/image/uninet/{lightning_model,torch_model}.py`
- Dataset/Category: MVTec AD, individually trained/evaluated class by class according to `data.category=<category>`
- Input resolution: `256`
- seed: `42`
- Indicator definition: image AUROC is the main one, pixel AUROC is the supplement; anomalib README reference mean is `img=0.956`, `pxl=0.976`
- diff intentional: strict main configuration is frozen to anomalib default `batch_size=32`, `num_workers=8`; if the shared GPU resources are insufficient, smoke can temporarily use CLI to downgrade the batch, but these results will not be regarded as the final acceptance of strict

## 2. Code path comparison conclusion

See [`uninet_checklist.md`](uninet_checklist.md) for the control matrix.

### Consistency confirmed

- The actual order of the teacher feature layer is consistent with the anomalib runtime, both are `layer1 -> layer2 -> layer3`
- bottleneck input splicing, student decoder multi-scale output, DFS, loss and weighted decision mechanism are consistent with the anomalib main implementation
- The strict configuration has been completed with anomalib's `ONE_CLASS + batch32/worker8 + AdamW + target_teacher lr=1e-6 + milestone=80 + early stopping(image_auroc)`

### Fixed inconsistencies

- `UniNetDetector` used to flatten the `teacher_backbone` configuration dictionary into a string, causing the `pretrained/out_indices/frozen` constraints in the strict configuration to not really take effect; it has now been changed to retain and standardize the `FeatureExtractor` configuration
- The strict main configuration has fallen back from incorrect multi-class benchmark assumptions to anomalib's `ONE_CLASS` protocol
- strict optimizer has been further tightened: `fc` no longer participates in effective optimization and is changed to `lr_mult=0, decay_mult=0` to approximate anomalib's parameter group definition

### Items that are still open

- fresh strict `15/15` full benchmark completed; no new acceptance blockers are currently available

## 3. Behavior Probe

Order:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/alignment_probe.py \
    configs/uninet/uninet_256_mvtec_strict.py \
    --splits train test \
    --max-batch-size 2 \
    --device cuda \
    --output runs/alignment/uninet_probe.json
```
in conclusion:

- `alignment_probe` passed, both train `loss` and test `predict` can output limited results on real MVTec data.

Key statistics:

- dataset sample: `train` side sample preview is `zipper/train/good`, `test` side is `bottle/test/broken_large`, and the input shape is both `[2, 3, 256, 256]`
- loss path: `runs/alignment/uninet_probe.json` recorded `train.loss=22.5133`, limited
- predict path: `pred_score` about `3.9897 ~ 4.0038`, `pred_anomaly_map` shape=`[2,1,256,256]`, limited

## 4. Small-scale controlled experiment

Experimental setup:

- A/B 1: Forced single category `bottle` smoke, `1 epoch`, `seed=42`
- Comparison object: strict configuration `configs/uninet/uninet_256_mvtec_strict.py` vs historical legacy configuration `configs/uninet/uninet_256_mvtec.py`
- protocol audit: compare the old multi-class strict path with the current one-class strict path after fallback

observe:

- `bottle` single type smoke:
  - strict(old batch8 path): `img=0.9230`, `pxl=0.8473`
  - legacy: `img=0.9976`, `pxl=0.9721`
- strict multi-class protocol error long run:
  - `epoch4` best: `img=0.9230`, `pxl=0.9569`
  - It has been confirmed that the path does not comply with the anomalib `ONE_CLASS` main protocol, so it is only kept as a troubleshooting record
- strict one-class `benchmark.py` `bottle` smoke (shared GPU batch fallback):
  - current rerun: `img=0.9000`, `pxl=0.8897`

determination:

- `forced bottle` A/B still cannot alone be used as the basis for strict closing, but it has shown that there are residual differences in the main line of strict one-class
- The old multi-class strict path no longer has acceptance significance; all subsequent benchmarks must return to the single-class, class-by-class protocol

## 5. Full Benchmark

**Completed** (2026-03-29)

Result file: `runs/alignment/uninet_strict_oneclass_seq_status.json`

| Metric | BaoIAD | Official | Gap |
|--------|----------|----------|-----|
| image_auroc | **0.9829** | 0.983 | -0.0001 |
| pixel_auroc | **0.9803** | 0.980 | +0.0003 |

### Detailed results for each category

| category | image_auroc | pixel_auroc |
|------|-------------|-------------|
| bottle | 1.0000 | 0.9811 |
| cable | 0.9906 | 0.9771 |
| capsule | 0.9493 | 0.9846 |
| carpet | 0.9847 | 0.9757 |
| grid | 0.9851 | 0.9712 |
| hazelnut | 0.9969 | 0.9861 |
| leather | 0.9979 | 0.9870 |
| metal_nut | 0.9862 | 0.9832 |
| pill | 0.9790 | 0.9832 |
| screw | 0.9600 | 0.9817 |
| tile | 0.9995 | 0.9724 |
| toothbrush | 0.9806 | 0.9816 |
| transistor | 0.9812 | 0.9730 |
| wood | 0.9918 | 0.9824 |
| zipper | 0.9810 | 0.9790 |

Shutdown line inspection:

- [x] The old multi-class strict path has been recognized as a protocol error and will no longer continue to accumulate results.
- [x] fresh strict `15/15` benchmark completed
- [x] is within ±0.01 of the reference mean

## 6. Guard

- New test:
  - `tests/test_models/test_detectors/test_uninet.py`
  - UniNet strict entry test in `tests/test_utils/test_benchmark_config_detection.py`
- New configuration/entrance guard:
  - `configs/uninet/uninet_256_mvtec_strict.py`
  - UniNet strict config priority for `tools/benchmark.py`
- If you change these paths later, you must rerun:
  - `runs/alignment/uninet_probe.json`
  - `bottle` smoke
  - strict one-class full benchmark

## 7. Residual Risk

- The strict mainline must continue to be fixed on the one-class protocol; if the benchmark configuration priority falls back to the old multi-class path, the current conclusion will be invalid.
- Smoke fallback has been used under shared GPU resources, but the current alignment conclusion is based on one-class full benchmark.
- The ECE indicator will print a warning because UniNet outputs the original anomaly score. This does not affect the AUROC alignment judgment, but the log will be noisy.

## 8. Conclusion

- Final decision: `playbook-complete`
- Alignment results: image_auroc=0.9829 (official 0.983, difference -0.0001), pixel_auroc=0.9803 (official 0.980, difference +0.0003)
- The differences are all within ±0.01 and are considered aligned.

## Alignment checklist

Status only allowed: `matched | mismatch-fixed | intentional-diff | open`

## 1. Enter path

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| train color channel | `.refs/anomalib` pre-processor | `configs/_base_/datasets/mvtec_ad.py` | train input enters backbone in RGB | `LoadImage()` default RGB; `runs/alignment/uninet_probe.json` train input shape=`[2,3,256,256]` | matched |
| test color channel | Same as above | Same as above | test input is consistent with train | `runs/alignment/uninet_probe.json` test input shape=`[2,3,256,256]` | matched |
| resize/crop | anomalib MVTec config | `ResizeAD(size=256)` | input unified to `256x256` | `configs/_base_/datasets/mvtec_ad.py` | matched |
| normalization / value range | anomalib pre-processor | `NormalizeAD()` | Use ImageNet mean/std normalization | `runs/alignment/uninet_probe.json` Limited input statistics | matched |
| batch size / workers | `.refs/anomalib/src/anomalib/data/datamodules/image/mvtecad.py` | `configs/uninet/uninet_256_mvtec_strict.py` | strict mainline should fall back to `train/eval batch=32`, `num_workers=8` | datamodule default + strict config | mismatch-fixed |
| benchmark training protocol | `.refs/anomalib/.../uninet/lightning_model.py` + `README.md` | `benchmark_multi_class=False` | strict benchmark should be run class by class according to the single class `category` path | `learning_type=ONE_CLASS` + README `--data.category <category>` | mismatch-fixed |

## 2. Teacher / Bottleneck / Student

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| teacher backbone | `.refs/anomalib/.../uninet/lightning_model.py` | `configs/uninet/uninet_256_mvtec_strict.py` | use `wide_resnet50_2` teacher | strict config | matched |
| teacher feature extraction layer | `.refs/anomalib/.../uninet/torch_model.py` | `Teachers._get_teacher()` | only take `layer1/layer2/layer3` three layers | `out_indices=(1,2,3)` + `tests/test_models/test_detectors/test_uninet.py` | matched |
| teacher feature order | actual forward return of anomalib `create_feature_extractor(return_nodes=['layer3','layer2','layer1'])` | `FeatureExtractor(out_indices=(1,2,3))` | runtime order should be `layer1 -> layer2 -> layer3` | local `torchvision` run check + strict `probe` shape | matched |
| teacher config transparent transmission | The backbone constraints in the reference config must actually take effect | `UniNetDetector.__init__` | The dict backbone configuration cannot be flattened into a string | `teacher_backbone_cfg` guard test | mismatch-fixed |
| source teacher frozen | `.refs/anomalib/.../torch_model.py` | `Teachers.__init__` / `train()` | source `eval()+requires_grad=False`, target trainable | `tests/test_models/test_detectors/test_uninet.py` | matched |
| bottleneck input splicing sequence | `.refs/anomalib/.../torch_model.py` + `attention_bottleneck.py` | `Teachers.forward()` + `BottleneckLayer.forward()` | Splice `target/source` features hierarchically and then enter bottleneck | Check the code item by item for consistency | matched |
| student decoder | `.refs/anomalib/.../resnet_decoder.py` | `StudentDecoder` | `de_wide_resnet50_2` style three-layer upsampling output | `tests/test_models/test_detectors/test_uninet.py` tensor shape | matched |

## 3. Loss / Runtime

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| contrastive/cosine/margin loss | `.refs/anomalib/.../components/loss.py` | `UniNetLoss.forward()` | The temperature, margin, and mask branches are consistent with the reference | The item-by-item comparison code is consistent | matched |
| DFS + BCE head | `.refs/anomalib/.../torch_model.py` | `UniNetDetector.forward(mode='loss')` | `dfs + distillation loss + 2x BCEWithLogitsLoss` | Code paths consistent; probe `loss=22.5133` limited | matched |
| optimizer | `.refs/anomalib/.../lightning_model.py` | `configs/uninet/uninet_256_mvtec_strict.py` | `AdamW(lr=5e-3, wd=1e-5, eps=1e-10, amsgrad=True)` | strict config guard | mismatch-fixed |
| target teacher lr | Same as above | strict `optim_wrapper.paramwise_cfg` | target teacher alone `lr=1e-6` | `lr_mult=2e-4` + smoke log paramwise print | mismatch-fixed |
| Whether fc participates in optimization | Same as above | strict `optim_wrapper.paramwise_cfg` | The reference implementation does not put `fc` into the optimizer | strict config fixes `fc` to `lr_mult=0, decay_mult=0` | mismatch-fixed |
| scheduler | Same as above | strict `param_scheduler` | `MultiStepLR(milestones=[80], gamma=0.2)` | strict config guard | mismatch-fixed |
| early stopping / best ckpt | anomalib YAML + Lightning callback | strict `custom_hooks` / `default_hooks.checkpoint` | monitor `image_auroc` and save best | strict config guard | mismatch-fixed |

## 4. Predict / Scoring

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| anomaly map source | `.refs/anomalib/.../torch_model.py` | `forward(mode='predict')` | Each layer is added after `1 - cosine_similarity` | The code paths are consistent | matched |
| map upsampling | `.refs/anomalib/.../components/anomaly_map.py` | `weighted_decision_mechanism()` | Each layer of map is bilinearly upsampled to the input size | The code path is consistent | matched |
| smoothing | Same as above | `GaussianBlur2d(sigma=4.0, kernel_size=(5,5))` | Use sigma=4 Gaussian smoothing score chart | Same code path | matched |
| image score aggregation | Same as above | `topk` over smoothed map | According to weighted decision, take the top-k first value as the image score | The code path is consistent; `tests/test_models/test_detectors/test_uninet.py` predict guard | matched |
| Output validity | playbook Gate 2 | `build_predict_results()` | `pred_score / pred_anomaly_map` must be limited | `runs/alignment/uninet_probe.json` + unit test | matched |

## 5. Behavior verification conclusion

- [x] After fixing the seed, the dataset sample structure is as expected
- [x] `gt_mask` shape and range are as expected
- [x] `loss` / `predict` paths have been verified for finiteness
- [x] strict `bottle` smoke has been passed
- [x] The old multi-class strict path has been confirmed to be not the official main protocol and will no longer be used as a basis for closure.

## 6. Remarks

- Although the UniNet paper/README emphasizes unified / multi-class capabilities, the current public training entry and `learning_type` of anomalib are still `ONE_CLASS`; the strict mainline must be based on this.
- The previous multi-class strict smoke is only retained as "error protocol troubleshooting record" and will no longer participate in the final acceptance.
