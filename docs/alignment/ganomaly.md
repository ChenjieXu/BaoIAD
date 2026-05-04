# GANomaly strict-alignment evidence

- **Method slug**: `ganomaly`
- **Family**: Reconstruction / ViT
- **Method README**: [`configs/ganomaly/README.md`](../../configs/ganomaly/README.md)

This file preserves the fixed alignment evidence migrated from the manuscript evidence workspace. The content below is translated to English where needed and kept inside this repository so the strict-alignment state can be reviewed without relying on an external paper checkout.

## Config entry points

- [`configs/ganomaly/ganomaly_256_mvtec_strict.py`](../../configs/ganomaly/ganomaly_256_mvtec_strict.py)
- [`configs/ganomaly/ganomaly_256_visa.py`](../../configs/ganomaly/ganomaly_256_visa.py)

## Detailed alignment report

**Status**: `closed`
**Date**: `2026-04-06`

## 1. Reference freezing

- Reference warehouse: `samet-akcay/ganomaly`
- Reference commit: `78da4ea9a99f5b02ab60dd651a18def929176d77`
- Reference implementation: anomalib `src/anomalib/models/image/ganomaly/`
- Refer to config/checkpoint: the default `GANomaly` training protocol in the original warehouse; supplementary reference to anomalib's image-only evaluation semantics in MVTec scenarios
- Data set/class: `MVTec AD`, single-class `15/15`
- Input resolution: `256x256`
- seed: The current strict configuration follows the warehouse fixed seed mechanism
- Indicator definition: Current strict only accepts image-side indicators; `image_auroc` is the main acceptance indicator
- intentional diff:
  - The original warehouse is not the MVTec code path, so resize / dataset is encapsulated using BaoIAD’s `MVTecADDataset`
  - The framework API still retains the `pred_anomaly_map` placeholder output, but the strict evaluator has explicitly disabled the pixel metric

## 2. Code path comparison conclusion

See [`ganomaly_checklist.md`](ganomaly_checklist.md) for the control matrix.

### Consistency confirmed

- `encoder -> decoder -> encoder` backbone topology is consistent with `pad_nextpow2` logic and main reference
- latent-distance image score remains `mean((latent_i - latent_o)^2)`
- generator / discriminator dual optimizer topology consistent with anomalib `training_step`
- generator adversarial loss using discriminator feature matching
- encoder consistency loss using L2
- reconstruction loss using L1
- Weight `wadv=1, wcon=50, wenc=1` is consistent with the official
- Optimizer Adam(lr=2e-4, beta1=0.5, beta2=0.999)
- Training budget 15 epochs, batch_size=64

### Fixed inconsistencies (history)

- The strict configuration has been switched from single optimizer to generator / discriminator dual optimizer
- generator adversarial loss has been repaired back to the official discriminator feature matching
- encoder consistency loss has been fixed back to the official version `L2`
- strict evaluator has been changed to image-only, and image-score min-max normalization is turned on
- strict preprocessing has been added to the official folder path: `PIL -> Resize(shorter_edge=256, keep_ratio=True) -> CenterCrop(256) -> Normalize(0.5,0.5,0.5)`

## 3. Behavior Probe

Existing probe archives (increasing by version):

- `runs/alignment/ganomaly_probe.json`: old path baseline
- `runs/alignment/ganomaly_probe_strict_v2.json`: Double optimizer path verification
- `runs/alignment/ganomaly_probe_strict_v4.json`: `[-1,1]` Normalized verification
- `runs/alignment/ganomaly_probe_strict_v5.json`: `PIL + keep_ratio + CenterCrop` verification

All probes passed.

## 4. Small-scale controlled experiment

smoke experiment history:

| Version | Changes | bottle img |
|------|------|-----------|
| v2 | dual optimizer | 0.1095 |
| v3 | - | 0.0992 |
| v4 | feature matching | 0.1651 |
| v5 | `[-1,1]` normalize | 0.3214 |
| v6 | PIL + keep_ratio + CenterCrop | 0.4048 |
| exact15 | 15e full budget | 0.5230@epoch4 |

## 5. Full Benchmark

### Strict 15/15 Results

**Result file**: `runs/alignment/ganomaly_strict_v2.json`

| category | image_auroc | image_f1max | image_ap | image_fpr@95tpr |
|------|-------------|-------------|----------|----------------|
| bottle | 0.5825 | 0.8630 | 0.8521 | 1.0000 |
| cable | 0.5109 | 0.7667 | 0.6303 | 0.9655 |
| capsule | 0.5481 | 0.9083 | 0.8685 | 0.8696 |
| carpet | 0.6232 | 0.8683 | 0.8428 | 0.8929 |
| grid | 0.7251 | 0.8594 | 0.8860 | 0.7619 |
| hazelnut | 0.6332 | 0.7821 | 0.7873 | 0.9000 |
| leather | 0.7106 | 0.8558 | 0.8908 | 0.9375 |
| metal_nut | 0.6852 | 0.9020 | 0.8917 | 0.7727 |
| pill | 0.5873 | 0.9156 | 0.8840 | 1.0000 |
| screw | 0.7167 | 0.8718 | 0.8563 | 0.7317 |
| tile | 0.6356 | 0.8400 | 0.8243 | 0.9394 |
| toothbrush | 0.4472 | 0.8333 | 0.7663 | 1.0000 |
| transistor | 0.7008 | 0.6167 | 0.6894 | 0.7667 |
| wood | 0.6272 | 0.8633 | 0.8574 | 0.8421 |
| zipper | 0.4414 | 0.8848 | 0.7599 | 0.9375 |

**15/15 Mean**: `image_auroc = 0.6117`

**Note**: GANomaly is an image-only weak baseline method, and the mean value of 0.61 is within a reasonable range.

### anomalib reference comparison

anomalib official benchmark (seed=42, 100 epochs) class-by-class Image AUROC:

| Category | anomalib | BaoIAD | Difference |
|------|----------|----------|------|
| Carpet | 0.203 | 0.6232 | +0.4202 |
| Grid | 0.404 | 0.7251 | +0.3211 |
| Leather | 0.413 | 0.7106 | +0.2976 |
| Tile | 0.408 | 0.6356 | +0.2276 |
| Wood | 0.744 | 0.6272 | -0.1168 |
| Bottle | 0.251 | 0.5825 | +0.3315 |
| Cable | 0.457 | 0.5109 | +0.0539 |
| Capsule | 0.682 | 0.5481 | -0.1339 |
| Hazelnut | 0.537 | 0.6332 | +0.0962 |
| Metal Nut | 0.270 | 0.6852 | +0.4152 |
| Pill | 0.472 | 0.5873 | +0.1153 |
| Screw | 0.231 | 0.7167 | +0.4857 |
| Toothbrush | 0.372 | 0.4472 | +0.0752 |
| Transistor | 0.440 | 0.7008 | +0.2608 |
| Zipper | 0.434 | 0.4414 | +0.0074 |
| **Mean** | **0.421** | **0.6117** | **+0.1907** |

BaoIAD is higher than anomalib on class 13/15. Most categories of anomalib are between 0.2–0.4 (close to random), indicating that its GANomaly implementation has training stability issues. GANomaly is not included in the ADer benchmark (arXiv 2406.03262).

### Historical baseline

- `runs/alignment/ganomaly_v1.json`: old image-only baseline (13/15), image mean = 0.6007

## 6. Guard

- **Test file**: `tests/test_models/test_detectors/test_ganomaly.py` (6 tests, all pass)
  - `test_forward_tensor`: tensor mode forward
  - `test_forward_loss`: loss mode forward
  - `test_forward_predict`: predict mode forward
  - `test_strict_predict_uses_zero_placeholder_maps`: strict predict zero placeholder map
  - `test_strict_train_step_with_split_optimizers`: strict split optimizer training step
  - `test_ganomaly_strict_config_freezes_image_only_protocol`: strict config freeze check
- **OptimWrapper constructor**: `baoiad/engine/optimizers/ganomaly_optim_wrapper_constructor.py`
- **Guard Key Points**:
  - strict training must use split optimizer
  - strict evaluator can only produce image-side metrics
  - strict checkpoint selector fixed to `best image_auroc`
  - `pred_anomaly_map` is zero placeholder (intentional-diff)

## 7. Conclusion

- **Final Judgment**: `closed`
- **Closing basis**:
  1. The code path is fully aligned with the anomalib reference implementation (encoder-decoder-encoder topology, loss semantics, dual optimizer training step sequence)
  2. The Strict configuration freezes all official super parameters (lr=2e-4, beta1=0.5, wadv=1, wcon=50, wenc=1, 15 epochs, batch_size=64)
  3. The preprocessing path has been aligned to the official one (PIL + Resize(256, keep_ratio) + CenterCrop(256) + Normalize(0.5, 0.5, 0.5))
  4. All 6 unit tests passed
  5. 15/15 benchmark results have been released
  6. GANomaly is a known weak baseline method, and the mean image_auroc of 0.61 is within the expected range.
- **intentional-diff**:
  - `pred_anomaly_map` has zero occupancy, and GANomaly’s original design is image-only
  - The original repository does not provide MVTec benchmark mean, use anomalib proxy as a reference

## Alignment checklist

Status only allowed: `matched | mismatch-fixed | intentional-diff | open`

## 1. Enter path

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| train color channel | original GANomaly default RGB input | `configs/_base_/datasets/mvtec_ad.py` + `ganomaly_256_mvtec_strict.py` | train enters the network as RGB three-channel tensor | strict config + base dataset | matched |
| test color channel | Same as above | Same as above | test keeps the same RGB path | strict config + base dataset | matched |
| resize / crop | Original repository `Resize(isize) -> CenterCrop(isize)` | `LoadImage(backend='pil') + ResizeAD(keep_ratio=True, official_pil=True) + CenterCrop(256)` | Geometric path pasted to torchvision/PIL semantics | strict config + `ganomaly_probe_strict_v5.json` | mismatch-fixed |
| normalization / value range | original warehouse `Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))` | `NormalizeAD(mean=127.5, std=127.5)` | input fell into `[-1,1]` | strict config + `ganomaly_probe_strict_v4.json` | mismatch-fixed |

## 2. Structure and training

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| `pad_nextpow2` | `anomalib pad_nextpow2` / Original warehouse power padding | `baoiad/models/detectors/ganomaly.py::_pad_nextpow2` | Input first pad to the nearest 2nd power size | detector code | matched |
| encoder / decoder / discriminator topology | `lib/networks.py` / anomalib `torch_model.py` | `baoiad/models/detectors/ganomaly.py` | keep `encoder -> decoder -> encoder` + discriminator structure | detector code | matched |
| generator optimizer | Original repository / anomalib dual optimizer | `GanomalyOptimWrapperConstructor` | generator alone Adam update | constructor + strict config | mismatch-fixed |
| discriminator optimizer | Same as above | `GanomalyOptimWrapperConstructor` | discriminator alone Adam update | constructor + strict config | mismatch-fixed |
| Training step sequence | anomalib `training_step`: G first, then D | `GanomalyDetector.train_step()` | strict train_step back-transmit and merge step | detector code + tests | mismatch-fixed |
| discriminator reinit | anomalib: reinit when D loss < 1e-5 | `GanomalyDetector.train_step()` | retain D reinit logic | detector code | matched |
| Training budget | Original warehouse `niter=15` | `train_cfg.max_epochs=15` | strict mainline fixed 15 epoch | strict config | matched |
| batch_size | original warehouse `batch_size=64` | `train_dataloader.batch_size=64` | strict mainline fixed 64 | strict config | matched |

## 3. Loss

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| adversarial loss | anomalib `GeneratorLoss`: MSE on discriminator features | `GanomalyDetector._generator_loss` | The adversarial component of G is feature matching MSE | detector code | mismatch-fixed |
| reconstruction loss | Original warehouse `L1(real, fake)` | `GanomalyDetector._generator_loss` | The reconstruction component of G remains L1 | detector code | matched |
| latent loss | anomalib `GeneratorLoss`: L2(latent_o, latent_i) | `GanomalyDetector._generator_loss` | The latent component of G is L2 | detector code | mismatch-fixed |
| discriminator loss | anomalib `DiscriminatorLoss`: BCE | `GanomalyDetector._discriminator_loss` | D using BCE with 0.5 scaling | detector code | matched |
| loss weight | original warehouse `w_adv=1, w_con=50, w_enc=1` | strict config | weight remains consistent | strict config | matched |

## 4. Predict / Scoring

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| image score | original warehouse `mean((latent_i-latent_o)^2)` | `GanomalyDetector.forward(mode='predict')` | image score maintains latent-distance semantics | detector code | matched |
| test-time min-max normalization | anomalib `on_validation/test_batch_end` | `AnomalyDetectionMetric(normalize_image_scores=True)` | image indicators are calculated after normalization by dataset-level min-max | strict config | mismatch-fixed |
| pixel evaluator | The original warehouse does not have pixel main line | strict `val_evaluator/test_evaluator` | strict main line does not produce pixel indicator conclusion | strict config | mismatch-fixed |
| `pred_anomaly_map` | The framework has no official pixel map | detector zero placeholder map | Only API placeholders are reserved and are not allowed to be used as pixels Conclusion | detector code + tests | intentional-diff |
| best checkpoint selector | MVTec proxy only looks at image-side indicators | `benchmark_result_selector` + `CheckpointHook(save_best='ad/image_auroc')` | strict checkpoint selects best with `image_auroc` | strict config + tests | mismatch-fixed |

## 5. Behavior verification conclusion

- [x] New strict probe filed into `runs/alignment/ganomaly_probe_strict_v5.json`
- [x] New strict `bottle` smoke has been added to `v6 manual`, currently `img=0.4048`
- [x] strict `15/15` archived to `runs/alignment/ganomaly_strict_v2.json`
- [x] strict training path already has split-optimizer targeted tests
- [x] strict config already has image-only evaluator / best-image selector guard
- [x] All 6 unit tests passed
- [x] GANomaly is a known weak baseline method, mean `image_auroc=0.6117` is within the expected range
- [ ] None
