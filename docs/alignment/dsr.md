# DSR strict-alignment evidence

- **Method slug**: `dsr`
- **Family**: Self-supervised synthesis
- **Method README**: [`configs/dsr/README.md`](../../configs/dsr/README.md)

This file preserves the fixed alignment evidence migrated from the manuscript evidence workspace. The content below is translated to English where needed and kept inside this repository so the strict-alignment state can be reviewed without relying on an external paper checkout.

## Config entry points

- [`configs/dsr/dsr_256_mvtec_strict.py`](../../configs/dsr/dsr_256_mvtec_strict.py)
- [`configs/dsr/dsr_256_visa.py`](../../configs/dsr/dsr_256_visa.py)

## Detailed alignment report

**Status**: `aligned`
**Date**: `2026-03-26`

## 1. Reference freezing

- Reference warehouse: `https://gh-proxy.com/https://github.com/open-edge-platform/anomalib`
- Reference commit: `4f6af1acb0ee7b81f54cda036dd9f1c27f63b69a`
- Refer to config/checkpoint:
  - `src/anomalib/models/image/dsr/torch_model.py`
  - `src/anomalib/models/image/dsr/loss.py`
  - `src/anomalib/models/image/dsr/anomaly_generator.py`
  - `src/anomalib/models/image/dsr/lightning_model.py`
  - `examples/configs/data/mvtec.yaml`
- Dataset/Category: MVTec AD, 15 categories of standard benchmark
- Input resolution: `256x256`
- seed: `42`
- Indicator definition: image AUROC / pixel AUROC
- intentional diff:
  - BaoIAD currently accesses the unified benchmark entrance through MMEngine strict configuration `configs/dsr/dsr_256_mvtec_strict.py`
  - The current local runtime retains `batch_size=4`, `val_interval=20` instead of `32 / 32` in the anomalib sample data configuration
  - `pretrained_vqvae_path='auto'` is the engineering compatible interface of BaoIAD, not the original API of anomalib

## 2. Code path comparison conclusion

See [`dsr_checklist.md`](dsr_checklist.md) for the control matrix.

### Consistency confirmed

- Input preprocessing remains `RGB + resize(256) + [0,1]`, ImageNet Normalize is not introduced
- The latent anomaly generation and feature/image/segmentation loss formula of Phase 2 are consistent with the current implementation of anomalib
- The predict path keeps `anomaly_map` from the upsampling branch, and `image_score` from the pre-upsampling map's `pool21 + max`
- The training switching logic of phase 2 -> 3 is consistent with the current `upsampling_train_ratio=0.7` caliber of anomalib

### Fixed inconsistencies

- `pretrained_vqvae_path='auto'` used to blindly load local old weights with the same name; if the file is not in anomalib compatible format, DSR will directly report an error during the model construction phase.
- Checkpoint key compatibility check has been added, and incompatible local weights will automatically trigger official weight re-download fallback in `auto` mode
- Path analysis also adds combined alias search for `pretrained/` vs `pre_trained/`, `.pth` vs `.pckl`

### Items that are still open

- No code path `open` item

## 3. Behavior Probe

Order:

```bash
python tools/alignment_probe.py configs/dsr/dsr_256_mvtec_strict.py \
    --splits train test \
    --max-batch-size 1 \
    --device cuda \
    --output runs/alignment/dsr_probe.json
```
in conclusion:

- `runs/alignment/dsr_probe.json` has been regenerated, all structure checks in train/test are `ok`
- The probe triggered the `auto` compatibility rollback for the first time: the local old `pre_trained/vq_model_pretrained_128_4096.pckl` was judged to be incompatible, and then automatically downloaded and replaced with anomalib compatible weights
- The current probe results are based on the replaced official compatible `.pckl`, proving that the DSR main configuration can be completed on real data `build -> loss -> predict` full path

Key statistics:

- dataset sample:
  - train: `bottle/train/good/072.png`
  - test: `bottle/test/broken_large/000.png`
  - Input shape is `1 x 3 x 256 x 256`
- loss path:
  - `train.loss` is limited, the single batch loss in probe is `4.2272`
- predict path:
  - `pred_score` is limited, `0.7047` in probe
  - `pred_anomaly_map` shape is `1 x 256 x 256`, and the value range is approximately `[0.0635, 0.9351]`

## 4. Small-scale controlled experiment

Experimental setup:

- Category: `bottle`
- Training budget: `2 epochs`
- seed: `42`
- Comparison object: current strict main configuration `configs/dsr/dsr_256_mvtec_strict.py`

observe:

- smoke work dir: `runs/alignment/dsr_bottle_smoke_e2`
- The training loss of the first epoch continues to decrease from `0.3055 -> 0.1809 -> 0.1102`
- The first verification result is `image_auroc = 0.9365`, `pixel_auroc = 0.5042`
- The training loss of the second epoch remains low and stable: `0.0298 -> 0.0410 -> 0.0330`
- The second verification results are `image_auroc = 0.9365`, `pixel_auroc = 0.8244`, and the pixel indicator has rebounded significantly compared with epoch 1.

determination:

- `pass`
- Reason: `bottle` smoke under the correct VQ-VAE weight has been completed, the entire train/val path is stable, the loss does not fluctuate abnormally, and neither the image-level nor pixel-level indicators trigger the shutdown line

## 5. Full Benchmark

Execute command:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/benchmark.py \
    --data_root /mnt/afs/acv/xuchenjie/projects/data/mvtec_ad \
    --categories bottle hazelnut metal_nut toothbrush \
    --methods dsr \
    --timeout 7200 \
    --output runs/alignment/dsr_v5_part0.json

CUDA_VISIBLE_DEVICES=1 python tools/benchmark.py \
    --data_root /mnt/afs/acv/xuchenjie/projects/data/mvtec_ad \
    --categories cable carpet screw \
    --methods dsr \
    --timeout 7200 \
    --output runs/alignment/dsr_v5_part1.json

CUDA_VISIBLE_DEVICES=2 python tools/benchmark.py \
    --data_root /mnt/afs/acv/xuchenjie/projects/data/mvtec_ad \
    --categories grid leather pill wood \
    --methods dsr \
    --timeout 7200 \
    --output runs/alignment/dsr_v5_part2.json

CUDA_VISIBLE_DEVICES=3 python tools/benchmark.py \
    --data_root /mnt/afs/acv/xuchenjie/projects/data/mvtec_ad \
    --categories capsule tile transistor zipper \
    --methods dsr \
    --timeout 7200 \
    --output runs/alignment/dsr_v5_part3.json
```
Summary of results:

| Metric | Reference | BaoIAD | Gap |
|--------|-----------|----------|-----|
| image_auroc | `0.943` | `0.9399` | `-0.3%` |
| pixel_auroc | — | `0.8426` | — |

illustrate:

- This round has completed the fresh `15/15` full benchmark and merged it into `runs/alignment/dsr_v5.json`
- Since DSR single class takes a long time, this round uses 4 groups of parallel shards to run; the first round `cable / screw / wood` was not completed due to `7200s` timeout, and was subsequently rerun with `14400s` timeout single class and successfully closed.
- The image mean value of fresh full benchmark is completely consistent with the historical archive and is still `0.9399`; the pixel mean value is updated to `0.8426`
- The main weak class is still `screw (image_auroc = 0.6118)`, which is consistent with the conclusion that `screw` is a significantly weak class in the historical archives

Shutdown line inspection:

- [x] fresh `15/15` results do not appear in large areas near `0.5` image AUROC
- [x] No unified platform value collapse occurs
- [x] probe / smoke / fresh full benchmark There are no new structural conflicts between the three.
- [x] fresh full benchmark original product has been archived to `runs/alignment/dsr_v5.json`

## 6. Guard

- New/enhanced test: `tests/test_models/test_detectors/test_dsr.py`
- Added document guard: `docs/alignment/dsr_checklist.md`
- Added new anti-regression points:
  - The loss paths of phase 2 / phase 3 must return finite values.
  - predict paths must output limited `pred_score` and `(1,H,W)` anomaly maps
  - `image_score` must come from the pre-upsampling pooled map, not the final upsampled map
  - After `set_epoch_info()` triggers phase switching, only `upsampling_module` remains trainable
  - `pretrained_vqvae_path='auto'` must be able to recognize the old format local checkpoint and use the official re-download method if necessary.
- If you change these paths later, you must rerun:
  - `baoiad/models/detectors/dsr.py`
  - `configs/dsr/dsr_256_mvtec_strict.py`
  -`tests/test_models/test_detectors/test_dsr.py`
  - `python tools/alignment_probe.py configs/dsr/dsr_256_mvtec_strict.py --splits train test --max-batch-size 1 --device cuda --output runs/alignment/dsr_probe.json`

## 7. Residual Risk

- `gh-proxy` may still be slow to pull DSR weights for the first time; although the current machine has finally successfully obtained compatibility with `.pckl`, it is still recommended to confirm the download link first in the new environment.
- The fresh image AUROC of `screw` is still only `0.6118`. Although this is consistent with the historical weak class trend, it is still the main stability/generalization weakness of DSR.
- The current `bottle` smoke budget is only `2 epochs`, which proves the health of the training/verification path and does not represent the final single-class optimal indicator.

## 8. Conclusion

- Final decision: `aligned`
- Allowed to proceed to next stage: `yes`
- If not allowed, next action: If the DSR conclusion needs to be further improved in the future, only conduct targeted diagnosis for `screw` instead of reopening the full benchmark

## Alignment checklist

Status only allowed: `matched | mismatch-fixed | intentional-diff | open`

## 1. Enter path

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| train color channels | `src/anomalib/pre_processing` + `examples/configs/data/mvtec.yaml` | `configs/dsr/dsr_256_mvtec_strict.py` + `LoadImage(to_rgb=True)` | Training images are fed into the model in `RGB` order | `LoadImage` defaults to `to_rgb=True`; `runs/alignment/dsr_probe.json` passes real `bottle/train/good/072.png` | matched |
| test color channel | Same as above | Same as above | Test image and training image remain the same `RGB` caliber | train/test pipeline symmetry; `bottle/test/broken_large/000.png` structure in probe is normal | matched |
| resize / crop | `lightning_model.py::configure_pre_processor()` | `ResizeAD(size=256, backend='pillow')` | Only resize to `256x256`, no additional crop | config has the same caliber as anomalib `Resize(image_size)` | matched |
| normalization / value range | `configure_pre_processor()` + `on_train_start()` | `ScaleNormalizeAD()` | Input remains `[0,1]`, disable ImageNet Normalize | DSR config explicitly uses `ScaleNormalizeAD`; probe input statistics fall in `[0.1159, 1.0000]` / `[0.1177, 1.0000]` | matched |
| batch size / runtime | `examples/configs/data/mvtec.yaml` | `configs/dsr/dsr_256_mvtec_strict.py` | anomalib sample data configuration is `train/eval batch_size=32`; BaoIAD currently retains a more conservative single-class runtime | current strict config is `batch_size=4`, `val_interval=20`, used to reduce local benchmark overhead | intentional-diff |

## 2. Anomaly Synthesis

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| Phase 2 Perlin mask generation | `src/anomalib/models/image/dsr/anomaly_generator.py` | `_generate_perlin_anomaly_mask/_batch()` | Randomly scaled Perlin mask, threshold `0.5`, rescale when all are below the threshold | `test_generate_perlin_anomaly_batch_returns_binary_masks` | matched |
| clean/anomaly sampling probability | `DsrAnomalyGenerator(p_anomalous=0.5)` | `_generate_perlin_anomaly_batch(..., p_anomalous=0.5)` | Default 50% to retain clean samples | Default parameters consistent | matched |
| Phase 3 texture enhancer | `data/utils/generators/perlin.py::PerlinAnomalyGenerator` | `phase3_augmenters` | `MultiRandomChoice(num_transforms=3)`, including enhancements such as solarize / affine / autoaugment | `test_phase3_augmenter_matches_reference_shape` | matched |
| beta range/blending | `PerlinAnomalyGenerator(blend_factor=(0.2, 1.0))` | `_forward_phase3_loss()` | `beta ∈ [0.2, 1.0]`, blending by `img*(1-mask) + beta*anom*mask + (1-beta)*img*mask` | code paths aligned with anomalib current generator | matched |

## 3. Reconstruct branch

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| Discrete latent model structure | `src/anomalib/models/image/dsr/torch_model.py::DiscreteLatentModel` | `DiscreteLatentModel` | Double-layer VQ-VAE + general decoder has the same structure | The key module naming and connection methods are consistent; `test_forward_tensor` covers the output key | matched |
| Subspace restriction module | `SubspaceRestrictionModule` | Module with the same name | Both hi/lo branches use InstanceNorm AE to restore no exception embedding | The current implementation is consistent with the reference structure | matched |
| object-specific decoder | `ImageReconstructionNetwork` | Module with the same name | Reconstruct object-specific images from reprojected hi/lo embedding | Phase 2/3 loss paths can produce limited loss | matched |
| Pre-training VQ-VAE weight loading | `lightning_model.py::prepare_pretrained_model()` | `_resolve_vqvae_path()` + `_try_load_discrete_latent_model_weights()` | Use discrete model weights with anomalib compatible keys; local old weights should not directly cause crashes | Added auto compatibility check and re-download fallback; `test_auto_pretrained_loading_retries_download_for_incompatible_local_checkpoint` | mismatch-fixed |

## 4. Discriminate / Upsampling branch

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| anomaly detection input splicing | `AnomalyDetectionModule.forward()` | module with the same name | use `[obj_spec_image, gen_image]` splicing to predict 2-type mask | current implementation consistent | matched |
| Number of output categories | `out_channels=2` | `anomaly_map_dim = 2` | mask logits are two categories | Current implementation is consistent | matched |
| upsampling input | `UpsamplingModule.forward()` | module with the same name | use `[obj_spec_image, gen_image, out_mask_sm]` for upsampling refinement | current implementation consistent | matched |
| phase 2 -> 3 switch | `training_step()` + `second_phase` | `set_epoch_info()` / `_maybe_switch_phase()` | only train upsampling module after reaching `upsampling_train_ratio` | `test_set_epoch_info_switches_to_phase3_and_freezes_non_upsampling_params` | matched |

## 5. Loss

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| hi/lo feature MSE | `loss.py::DsrSecondStageLoss` | `_forward_phase2_loss()` | `recon_feat_hi/lo` Do MSE on `embedding_bot/top` respectively | The formula is consistent with the reference | matched |
| image reconstruction MSE | Same as above | `_forward_phase2_loss()` | `MSE(obj_spec_image, input) * 10` | Currently implements explicit multiplication `10` | matched |
| segmentation focal | `DsrSecondStageLoss` / `DsrThirdStageLoss` | `_forward_phase2_loss()` / `_forward_phase3_loss()` | Perform focal loss on the second type logits and `anomaly_mask.squeeze(1).long()` | `FocalLoss(alpha=1.0, gamma=2.0)`, the input form is consistent | matched |
| phase 2 loss path | `training_step()` | `forward(mode='loss')` phase 2 | loss dict non-empty and finite | `test_forward_loss_phase2_is_finite` + probe `train.loss.all_finite` | matched |
| phase 3 loss path | `training_step()` | `forward(mode='loss')` phase 3 | upsampling phase loss non-empty and finite | `test_forward_loss_phase3_is_finite` | matched |

## 6. Predict / Scoring

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| anomaly map source | `torch_model.py` | `_forward_predict()` | `anomaly_map = softmax(upsampled_mask)[:,1]` | `test_predict_uses_pre_upsampling_scores_and_post_upsampling_maps` | matched |
| pooling | `avg_pool2d(..., 21, stride=1, padding=10)` | `_forward_predict()` | image score First do `21x21` average pooling on the pre-upsampling map | Same as above | matched |
| image score aggregation | `amax` over pooled map | `_forward_predict()` | `pred_score = max(pool21(anomaly_prob))` | Same as above | matched |
| predict output shape / finiteness | `InferenceBatch(pred_score, anomaly_map)` | `build_predict_results()` | return finite `pred_score` and `(1,H,W)` map | `test_forward_predict_outputs_finite_scores_and_maps` + probe `test.predict.*` | matched |

## 7. Behavior verification conclusion

- [x] After fixing the seed, the dataset sample structure is as expected
- [x] mask shape and range are as expected
- [x] The key intermediate quantities of the loss path have made shape / range / finiteness assertions
- [x] predict path's score / map makes shape / range / finiteness assertions
- [x] `bottle` smoke does not trigger the structural shutdown line

## 8. Remarks

- The original old format `pre_trained/vq_model_pretrained_128_4096.pckl` in the current repo has been replaced by the official compatibility weight; even if the old file appears again later, the `auto` path will be checked for compatibility first.
- `pretrained_vqvae_path='auto'` is still an engineering interface of BaoIAD and does not belong to the original API of anomalib; it will now give priority to trying local compatible weights and fall back to the official download if it is not compatible.
- This turn has completed Gate 1-4: checklist, guard, probe, `bottle` smoke and fresh `15/15` full benchmark have been archived.
- The fresh full benchmark uses `dsr_v5_part0-3.json` + `retry_cable/screw/wood.json` and merges them into `runs/alignment/dsr_v5.json`; this is to shorten the wall time, not a difference in model caliber.
