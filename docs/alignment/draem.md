# DRAEM strict-alignment evidence

- **Method slug**: `draem`
- **Family**: Self-supervised synthesis
- **Method README**: [`configs/draem/README.md`](../../configs/draem/README.md)

This file preserves the fixed alignment evidence migrated from the manuscript evidence workspace. The content below is translated to English where needed and kept inside this repository so the strict-alignment state can be reviewed without relying on an external paper checkout.

## Config entry points

- [`configs/draem/draem_256_mvtec_strict.py`](../../configs/draem/draem_256_mvtec_strict.py)
- [`configs/draem/draem_256_visa.py`](../../configs/draem/draem_256_visa.py)

## Detailed alignment report

**Status**: `playbook-complete`
**Date**: `2026-04-02`

## 1. Reference freezing

- Reference warehouse: `https://github.com/VitjanZ/DRAEM`
- Reference commit: `2dbf67397ab5c10a1494e5ae70ab59a25d7c35ef`
- Refer to config / checkpoint: official `train_DRAEM.py` / `test_DRAEM.py` single-class training and testing path
- Current mainline configuration: `configs/draem/draem_256_mvtec_strict.py`
- Dataset/Category: MVTec AD single class training + DTD texture data
- Input resolution: `256x256`
- seed: The official script is not explicitly fixed; BaoIAD’s probe / smoke / benchmark are unified according to `seed=42`
- Indicator definition: image AUROC / pixel AUROC
- compatibility knobs:
  - Reserve `dtd_path='auto'` as BaoIAD compatible entry
  - Keep the `anomaly_ratio` / `beta_range` configuration items; the default values are consistent with the official

## 2. Conclusion

See [`draem_checklist.md`](draem_checklist.md) for the control matrix.

- Official DRAEM's Perlin, data synthesis, color channel, SSIM, test resize and GT mask resize semantics have been aligned item by item.
- `probe`, `bottle` smoke, strict `15/15` rerun and official-compatible test-only summaries have been completed.
- The current main line of playbook acceptance is strictly official-compatible, and the intermediate conclusion of the early `not aligned yet` will no longer be used.
- Historical aligned-plus results continue to be retained as a supplementary archive to explain image-score improvements on `screw`, but no longer define the current playbook acceptance conclusion.

## 3. Critical code path conclusion

### Consistency confirmed

- The discriminative branch splicing order is still `[reconstruction, augmented/input]`
- The overall loss structure of `MSE + SSIM + Focal` is still consistent with the official one
- Training/Testing/DTD Texture Preservation `BGR`
- Test resize path remains official `cv2.resize`
- test GT mask resize semantics have been completed and fixed

### Fixed inconsistencies

- The reconstruct branch removes the historical deviation path and restores it to the official caliber
- focal loss reverted to official `alpha=None` semantics
- Perlin mask generation is restored to the official rotation and threshold logic, and the anomalib style rescale is no longer retained.
- DRAEM dataset restores official random resampling and internal clean/anomaly determination order
- SSIM reverted to official dynamic value-range version

### Supplementary archiving instructions

- strict official-compatible archive size: `pool21_max + last checkpoint(epoch_300)`
- History aligned-plus archive size: `pool7_max + best_balanced`
- The current mainline of the current warehouse is only defined with strict official-compatible `playbook-complete`

## 4. Probe / Smoke / Guard

Execution entry:

```bash
python tools/alignment_probe.py configs/draem/draem_256_mvtec_strict.py \
    --splits train test \
    --output runs/alignment/draem_probe.json
```
in conclusion:

- `runs/alignment/draem_probe.json` passed, the batch, loss, and predict structures of train/test are all normal.
- `bottle` `1 epoch` smoke passed and the results are `image_auroc=0.8698`, `pixel_auroc=0.7905`.
- guard has been completed:
  - `tests/test_datasets/test_draem_dataset.py`
  - `tests/test_models/test_detectors/test_draem.py`

The current fixed anti-regression point of guard:

- Perlin mask does not allow fallback to anomalib rescale logic
- SSIM is no longer allowed to fix to wrong `[0,1]` value range
- dataset no longer allows changing the train/DTD channel to `RGB`

## 5. Full Benchmark and Archived Results

The summary of strict `15/15` rerun and official-compatible test-only has been completed, and the current archive conclusions are as follows.

| Caliber | Image AUROC | Pixel AUROC | Description |
|------|-------------|-------------|------|
| Official Freeze Reference | `0.940` | — | Official README / Code Path Agent Reference |
| strict official-compatible | `0.9709` | `0.9649` | `pool21_max + last checkpoint(epoch_300)`; archived in `runs/alignment/draem_official_last_pool21.json` |
| History aligned-plus supplementary archive | `0.9855` | `0.9718` | `pool7_max + best_balanced`; major gains concentrated in `screw` |
| Legacy baseline before restoration | `0.7806` | `0.6040` | Only retained as old caliber record and no longer used as current conclusion |

illustrate:

- strict official-compatible is above the frozen reference `image_auroc=0.940`, meeting the current closing requirements.
- aligned-plus results remain as supplementary material, but do not back-define the strict mainline.
-Historical baseline and mid-way targeted diagnoses will continue to be retained in the running archive and are only used to trace the problem location process.

## 6. Remaining instructions

- The official training script still does not explicitly fix the random seed; reproducible conclusions in the warehouse continue to be archived under the BaoIAD caliber of `seed=42`.
- Historical diagnostics on image-score aggregation on `screw` have been completed and archived, but these analyzes no longer block DRAEM's playbook closure.
- The current conclusions at the method level are subject to this document; `docs/alignment/README.md` and `docs/alignment/CONFIG_MATRIX.md` are only synchronized summaries.

## 7. Conclusion

- Final decision: `playbook-complete`
- Current mainline configuration: `configs/draem/draem_256_mvtec_strict.py`
- Current mainline results: strict official-compatible `image_auroc=0.9709`, `pixel_auroc=0.9649`
- Historical aligned-plus results `image_auroc=0.9855`, `pixel_auroc=0.9718` continue to be retained as supplementary archives and do not replace the strict acceptance conclusion

## Alignment checklist

Status only allowed: `matched | mismatch-fixed | intentional-diff | open`

## 1. Enter path

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| train color channel | `VitjanZ/DRAEM:data_loader.py` | `baoiad/datasets/draem_dataset.py` | Training images maintain `BGR` order of `cv2.imread` | `RGB` conversion of training images and DTD textures removed; `test_draem_dataset_keeps_bgr_channels` | mismatch-fixed |
| test color channels | `VitjanZ/DRAEM:test_DRAEM.py` | `configs/draem/draem_256_mvtec.py` + `LoadImage` | test image maintains `BGR` order | `LoadImage(to_rgb=False)`; `test_load_image_keep_bgr` | mismatch-fixed |
| DTD/texture color channels | `VitjanZ/DRAEM:data_loader.py` | `baoiad/datasets/draem_dataset.py` | DTD textures maintain `BGR` order | `RGB` conversion of DTD textures removed | mismatch-fixed |
| resize / crop | `VitjanZ/DRAEM:data_loader.py` | `DRAEMDataset` + `configs/draem/draem_256_mvtec.py` | train/test only use `cv2.resize` to `256x256`, no additional crop | test pipeline changed from `pillow` back to `cv2` backend | mismatch-fixed |
| normalization / value range | `VitjanZ/DRAEM:data_loader.py` | `DRAEMDataset` + `ScaleNormalizeAD` | The train/test input remains `[0, 1]`, without ImageNet normalization | The current train/test configuration is consistent | matched |

## 2. Anomaly Synthesis

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| Perlin mask generation | `VitjanZ/DRAEM:data_loader.py` + `perlin.py` | `generate_perlin_mask()` | Uses official Perlin + `imgaug` rotation; does not do anomalib-style rescale | `test_generate_perlin_mask_does_not_rescale_low_noise` | mismatch-fixed |
| Texture blending formula | `VitjanZ/DRAEM:data_loader.py` | `DRAEMDataset._generate_anomaly()` | `image*(1-mask) + (1-beta)*texture*mask + beta*image*mask` | Code paths aligned | matched |
| beta range | `VitjanZ/DRAEM:data_loader.py` | `configs/draem/draem_256_mvtec.py` | `beta = rand()*0.8` | `beta_range=(0.0, 0.8)` | matched |
| clean/anomaly sampling probability | `VitjanZ/DRAEM:data_loader.py` | `DRAEMDataset._generate_anomaly()` | Default 50% returns clean, 50% returns anomaly | The sampling timing has been moved back to the synthesis path; retains `anomaly_ratio` compatible parameters | mismatch-fixed |

## 3. Reconstruct branch

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| Encoder structure | `VitjanZ/DRAEM:model_unet.py` | `EncoderReconstructive` | 5-stage encoder, no skip output | Current implementation consistent | matched |
| Decoder structure | `VitjanZ/DRAEM:model_unet.py` | `DecoderReconstructive` | 4-stage decoder, no skip connection | Current implementation consistent | matched |
| Output activation | `VitjanZ/DRAEM:train_DRAEM.py` | `DRAEMDetector.reconstruct()` | reconstruct output is raw decoder output, no sigmoid | Removed `sigmoid` | mismatch-fixed |
| loss input | `VitjanZ/DRAEM:train_DRAEM.py` | `DRAEMDetector.forward(mode='loss')` | Use `recon(augmented)` to do `MSE + SSIM` on the original image | The current implementation is consistent | matched |

## 4. Discriminate branch

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| Splicing order | `VitjanZ/DRAEM:train_DRAEM.py` | `DRAEMDetector.forward()` | `[reconstruction, augmented/input]` | The current implementation is consistent | matched |
| Output the number of categories | `VitjanZ/DRAEM:model_unet.py` | `DecoderDiscriminative` | Output 2 categories of mask logits | Current implementation consistent | matched |
| upsample path | `VitjanZ/DRAEM:model_unet.py` | `DecoderDiscriminative` | 5-stage bilinear upsample path | current implementation consistent | matched |
| skip connections | `VitjanZ/DRAEM:model_unet.py` | `DecoderDiscriminative` | full skip connections | current implementation consistent | matched |

## 5. Loss

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| MSE input form | `VitjanZ/DRAEM:train_DRAEM.py` | `DRAEMDetector.forward(mode='loss')` | `MSE(gray_rec, gray_batch)` | Current implementation consistent | matched |
| SSIM input form | `VitjanZ/DRAEM:loss.py` | `SSIMLoss` | Use official dynamic value-range SSIM | `test_ssim_loss_uses_dynamic_value_range_for_negative_reconstruction` | mismatch-fixed |
| focal input form | `VitjanZ/DRAEM:train_DRAEM.py` + `loss.py` | `FocalLoss` | `softmax(pred)` + `alpha=None` | `alpha=None` restored | mismatch-fixed |
| loss weight | `VitjanZ/DRAEM:train_DRAEM.py` | `configs/draem/draem_256_mvtec.py` | `l2 + ssim + focal` equal weight addition | `ssim_weight=1.0` | matched |
| reduction | `VitjanZ/DRAEM:loss.py` | `FocalLoss` + `SSIMLoss` | `mean` reduction | current implementation consistent | matched |

## 6. Predict / Scoring

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| anomaly map source | `VitjanZ/DRAEM:test_DRAEM.py` | `DRAEMDetector.forward(mode='predict')` | take `softmax(pred)[:, 1]` | current implementation consistent | matched |
| pooling | `VitjanZ/DRAEM:test_DRAEM.py` | `DRAEMDetector(score_mode='pool7_max')` | The official is `21x21` pooling; the current aligned config uses `7x7` to retain small defects | `pool7_max` on `screw` is significantly better than `pool21_max`, and `cable` basically does not degrade | intentional-diff |
| image score aggregation | `VitjanZ/DRAEM:test_DRAEM.py` | `max` over pooled map | The aggregation form is still `max`, but the pooling kernel is changed to `7` | Same as above | intentional-diff |
| Post-processing / smoothing | `VitjanZ/DRAEM:test_DRAEM.py` | `predict()` | No additional post-processing | Current implementation consistent | matched |

## 7. Behavior verification conclusion

- [x] After fixing the seed, the dataset sample structure is as expected
- [x] mask shape and range are as expected
- [x] The key intermediate quantity of the loss path has a shape / range assertion.
- [x] predict path's score / map makes shape / range assertions
- [x] bottle smoke did not trigger the abnormal shutdown line

## 8. Remarks

- `dtd_path='auto'` is a compatible interface reserved by BaoIAD and does not belong to the official warehouse behavior; the default value does not affect the official caliber.
- The `anomaly_ratio` parameter is reserved for configuration compatibility; the default `0.5` is consistent with the official one.
- `score_mode='pool7_max'` is currently the only intentional diff for DRAEM and is used to alleviate the image-level score inversion problem of `screw`.
