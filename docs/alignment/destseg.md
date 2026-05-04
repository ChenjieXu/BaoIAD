# DeSTSeg strict-alignment evidence

- **Method slug**: `destseg`
- **Family**: Knowledge distillation
- **Method README**: [`configs/destseg/README.md`](../../configs/destseg/README.md)

This file preserves the fixed alignment evidence migrated from the manuscript evidence workspace. The content below is translated to English where needed and kept inside this repository so the strict-alignment state can be reviewed without relying on an external paper checkout.

## Config entry points

- [`configs/destseg/destseg_rn18_256_mvtec_strict.py`](../../configs/destseg/destseg_rn18_256_mvtec_strict.py)
- [`configs/destseg/destseg_rn18_256_visa.py`](../../configs/destseg/destseg_rn18_256_visa.py)

## Detailed alignment report

**Status**: `playbook-in-progress` (constructor bug has been fixed; fresh current mainline has been corrected, waiting for fresh full rerun)
**Main Reference**: Local `.refs/destseg` `main@f6ea31fb5b097698b195f85b1d5e3efaedce9eb6`
**Supplementary evidence**: `zhangzjn/ADer`
**Main Configuration**: `configs/destseg/destseg_rn18_256_mvtec_strict.py`
**Compatible Aliases**: `configs/destseg/destseg_wrn50_256_mvtec.py`
**Checklist**: [`destseg_checklist.md`](destseg_checklist.md)

## Implementation of this round

### 1. Switch the main training line back to the official step-based protocol

Although the old version is close to the official structure, the training protocol is still an approximation of `300 epoch + phase_ratio`, not the official `5000 steps / the first 1000 steps training the student`.

Currently changed to:

- `train_cfg = dict(by_epoch=False, max_iters=5000, val_interval=5000)`
- `model.de_st_steps = 1000`
- `train/test num_workers = 16` (return to official `train.py` default)
- Remove unofficial scheduler
- Retain the official three sets of learning rates:
  - `student_net`: `0.4`
  - `segmentation_net.res`: `0.1`
  - `segmentation_net.head`: `0.01`

### 2. Data synthesis is migrated from the detector to the train pipeline

The old version temporarily performs Perlin synthesis in `forward(mode='loss')`, and neither probe nor checklist can directly observe the train sample structure.

Currently added:

- `baoiad/datasets/transforms/destseg.py::DeSTSegAugment`
- `baoiad/datasets/transforms/destseg.py::PackDeSTSegInputs`

Now train batch will explicitly output:

- `inputs = img_aug`
- `data_sample.img_origin`
- `data_sample.img_aug`
- `data_sample.gt_mask`

This path is already consistent with the official one:

- DTD texture reading
- Category related rotation strategies
-Perlin mask
- `beta in [0, 0.8)`
-ImageNet normalize

Alignment.

### 3. Switch the model trunk back to the official `timm` code form

Currently `DeSTSegDetector` has been changed to:

- teacher: `timm.create_model('resnet18', pretrained=..., features_only=True, out_indices=[1,2,3])`
- strict config is now fixed to `teacher_pretrained=True` and no longer uses the legacy `auto` checkpoint mainline
- student: `timm.create_model('resnet18', pretrained=False, features_only=True, out_indices=[1,2,3,4])`
- segmentation head: official `res + ASPP + conv + sigmoid`
- image score: `top-100 mean`

At the same time, the initialization order of the student / segmentation branch has been corrected to match the official implementation, instead of continuing to use the historical handwritten `torchvision` trunk.

### 4. compile blocker has been converted to explicit guard

The direct blocking of the historical `4/15` benchmark is the poor compatibility between DeSTSeg and `torch.compile`.

Currently added:

- config level `train_disable_compile = True`
- config level `benchmark_disable_compile = True`
- runtime opt-out for `tools/train.py` / `tools/test.py` / `tools/benchmark.py`

Targeted test is also added to prevent subsequent infrastructure from resetting compile back to DeSTSeg.

### 5. The root cause of full strict stop-line has converged to "single optimizer missing segmentation branch"

In the first full rerun of fresh strict `15/15`, categories such as `bottle / leather / transistor` have obvious image-side stop-lines, for example:

- `bottle image_auroc = 0.1294`
- `leather image_auroc = 0.0914`

After further investigation, it was confirmed that the old strict config had only one optimizer, and it was built in `_phase='student'`, so there was only `student_net` parameter in the optimizer and no `segmentation_net` parameter. When the second half of the phase switches to segmentation, although loss is being calculated, the segmentation branch is not actually updated by the optimizer.

Currently fixed to:

- `baoiad.DeSTSegOptimWrapperConstructor`
- `OptimWrapperDict(student, segmentation)`
- `DeSTSegDetector.train_step()` only updates the corresponding optimizer according to phase

And added `constructor + train_step` single test guard.

### 6. There is also a parameter deepcopy bug in the split optimizer constructor, which has been fixed.

While continuing to pursue the main line of `strictteacher_v2`, I located a lower-level problem:

- `DeSTSegOptimWrapperConstructor` after stuffing the real model parameters into `optimizer_cfg['params']`,
  And did `deepcopy` on the entire optimizer cfg
- This will copy the `Parameter` object into a free copy. What the optimizer actually updates is not the model parameters.

This explains why the previous round appears:

- `loss` has value
- `grad` non-zero
- But the model parameters remain completely unchanged after one-step training

Currently this problem has been fixed by:

- optimizer cfg only deep copies non-`params` fields
- `params` retains live `Parameter` reference

And added a stronger guard:

- `tests/test_engine/test_destseg_optim_wrapper_constructor.py`
  - Assert that the objects in the optimizer param group are the model parameters themselves
- `tests/test_models/test_detectors/test_destseg.py`
  - Assert that the wrappers built by the constructor can really update the real model parameters

Therefore, the following products should now be regarded as `stale evidence`, and will only be retained as historical positioning materials, and will no longer be used as the main conclusion of the current strict:

- `runs/alignment/destseg_bottle_smoke_compressed_strictteacher_v2`
- `runs/alignment/destseg_bottle_smoke_compressed_defaultsampler_v1`
- `runs/alignment/destseg_bottle_exact1000_strictteacher_v2`
- `runs/alignment/destseg_strict_merged_progress.json`

## Completed lightweight verification

- `pytest tests/test_models/test_detectors/test_destseg.py -q`
- `pytest tests/test_utils/test_alignment_probe.py tests/test_utils/test_benchmark_config_detection.py::test_destseg_benchmark_prefers_rn18_strict_config tests/test_utils/test_benchmark_config_detection.py::test_destseg_benchmark_disables_compile tests/test_utils/test_benchmark_config_detection.py::test_prepare_subprocess_env_can_disable_compile -q`
- strict probe:

```bash
python tools/alignment_probe.py configs/destseg/destseg_rn18_256_mvtec_strict.py \
    --splits train test \
    --max-batch-size 1 \
    --device cpu \
    --output runs/alignment/destseg_probe_strict_cpu.json
```
`destseg_probe_strict_cpu.json` Confirmed:

- `img_origin / img_aug / gt_mask` is visible in train sample
- `loss` Limited paths
- `predict` path's `pred_score / pred_anomaly_map` limited

## New evidence for this round

### 1. The number of workers returns to the official default `16`

The first version of strict config still uses the warehouse's common dataloader default `num_workers=4`, which will slow down DeSTSeg's DTD + Perlin train pipeline too slowly.

Currently changed to:

- `train_dataloader.num_workers = 16`
- `test/val_dataloader.num_workers = 16`
- `benchmark_keep_dataloader_workers = True`

Short throughput calibration:

- `runs/alignment_check/destseg_bottle_speedcheck_w16/20260327_164846/20260327_164846.log`
- `6 iter` next `time ≈ 24.61s/iter`

Compared to the previous `workers=4`:

- `runs/alignment/destseg_bottle_smoke_v1/20260327_164142/20260327_164142.log`
- `6 iter` next `time ≈ 30.54s/iter`

Note that worker alignment does improve throughput, but strict full budget is still heavy.

### 2. `bottle` compressed smoke has been passed

To complete playbook Gate 3, a clearly marked compressed smoke is added in this round:

```bash
CUDA_VISIBLE_DEVICES=3 python tools/train.py configs/destseg/destseg_rn18_256_mvtec_strict.py \
    --work-dir runs/alignment/destseg_bottle_smoke_compressed_v1 \
    --cfg-options train_cfg.max_iters=60 train_cfg.val_interval=60 model.de_st_steps=12 default_hooks.logger.interval=10
```
Key log:

- `runs/alignment/destseg_bottle_smoke_compressed_v1/20260327_165438/20260327_165438.log`
- student phase: `iter 6 -> loss 37.8005`, `iter 10 -> loss 31.0420`
- segmentation phase: `iter 20 -> loss 3.8564`, `iter 30/40/50/60 -> loss ≈ 0.603`

Last `bottle` verification:

- `image_auroc = 0.7817`
- `pixel_auroc = 0.5159`
- `image_ap = 0.9357`

This means:

- Training did not explode after phase switching
- image score did not collapse to the `0.5` platform
- The current strict mainline at least has the structural conditions to continue advancing to full benchmark.

But this is still only a low-cost proof of Gate 3, not a strict final result.

### 2.1 compressed smoke has been confirmed to be "historical log evidence", not a reproducible value in the current strict mainline

Subsequently, use the current warehouse `tools/test.py` to check the same checkpoint.
`runs/alignment/destseg_bottle_smoke_compressed_v1/epoch_1.pth` Retest, the results are:

- `image_auroc = 0.7032`
- `pixel_auroc = 0.5160`

The corresponding log is located at:

- `runs/alignment/destseg_bottle_smoke_compressed_v1_retest`

At the same time, the `vis_data/config.py` that comes with this checkpoint clearly shows that this `compressed_v1` is still used when running:

- `teacher_pretrained=True`
- Single `OptimWrapper + paramwise_cfg`

That is, it is not a direct outgrowth of the subsequent dual optimizer strict mainline. therefore:

- `0.7817 / 0.5159` is only retained as historical training log evidence
- The replay value that can be reproduced in the current warehouse should be based on `0.7032 / 0.5160`
- This `compressed_v1` can no longer directly serve as evidence that "the current strict mainline has been verified"

### 2.2 The fresh compressed smoke of the current strict config has been completed

After tightening the strict main configuration to:

- `teacher_pretrained=True`
- `OptimWrapperDict(student, segmentation)`

After that, the new `bottle` compressed smoke was re-run:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/destseg/destseg_rn18_256_mvtec_strict.py \
    --work-dir runs/alignment/destseg_bottle_smoke_compressed_strictteacher_v2 \
    --cfg-options train_cfg.max_iters=60 train_cfg.val_interval=60 model.de_st_steps=12 default_hooks.logger.interval=10
```
After fixing the parameter deepcopy bug of `DeSTSegOptimWrapperConstructor`, a new fresh smoke was run again:

```bash
CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/destseg/destseg_rn18_256_mvtec_strict.py \
    --work-dir runs/alignment/destseg_bottle_smoke_compressed_strictteacher_v3 \
    --cfg-options train_cfg.max_iters=60 train_cfg.val_interval=60 model.de_st_steps=12 default_hooks.logger.interval=10
```
Current archive results:

- Training log: `runs/alignment/destseg_bottle_smoke_compressed_strictteacher_v3`
- `Epoch(val)` Result: `image_auroc = 0.8008`, `pixel_auroc = 0.8119`

And the `tools/test.py` replay results for the same checkpoint are exactly the same:

- replay work dir: `runs/alignment/destseg_bottle_smoke_compressed_strictteacher_v3_retest`
- replay: `image_auroc = 0.8008`, `pixel_auroc = 0.8119`

This means:

- After fixing optimizer parameter binding, the fresh smoke of DeSTSeg’s current strict config has been significantly corrected.
- This result is not directly comparable to the previous stale `strictteacher_v2` (`0.6929 / 0.4454`) because the latter was generated on the broken constructor
- The current fresh smoke can finally be used as effective Gate 3 evidence of the "current strict mainline"

### 2.2.1 fresh `grid / transistor` compressed smoke has also been regularized

In order to confirm that `bottle` is not an accident, two fresh cheap smokes of the old weak stop-line class have been added:

- `runs/alignment/destseg_grid_smoke_compressed_strictteacher_v1`
  - `image_auroc = 0.7519`
  - `pixel_auroc = 0.7113`
- `runs/alignment/destseg_transistor_smoke_compressed_strictteacher_v1`
  - `image_auroc = 0.7379`
  - `pixel_auroc = 0.5726`

This means:

- Recovery of current strict mainline is not a `bottle` single-class accident
- At least on `bottle / grid / transistor`, fresh cheap smoke has been completely separated from the old stop-line platform

### 2.3 stale `strictteacher_v2` Evidence is only retained as positioning material

Before the constructor was fixed, the old `strictteacher_v2` resulted in:

- Training log: `runs/alignment/destseg_bottle_smoke_compressed_strictteacher_v2`
- `Epoch(val)` Result: `image_auroc = 0.6929`, `pixel_auroc = 0.4454`

At the same time, a replay of "only replace the current test pipeline" was made for the previous round of `strictteacher_v1` checkpoint:

- replay work dir: `runs/alignment/destseg_bottle_smoke_compressed_strictteacher_v2_maskretest`
- replay: `image_auroc = 0.6929`, `pixel_auroc = 0.4454`

This means:

- stale `strictteacher_v2` still has diagnostic value
- The conclusion that test-side mask alignment brings only a small pixel improvement still holds
- But it cannot continue to represent the smoke result of the "current strict mainline"

### 3. strict full The first round of results has triggered stop-line, and then the verification has been restarted according to the root cause repair

The results of partial strict full of the old single optimizer version are as follows:

- `bottle = 0.1294 / 0.4409`
- `cable = 0.3810 / 0.4568`
- `grid = 0.5831 / 0.5659`
- `hazelnut = 0.4200 / 0.3742`
- `leather = 0.0914 / 0.3381`
- `pill = 0.3898 / 0.6717`
- `screw = 0.5518 / 0.5825`
- `tile = 0.4076 / 0.4797`
- `transistor = 0.3800 / 0.5330`
- `wood = 0.5921 / 0.6378`

This round of results has been regarded as stop-line according to the playbook, and the old mainline results are no longer accumulated; currently, both bottle strict full and compressed smoke have been restarted for verification on the new mainline of dual optimizers.

## Latest full-run recycling

As of `2026-03-29 16:09 UTC`, the current strict full evidence has been recovered from the shard log:

- `runs/alignment/destseg_part0_recomputed.json`
- `runs/alignment/destseg_part1_recomputed.json`
- `runs/alignment/destseg_part2_recomputed.json`
- `runs/alignment/destseg_part3_recomputed.json`
- `runs/alignment/destseg_strict_merged_progress.json`

Currently only 4 categories produce final indicators:

- `bottle = 0.4746 / 0.5235`
- `grid = 0.3175 / 0.4833`
- `pill = 0.4468 / 0.6715`
- `transistor = 0.3104 / 0.5031`

`destseg_strict_merged_progress.json` of `_average` is:

- `image_auroc = 0.3873`
- `pixel_auroc = 0.5454`
- `num_categories = 4`

An additional 4 launched categories do not have final metrics:

- `cable`
- `hazelnut`
- `screw`
- `wood`

The latest logs of these four shards stop at around `2026-03-28 18:27 UTC`, and the final visible progress is only as follows:

- `cable iter 1500`
- `hazelnut iter 2350`
- `screw iter 2250`
- `wood iter 1100`

The remaining 7 categories have not yet started:

- `capsule`
- `carpet`
- `leather`
- `metal_nut`
- `tile`
- `toothbrush`
- `zipper`

The corresponding suspended process has stopped at `2026-03-29 16:09 UTC`, and this round of sharding will no longer be recorded as "running".

## replay review

In order to confirm whether the replay caliber is reliable, `tools/test.py` retest has been performed on the two intermediate checkpoints:

- `runs/alignment/destseg_bottle_exact1000_v1/epoch_1.pth`
  - retest: `image_auroc = 0.4925`, `pixel_auroc = 0.5753`
  - History log: `0.4956 / 0.5753`
- `runs/alignment/destseg_bottle_midcheck_v5/epoch_1.pth`
  - retest: `image_auroc = 0.4754`, `pixel_auroc = 0.5229`
  - History log: `0.4770 / 0.5228`

This means:

- The two key windows `1000 iter` and `1200 iter` are reproducible under the current code.
- The main inconsistency between the current replay and the historical log is concentrated in `compressed_v1`
- So at this stage it is more like a "historical smoke configuration drift" problem rather than an overall error in the current `tools/test.py` or metric caliber

## checkpoint branch comparison

Currently, `tools/destseg_targeted_diagnose.py` has been used to generate three unified branch comparisons of `bottle` checkpoint:

- `runs/alignment/destseg_bottle_checkpoint_compare_cuda.json`

The key results are as follows:

| checkpoint | segmentation `img/pxl` | segmentation `score_gap / region_delta` | DeST `img/pxl` | DeST `score_gap / region_delta` |
|------|------|------|------|------|
| `destseg_bottle_smoke_compressed_strictteacher_v1:epoch_1` | `0.6913 / 0.4431` | `+0.0022 / -0.0067` | `0.3802 / 0.3725` | `-0.0053 / -0.0172` |
| `destseg_bottle_exact1000_v1:epoch_1` | `0.4937 / 0.5754` | `~0 / ~0` | `0.5183 / 0.3635` | `+0.0013 / -0.0209` |
| `destseg_bottle_midcheck_v5:epoch_1` | `0.4762 / 0.5229` | `-0.0018 / +0.0060` | `0.5222 / 0.3634` | `+0.0012 / -0.0209` |

The conclusion that can be drawn directly at present is:

- The `DeST` branch is very weak pixel-side on these three checkpoints, and `region_delta` continues to be negative, indicating that the abnormal area is usually not brighter than the background
- The `segmentation` branch only retains a certain image-side separation with very-short smoke (`60/12`); after entering `1000/1000` and `1200/1000`, the image-side basically falls to `~0.5`
- Therefore, the next round of targeted diagnoses should give priority to explaining: why the image separation of the segmentation branch is only briefly established in very-short smoke, and why the region ordering of the DeST branch is reversed for a long time

A single checkpoint diagnose for the current fresh smoke `strictteacher_v2` has also been archived to:

- `runs/alignment/destseg_bottle_smoke_compressed_strictteacher_v2/destseg_targeted_diag.json`

Its key summary is:

- segmentation: `img=0.6913`, `pxl=0.4454`, `score_gap=+0.0022`, `region_delta=-0.0063`
- DeST: `img=0.3802`, `pxl=0.3742`, `score_gap=-0.0053`, `region_delta=-0.0169`

This further illustrates that even under the current strict config:

- image ranking of segmentation branch is still weak
- The direction of the abnormal area of the DeST branch is still reversed

### 2.4 sampler A/B only brings marginal image improvement, not the current main reason

For the current strict config, an additional `DefaultSampler(shuffle=True)` of `bottle 60/12` cheap smoke is run:

- `runs/alignment/destseg_bottle_smoke_compressed_defaultsampler_v1`

Its training log consistent with `tools/test.py` replay converges to:

- `image_auroc = 0.7056`
- `pixel_auroc = 0.4459`

Relative to the current mainline `PersistentShuffleSampler`:

- `image_auroc = 0.6929`
- `pixel_auroc = 0.4454`

It only reflects a small image side improvement, and the pixel is almost unchanged.

The corresponding branch diagnose has also been archived to:

- `runs/alignment/destseg_bottle_smoke_compressed_defaultsampler_v1/destseg_targeted_diag.json`

The key branches of the two lines are compared as follows:

| config | segmentation `img/pxl` | segmentation inverted `img/pxl` | DeST `img/pxl` | DeST inverted `img/pxl` |
|------|------|------|------|------|
| `strictteacher_v2 + PersistentShuffleSampler` | `0.6913 / 0.4454` | `0.5190 / 0.5546` | `0.3802 / 0.3742` | `0.6159 / 0.6258` |
| `strictteacher_v2 + DefaultSampler` | `0.7087 / 0.4460` | `0.5119 / 0.5540` | `0.3857 / 0.3754` | `0.6087 / 0.6246` |

This means:

- sampler changes are not the main reason for the current stop-line
- The `inverted` indicator is systematically higher on both segmentation / DeST branches
- Currently it is more like "branch polarity/ordering semantics reversed" rather than a simple data order issue

### 2.5 synthetic train diagnose proves that the augmentation mask itself is correct

In order to determine whether the problem is with the training target or only occurs on test real anomalies, we currently
`strictteacher_v2` checkpoint ran `train` split diagnose:

- `runs/alignment/destseg_bottle_smoke_compressed_strictteacher_v2/destseg_train_targeted_diag.json`

Among them, the newly added `augmentation_diff` (`mean(abs(img_aug - img_origin))`) and synthetic `gt_mask`
Height alignment:

- raw `pixel_auroc = 0.9984`
- raw `aupro = 0.9973`
- raw `region_delta = +0.9469`

On the same batch of synthetic train samples, the two branches of the model still show bias:

- segmentation raw `pixel_auroc = 0.4808`, `region_delta = -0.0019`
- segmentation inverted `pixel_auroc = 0.5192`, `region_delta = +0.0019`
- DeST raw `pixel_auroc = 0.4563`, `region_delta = -0.0058`
- DeST inverted `pixel_auroc = 0.5437`, `region_delta = +0.0058`

This means:

- The synthetic anomaly mask produced by `DeSTSegAugment + PackDeSTSegInputs` has no directional errors.
- The current phenomenon of "reverse is better" already exists on train synthetic data
- So the main suspect has further converged to branch output semantics/loss alignment, rather than augmentation mask or test-only domain gap

### 2.6 The `1000/1000` baseline of the current strict config has been completed

After the constructor bug was discovered, `exact1000_strictteacher_v2` should also be considered stale; therefore the current strict `1000/1000` is currently being re-run.
The baseline should be based on subsequent fresh reruns. The current `v2` result is only retained as the "pre-repair broken constructor" positioning material:

- `runs/alignment/destseg_bottle_exact1000_strictteacher_v2`

The result is:

- `image_auroc = 0.6349`
- `pixel_auroc = 0.4769`
- `aupro = 0.1390`

And `tools/test.py` replay is exactly the same:

- `runs/alignment/destseg_bottle_exact1000_strictteacher_v2_retest`
- replay: `0.6349 / 0.4769`

This means:

- The current strict config's `1000/1000` is significantly better than the old `exact1000_v1`'s `0.4925 / 0.5753`
- But it's still significantly lower than the current `60/12` smoke's `0.6929 / 0.4454`
- The current strict mainline still has the problem of "very-short smoke can still be separated, and image-side falls back after entering official `1000/1000`"

### 2.7 `de_st` Multi-scale aggregation A/B proves that `prod` is not the only suspect

For `destseg_bottle_exact1000_strictteacher_v2:epoch_1`, current `tools/destseg_targeted_diagnose.py`
Output `de_st_aggregations` (`prod / mean / max` and its inverted):

- Product: `runs/alignment/destseg_bottle_exact1000_strictteacher_v2/destseg_targeted_diag.json`

Key summary:

| aggregation | raw `img/pxl` | inverted `img/pxl` | region delta(raw/inv) |
|------|------|------|------|
| `prod` | `0.3429 / 0.3848` | `0.3071 / 0.6152` | `-0.0160 / +0.0160` |
| `mean` | `0.3381 / 0.3849` | `0.2738 / 0.6151` | `-0.0132 / +0.0132` |
| `max` | `0.3579 / 0.4463` | `0.3921 / 0.5537` | `-0.0078 / +0.0078` |

At the same time, three single-scale `de_st_per_scale` are also displayed:

- `scale0`: inverted is obviously better
- `scale1`: raw image is slightly better, but pixel is still weak
- `scale2`: inverted is obviously better

This means:

- `torch.prod` aggregation is indeed amplifying the mixed-polarity problem
- But even if changed to `mean` or `max`, the raw path is not directly restored to an acceptable level
- Currently it is more like "the multi-scale DeST map itself is semantically inconsistent, and at least two of the layers have reversed polarity" rather than just choosing the wrong aggregator

### 2.8 fresh `1200/1000` The current main line has obviously turned positive

After repairing the constructor, the current main line of fresh `1200/1000` re-run is:

- `runs/alignment/destseg_bottle_midcheck_strictteacher_v4`

Current results:

- `image_auroc = 0.9944`
- `pixel_auroc = 0.9837`
- `aupro = 0.9399`

And `tools/test.py` replay is exactly the same:

- `runs/alignment/destseg_bottle_midcheck_strictteacher_v4_retest`
  - `0.9944 / 0.9837`

This means:

- The current strict mainline has not continued to deteriorate after entering the segmentation phase.
- The old `1200/1000` stop-line evidence was entirely a stale result of the broken constructor
- Currently DeSTSeg is back to `playbook-in-progress`, the next step should be to move directly to a fresh full rerun rather than continue downtime around old stale results

## Current judgment

The old `stale` full result no longer represents the current strict mainline:

- old `destseg_strict_merged_progress.json` is the result of stale under broken constructor
- fresh current mainline has been turned positive overall on `bottle / grid / transistor` cheap smoke
- fresh `bottle 1200/1000` has also arrived `0.9944 / 0.9837`
- `augmentation_diff` is highly consistent with synthetic mask, indicating that it is not the train synthetic label itself that is reversed.
- `de_st_per_scale / aggregation` still prompts for mixed-polarity, but this no longer prevents fresh full rerun

Therefore the current state of DeSTSeg should be adjusted back from the old `stop-line` to `playbook-in-progress`:

- Allow fresh full rerun
- But old stale full results cannot continue to be mixed into the current main story conclusion

## Next step

- fresh current sanity Completed `10/15`: `bottle / grid / pill / transistor / cable / hazelnut / screw / wood / capsule / carpet`
- The current merged progress is fixed at `runs/alignment/destseg_current_sanity_progress_merged.json`, and the completed class mean is `img=0.9733 / pxl=0.9749`
- `leather` in `runs/alignment/benchmark_destseg_current_sanity_rest` did not write the verification index after train `5000/5000`, so it has been changed to single-category make-up run: `runs/alignment/benchmark_destseg_current_sanity_leather`
- The tail class is currently closed according to the parallel strategy:
  - `metal_nut`: `runs/alignment/benchmark_destseg_current_sanity_mttz`
  - `tile`: `runs/alignment/benchmark_destseg_current_sanity_tile`
  - `toothbrush / zipper`: `runs/alignment/benchmark_destseg_current_sanity_tzz`
  - `leather`: `runs/alignment/benchmark_destseg_current_sanity_leather`
- `mttz` The original serial fragment is only retained until `metal_nut` is closed; once `metal_nut` falls out of the verification index, its subsequent `tile / toothbrush / zipper` repeated calculations will be stopped.
- While fresh full rerun is in progress, continue to use `de_st_per_scale / aggregation / inverted` diagnostics to track whether the mixed-polarity phenomenon will still occur on weak classes.
- If weak classes fall to `~0.5` again in the fresh full rerun, priority should be given to targeted corrections around the `de_st_per_scale scale0/scale2` semantics.

## Add diagnostic tools

A checkpoint-level targeted diagnose script has been added to the current warehouse:

```bash
python tools/destseg_targeted_diagnose.py \
    configs/destseg/destseg_rn18_256_mvtec_strict.py \
    runs/alignment/destseg_bottle_smoke_compressed_v1/epoch_1.pth \
    runs/alignment/destseg_bottle_exact1000_v1/epoch_1.pth \
    --class-name bottle \
    --device cpu \
    --output runs/alignment/destseg_bottle_compare_diag.json
```
This script exports for each checkpoint:

- segmentation branch indicators and score/map statistics
- DeST branch indicators and score/map statistics
- Layer-by-layer statistics of three `output_de_st_list` scales
- Foreground/background region gap summary of several abnormal samples

## Alignment checklist

Status only allowed: `matched | mismatch-fixed | intentional-diff | open`

The primary reference is frozen as local `.refs/destseg` `main@f6ea31fb5b097698b195f85b1d5e3efaedce9eb6`. `ADer` is only used as auxiliary evidence and is no longer used as the main reference for implementation.

## 1. Enter path

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| train color channel | `.refs/destseg/data/mvtec_dataset.py` | `configs/destseg/destseg_rn18_256_mvtec_strict.py` + `baoiad/datasets/transforms/destseg.py` | train image enters the synthesis logic in RGB | `LoadImage` defaults to RGB; `runs/alignment/destseg_probe_dev.json` train sample `img_aug/img_origin` is `[3,256,256]` | `matched` |
| test color channel | `.refs/destseg/data/mvtec_dataset.py` | `configs/destseg/destseg_rn18_256_mvtec_strict.py` | test image also uses RGB + ImageNet normalize | `NormalizeAD` + probe test sample | `matched` |
| test mask resize / threshold | `.refs/destseg/data/mvtec_dataset.py` | `LoadMask(backend='pil', to_binary=False) + ResizeAD(mask_interpolation='bilinear') + ThresholdMask(0.5)` | test mask first bilinear resize the grayscale image, and then `0.5` binarize | `tests/test_datasets/test_transforms.py` + `test_destseg_strict_config_uses_soft_mask_resize_then_threshold` | `mismatch-fixed` |
| DTD / texture color channel | `.refs/destseg/data/mvtec_dataset.py` | `baoiad/datasets/transforms/destseg.py` | DTD texture read in as RGB and resize | `cv2.imread + cvtColor(BGR->RGB)` | `matched` |
| resize / crop | `.refs/destseg/constant.py` + dataset resize | `ResizeAD(size=256)` | training/test unified resize to `256x256` | config fixed `img_size=256` | `matched` |
| normalization / value range | `.refs/destseg/data/mvtec_dataset.py` | `DeSTSegAugment` / `NormalizeAD` | After synthesis and testing, both fall into the ImageNet normalized tensor | probe train/test inputs finite | `matched` |

## 2. Anomaly Synthesis

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| Perlin mask generation | `.refs/destseg/data/data_utils.py` | `baoiad/datasets/transforms/destseg.py` | Perlin scale `2^k`, threshold `0.5` binarization | Homology formula; probe train `gt_mask` range `0..1` | `matched` |
| Texture blending formula | `.refs/destseg/data/data_utils.py` | `DeSTSegAugment.transform()` | `image*(1-mask) + (1-beta)*texture*mask + beta*image*mask` | Code alignment item by item | `matched` |
| beta range | `.refs/destseg/data/data_utils.py` | `DeSTSegAugment(beta_max=0.8)` | `beta in [0,0.8)` | parameters fixed `0.8` | `matched` |
| clean/anomaly sampling probability | `.refs/destseg/train.py` (`aug_prob=1.0`) | `DeSTSegAugment` | train all uses anomaly-synth sample | transform always generates mask and overwrites `gt_label=1` | `matched` |

## 3. Teacher / Student backbone

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| teacher encoder | `.refs/destseg/model/destseg.py::TeacherNet` | `baoiad/models/detectors/destseg.py::TeacherNet` | `timm resnet18 features_only out_indices=[1,2,3]` + frozen | Code alignment item by item | `matched` |
| student coder | `.refs/destseg/model/destseg.py::StudentNet` | `baoiad/models/detectors/destseg.py::StudentNet` | `timm resnet18 features_only out_indices=[1,2,3,4]` | Code alignment item by item | `matched` |
| student decoder | Same as above | Same as above | `512->512->256->128->64` Level-by-level upsampling | Code alignment item by item | `matched` |
| segmentation head | `.refs/destseg/model/destseg.py::SegmentationNet` | `baoiad/models/detectors/destseg.py::SegmentationNet` | `res + ASPP + conv + sigmoid` | Code alignment item by item | `matched` |

## 4. Loss / Predict

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| phase switching protocol | `.refs/destseg/train.py` | `DeSTSegDetector.set_iter_info()` + config | The first `1000/5000` steps are student phase | config `de_st_steps=1000`, `train_cfg.max_iters=5000` | `mismatch-fixed` |
| optimizer split | `.refs/destseg/train.py` | `DeSTSegOptimWrapperConstructor` + `DeSTSegDetector.train_step()` | student / segmentation each use independent optimizer | constructor single test + train_step single test | `mismatch-fixed` |
| Optimizer parameter binding | The official optimizer must directly update the real model parameters | `DeSTSegOptimWrapperConstructor` | The object in the optimizer param group must be live `Parameter`, not a deepcopy copy | `test_destseg_optim_wrapper_constructor_builds_split_wrappers` + `test_constructor_built_wrappers_update_real_model_params` | `mismatch-fixed` |
| student phase loss | `.refs/destseg/train.py` | `forward(mode='loss')` | `cosine_similarity_loss(output_de_st_list)` | detector single test + probe train loss finite | `matched` |
| segmentation phase loss | `.refs/destseg/train.py` | `forward(mode='loss')` | `focal + l1` | detector single test | `matched` |
| image score aggregation | `.refs/destseg/eval.py` | `forward(mode='predict')` | segmentation map `top-100 mean` | code alignment item by item | `matched` |
| Post-processing / smoothing | `.refs/destseg/eval.py` | `forward(mode='predict')` | No additional smoothing; bilinear resize back to input size | Code alignment item by item | `matched` |

## 5. Runtime / Guard

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| strict main configuration | official `train.py` super parameters | `configs/destseg/destseg_rn18_256_mvtec_strict.py` | `iter-based 5000/1000`, batch=32, workers=16, no scheduler | config has been rewritten | `mismatch-fixed` |
| compile opt-out | History blocker | `tools/train.py` / `tools/test.py` / `tools/benchmark.py` | benchmark/train/test can be explicitly disabled compile | `test_destseg_benchmark_disables_compile` | `mismatch-fixed` |
| alias config | History `wrn50` misleading entry | `configs/destseg/destseg_wrn50_256_mvtec.py` | Only compatible aliases retained, no longer represent WRN-50 mainline | alias annotation fixed | `mismatch-fixed` |

## 6. Behavior verification conclusion

- [x] The train/test sample structure has been verified with probe, and `img_origin / img_aug / gt_mask` is visible on the train side.
- [x] `loss` path returned a finite scalar
- [x] `predict` path returned finite `pred_score / pred_anomaly_map`
- [x] compile opt-out already has targeted test
- [x] strict `teacher_pretrained=True` probe filed into `runs/alignment/destseg_probe_strict_cpu.json`
- [x] The current strict config's fresh `bottle` compressed smoke has been archived to `runs/alignment/destseg_bottle_smoke_compressed_strictteacher_v3`
- [x] The current strict config's fresh `grid/transistor` compressed smoke has been archived to `runs/alignment/destseg_grid_smoke_compressed_strictteacher_v1` and `runs/alignment/destseg_transistor_smoke_compressed_strictteacher_v1`
- [x] The current strict config's fresh `bottle exact1000` has been archived to `runs/alignment/destseg_bottle_exact1000_strictteacher_v2`
- [x] `runs/alignment/destseg_bottle_checkpoint_compare_cuda.json` Fixed branch-level comparisons for `60/12`, `1000/1000`, `1200/1000`
- [x] `runs/alignment/destseg_bottle_smoke_compressed_strictteacher_v2/destseg_train_targeted_diag.json` has been shown to be highly aligned with synthetic `gt_mask`, while the segmentation / DeST branch is still biased in the opposite direction on train synthetic data
- [x] `runs/alignment/destseg_bottle_exact1000_strictteacher_v2/destseg_targeted_diag.json` has completed the branch-level + aggregation-level comparison under the current strict `1000/1000`
- [x] `runs/alignment/destseg_bottle_exact1000_strictteacher_v2/destseg_segphase_onestep.json` It has been proven that on the `exact1000` checkpoint, the gradient direction of the segmentation phase single-step update is correct, and the real parameters will be updated after the constructor is repaired.
- [x] `runs/alignment/destseg_bottle_midcheck_strictteacher_v4` archived, fresh `1200/1000` results in `0.9944 / 0.9837`
- [x] strict full shard log has been recycled to `runs/alignment/destseg_part{0,1,2,3}_recomputed.json` and merged into `runs/alignment/destseg_strict_merged_progress.json`
- [x] Hanging full-run process at `cable / hazelnut / screw / wood` confirmed no log increment and stopped at `2026-03-29 16:09 UTC`
- [ ] old `destseg_strict_merged_progress.json` has been confirmed as stale full result; fresh current mainline is currently completed `10/15`, tail `leather / metal_nut / tile / toothbrush / zipper` is active tail rerun
