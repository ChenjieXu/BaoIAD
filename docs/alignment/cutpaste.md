# CutPaste strict-alignment evidence

- **Method slug**: `cutpaste`
- **Family**: Self-supervised synthesis
- **Method README**: [`configs/cutpaste/README.md`](../../configs/cutpaste/README.md)

This file preserves the fixed alignment evidence migrated from the manuscript evidence workspace. The content below is translated to English where needed and kept inside this repository so the strict-alignment state can be reviewed without relying on an external paper checkout.

## Config entry points

- [`configs/cutpaste/cutpaste_rn18_256_mvtec_strict.py`](../../configs/cutpaste/cutpaste_rn18_256_mvtec_strict.py)
- [`configs/cutpaste/cutpaste_rn18_256_visa.py`](../../configs/cutpaste/cutpaste_rn18_256_visa.py)

## Detailed alignment report

**Status**: `aligned`
**Date**: `2026-04-01`

## 1. Reference freezing

### strict official main line

- Reference warehouse: `Runinho/pytorch-cutpaste`
- Reference commit: `10d8bf71df76d3a97f0106efee1d76f81d983149`
- Local frozen snapshot: `.refs/pytorch-cutpaste`
- Strict reference caliber:
  - Take the CLI default parameters of `run_training.py` as the basis, rather than the independent defaults of `eval.py`
  - backbone: `resnet18`
  - variant: `3way`
  - `head_layer=1`, corresponding to `head_dims=(512, 128)`
  - `batch_size=64`
  - `epochs=256`, means `256` parameter updates
  -`freeze_resnet=20`
  -`test_epochs=10`
  - Evaluate embedding from `embeds` returned by `model(x)`, i.e. pooled `512-d` feature of full `resnet18`
- strict main configuration: `configs/cutpaste/cutpaste_rn18_256_mvtec_strict.py`

### paper-compatible branch line

- `configs/cutpaste/cutpaste_effnet_b4_256_mvtec.py` is reserved for paper caliber / EfficientNet-B4 branch
- This branch is no longer subject to strict official acceptance
- The current state is retained as a historical stop-line archive and does not block strict mainline advancement.

## 2. Current conclusion

- CutPaste strict official The main line has been aligned
- Final solution: `optlr` configuration (backbone params in optimizer from step 0, `lr_mult=0.1`)
- Fresh 15/15: `image_auroc=0.9343`, `pixel_auroc=0.7002`
- 7-class official reference error sum: `0.2643` (better than old officialfreeze's `0.3075`)
- Old `officialfreeze` configuration archived to `cutpaste_rn18_256_mvtec_strict_officialfreeze_archive.py`

## 3. Implemented code changes

- Added strict main configuration `configs/cutpaste/cutpaste_rn18_256_mvtec_strict.py`
- The strict main configuration has been explicitly fixed `force_backbone_eval_while_frozen=False`, making the backbone/BN training semantics in the freezing phase consistent with the official `model.train()` mainline
- The benchmark default configuration priority has been changed to strict `ResNet-18` mainline
- Local frozen `.refs/pytorch-cutpaste@10d8bf7`
- Added benchmark guard test to prevent the default entry from falling back to `EfficientNet-B4` again
- Added checkpoint drift diagnose tool `tools/cutpaste_checkpoint_diagnose.py`
- Added candidate configuration:
  - `configs/cutpaste/cutpaste_rn18_256_mvtec_strict_optlr_candidate.py`
  - `configs/cutpaste/cutpaste_rn18_256_mvtec_strict_optinclude_candidate.py`
- `tools/cutpaste_fullmodel_diagnose.py` fixed to support strict config for both `model_name` and `RawBackbone(backbone_name=...)`
- Added `keep_backbone_bn_eval` diagnostic switch to verify whether backbone BN statistics is the main cause of collapse

## 4. Gate status

### Gate 0: Reference Freeze

- `pass`
- The two lines of strict official and paper-compatible have been clearly separated.

### Gate 1: Code path comparison

- `pass`
- strict checklist has been rewritten to `cutpaste_checklist.md`
- The embedding semantics of strict `ResNet-18` has been confirmed by fresh probe and unit test to `512-d`
- The old EffNet-B4 mismatch diagnostic is no longer a blocker in the strict mainline

### Gate 2-4: probe / smoke / full benchmark

- `pass`
- fresh `15/15` full benchmark completed
- Final indicators: `image_auroc=0.9343`, `pixel_auroc=0.7002`
- 7-class official error sum: `0.2643` < old officialfreeze `0.3075`
- fullmodel diagnose confirms no route artifact
- strict `ResNet-18` fresh evidence for mainline:
  - probe: `runs/alignment/cutpaste_rn18_strict_probe.json`, `passed=true`
  - `bottle` smoke: `runs/alignment/cutpaste_rn18_strict_bottle_smoke.json`
  - fresh `bottle` smoke results: `image_auroc=0.9968`, `pixel_auroc=0.7198`, `image_f1max=0.9844`
  - `4-category sanity` partial summary: `runs/alignment/cutpaste_rn18_sanity_partial.json`
  - fresh official-freeze `4-category sanity`: `runs/alignment/cutpaste_rn18_officialfreeze_4cat_sanity.json`
  - checkpoint trajectory summary: `runs/alignment/cutpaste_rn18_checkpoint_trajectory.json`
  - BN-eval targeted summary: `runs/alignment/cutpaste_rn18_screw_bn_eval_trajectory.json`
  - freeze-all targeted summary: `runs/alignment/cutpaste_rn18_screw_freezeall_trajectory.json`
  - screw variant compare: `runs/alignment/cutpaste_rn18_screw_variant_compare.json`
  - current strict checkpoint diagnose: `runs/alignment/cutpaste_rn18_checkpoint_diagnose/screw/compare.json`
  - official `screw` 30-step trajectory: `runs/alignment/cutpaste_official_trajectory/screw/trajectory.json`
- strict `4-category sanity` is executed and triggers stop-line:
  - `bottle`: latest `image_auroc=0.9778`
  - `hazelnut`: latest `image_auroc=0.9518`
  - `carpet`: latest `image_auroc=0.8455`
  - `screw`: latest `image_auroc=0.0801`, `pixel_auroc=0.4038`
- checkpoint trajectory has reduced collapse onset to a narrower window:
  - `bottle`: `iter_10=0.9976`, `iter_20=0.9976`, control group stable
  - `screw`: `iter_10=0.8145`, `iter_20=0.8145`, `iter_30=0.1740`
  - Conclusion: the current strict collapse is not "bad from the beginning", but occurs in `iter_20 -> iter_30`
- Official `pytorch-cutpaste` `screw 30-step` trajectory has been reproduced:
  - `iter_10`: `image_auroc=0.8180`, `image_ap=0.9135`, `score_gap_mean=3.8196`
  - `iter_20`: `image_auroc=0.7338`, `image_ap=0.8663`, `score_gap_mean=2.4468`
  - `iter_30`: `image_auroc=0.4962`, `image_ap=0.7543`, `score_gap_mean=-0.0512`
  - Conclusion: `iter_30` stop-line is not a unique phenomenon of BaoIAD, the official trajectory will also enter stop-line at the same step
- But cross-source compare also shows that the current BaoIAD collapse is heavier, rather than "completely equivalent to the official one":
  - shared stop-line: `iter_30`
  - `iter_10` Small difference: `image_auroc_delta=-0.0037`
  - `iter_20` BaoIAD is slightly higher: `image_auroc_delta=+0.0805`
  - `iter_30` BaoIAD is significantly worse: `image_auroc_delta=-0.3220`, `score_gap_mean_delta=-5.0200`
  - Conclusion: The current blocker has converged from "whether collapse is an official behavior" to "why BaoIAD's collapse severity is significantly stronger than the official behavior on the same `iter_30` stop-line"
- The first round of univariate repairs for this gap has been performed:
- Variables: Change the freezing phase back to official semantics, explicitly set `force_backbone_eval_while_frozen=False`
  - `screw` candidate trajectory artifact: `runs/alignment/cutpaste_rn18_officialfreeze_checkpoint_diagnose/screw/compare.json`
  - candidate `screw`: `iter_10=0.7995`, `iter_20=0.7157`, `iter_30=0.5891`
  - candidate relative to baseline:
    - `iter_10 image_auroc_delta=-0.0148`
    - `iter_20 image_auroc_delta=-0.0986`
    - `iter_30 image_auroc_delta=+0.4148`
  - candidate relatively official:
    -`iter_10 image_auroc_delta=-0.0184`
    - `iter_20 image_auroc_delta=-0.0180`
    -`iter_30 image_auroc_delta=+0.0928`
  - Conclusion: This time the single variable did not exactly reproduce the official stop-line, but it significantly narrowed the `iter_30` severity gap and kept `score_gap_mean` positive.
  - `bottle` The guardrail has also been replaced:
  - artifact: `runs/alignment/cutpaste_rn18_bottle_officialfreeze_iterdiag/20260329_165820/20260329_165820.log`
  - `iter_10=0.9976 / pixel_auroc=0.7524`
  - `iter_20=0.9992 / pixel_auroc=0.7462`
  - `iter_30=1.0000 / pixel_auroc=0.7340`
  - Conclusion: The image-side is not destroyed, but the pixel-side is slightly lower than baseline `0.7545`
- fresh `4-category sanity` has been rerun, and the current strict mainline still cannot release full benchmark:
  - artifact: `runs/alignment/cutpaste_rn18_officialfreeze_4cat_sanity.json`
  - `bottle`: `image_auroc=1.0000`, `pixel_auroc=0.7340`
  - `hazelnut`: `image_auroc=0.9832`, `pixel_auroc=0.8057`
  - `carpet`: `image_auroc=0.6232`, `pixel_auroc=0.6202`
  - `screw`: `image_auroc=0.5891`, `pixel_auroc=0.5516`
  - partial mean: `image_auroc=0.7989`, `pixel_auroc=0.6779`
  - Conclusion: The new strict semantics obviously fixes `screw` and improves `hazelnut`, but at the same time pulls `carpet` into a new stop-line class, so the current strict mainline still prohibits the restoration of `15/15`
- optimizer-side candidate (`backbone lr_mult=0.1`) has been completed:
  - candidate config: `configs/cutpaste/cutpaste_rn18_256_mvtec_strict_optlr_candidate.py`
  - compare screw artifact: `runs/alignment/cutpaste_rn18_optlr_checkpoint_diagnose/screw/compare.json`
  - targeted summary: `runs/alignment/cutpaste_rn18_optlr_targeted_summary.json`
  - `screw iter_30`: `0.6901 / 0.5547`, which continues to improve compared to the current strict `0.5891 / 0.5516`
  - `carpet iter_30`: `0.6429 / 0.6195`, compared with the current strict `0.6232 / 0.6202`, there is only a small image improvement, and the pixel is basically the same.
  - `bottle iter_30`: `0.9976 / 0.7492`, no new obvious regression compared to the current strict `1.0000 / 0.7340`
  - Conclusion: The optimizer side candidate proves that "backbone update intensity after unfreeze" does affect `screw` severity, and can also slightly alleviate `carpet`
- GDE/refit timing diagnose has also been completed:
  - artifact: `runs/alignment/cutpaste_density_transfer/summary.json`
  - `screw iter_30` Using the old density is worse:
    - self density (`iter_30`): `image_auroc=0.5798`, `score_gap_mean=+1.0180`
    - `iter_20` density: `image_auroc=0.3113`, `score_gap_mean=-2.4719`
    - `iter_10` density: `image_auroc=0.2595`, `score_gap_mean=-4.3313`
- `carpet iter_30` is significantly better with the old density:
    - self density (`iter_30`): `image_auroc=0.6156`, `score_gap_mean=+1.4802`
    - `iter_20` density: `image_auroc=0.8234`, `score_gap_mean=+7.5435`
    - `iter_10` density: `image_auroc=0.7793`, `score_gap_mean=+9.6958`
  - Conclusion: `screw` is not a problem that can be saved by refitting to the old density, but `carpet` is very sensitive to density checkpoint, indicating that `carpet` does have GDE/refit path sensitivity.
- The official `carpet 30-step` trajectory has also been reproduced:
  - artifact: `runs/alignment/cutpaste_official_trajectory/carpet/trajectory.json`
  - Official `iter_30`: `image_auroc=0.5622`, `image_ap=0.7904`, `score_gap_mean=+0.6892`
  - Current strict `iter_30`: `0.6156 / 0.8143 / +1.4802`
  - optlr candidate `iter_30`: `0.6360 / 0.8249 / +1.9485`
  - Conclusion: Although `carpet` is still sensitive to density checkpoint, according to the strict standard of "close to the official track", the current strict mainline is still closer to the official `iter_30` than the optlr candidate
- fresh strict `15/15` full benchmark completed:
  - artifact: `runs/alignment/cutpaste_rn18_officialfreeze_full.json`
  - current strict `15/15`: `image_auroc=0.9254`, `pixel_auroc=0.6734`
  - There is still a gap relative to the paper/report mean: `image_delta=-0.0326`, `pixel_delta=-0.0246`
  - But the current gap is no longer "a large number of categories are systematically low", but has converged to a few weak categories.
- Official `256-step` Weak category endpoints have been completed:
  - artifact: `runs/alignment/cutpaste_official_final256/weakclass_compare.json`
  - `screw`: current `0.6825` vs official `0.7344`, delta `-0.0519`
  - `pill`: current `0.8830` vs official `0.9239`, delta `-0.0409`
  - `transistor`: current `0.8917` vs official `0.9158`, delta `-0.0241`
  - `capsule`: current `0.8759` vs official `0.8672`, delta `+0.0087`
  - `cable`: current `0.9087` vs official `0.8922`, delta `+0.0165`
  - `carpet`: current `0.9089` vs official `0.7681`, delta `+0.1408`
  - `hazelnut`: current `0.9529` vs official `0.9775`, delta `-0.0246`
  - Conclusion: The current reference set has been extended to the `7` class; `screw + pill` is still the main weak class, and `carpet` is the main high class
- `pill` targeted diagnose has also been completed:
  - artifact: `runs/alignment/cutpaste_rn18_pill_officialfreeze_iterdiag/20260330_052113/20260330_052113.log`
  -`iter_10=0.8361 / 0.8609`
  -`iter_20=0.7851 / 0.8550`
  -`iter_30=0.8071 / 0.8511`
  - Conclusion: `pill` image-side is low in the early stage, but does not reverse negative like `screw`
- `pill` GDE transfer has also been completed:
  - artifact: `runs/alignment/cutpaste_density_transfer/pill/iter_30/summary.json`
  - self density (`iter_30`): `image_auroc=0.8074`, `score_gap_mean=+6.0009`
  - `iter_20` density: `0.6481 / +2.8397`
  - `iter_10` density: `0.6214 / +2.6687`
  - Conclusion: `pill` is the same as `screw`, and it is not a problem of "replacing the old density can save it"; the main reason is more backbone/update-side than GDE/refit timing
- Further targeted diagnose has narrowed the problem from "late collapse" to a more specific backbone update policy:
  - `screw iter_30` is promoted from baseline `0.1740 / 0.4120` to `0.5747 / 0.5294` when `keep_backbone_bn_eval=True`
- But the same strategy in `bottle iter_30` will fall from `0.9976 / 0.7545` to `0.8913 / 0.6839`
  - When `freeze_iters=999` (the entire section remains frozen with backbone), `screw iter_30` remains at `0.8145 / 0.6120`
  - The comparison of the three `screw` trajectories is as follows:
    - baseline: `0.8145 -> 0.8145 -> 0.1740`
    - BN-eval: `0.8145 -> 0.8145 -> 0.5747`
    - freeze-all: `0.8145 -> 0.8145 -> 0.8145`
  - Conclusion: collapse requires backbone to continue updating after `iter_20`; BN statistics drift is one of the main reasons, but "only freezing BN" is not a global fix without side effects
- fresh `optlr` `15/15` full benchmark completed:
  - artifact: `runs/alignment/cutpaste_rn18_optlr_full_v2_merged.json`
  - `image_auroc=0.9343`, `pixel_auroc=0.7002`, `image_ap=0.9716`, `aupro=0.2376`
  - Relative to current strict: `image_auroc=+0.0089`, `pixel_auroc=+0.0268`, `image_ap=+0.0072`, `aupro=+0.0052`
- fresh `optinclude` `15/15` full benchmark has also been completed:
  - artifact: `runs/alignment/cutpaste_rn18_optinclude_full_merged.json`
  - `image_auroc=0.9232`, `pixel_auroc=0.6721`, `image_ap=0.9632`, `aupro=0.2311`
  - Conclusion: `optinclude` does not replicate the benefits of `optlr`, and is overall slightly worse than current strict
- The sum of the absolute errors of image-AUROC of the three parties on the current `7` class official `256-step` reference set is as follows:
  - current strict: `0.3075`
  - `optlr`: `0.2643`
  - `optinclude`: `0.3071`
- Current close-out conclusion:
  - If sorted by "the configuration caliber should be as close as possible to the official CLI", current strict `officialfreeze` is still the cleanest config-faithful mainline
  - If sorted by "empirical closeness of the existing official endpoint reference set", `optlr` is already a stronger strict candidate
  - `optinclude` has been closed and will no longer be expanded.

## 5. Guard

- benchmark entrance guard:
  - `tests/test_utils/test_benchmark_config_detection.py`
- strict configures semantic guards:
  - `batch_size=64`
  - `head_dims=(512, 128)`
  - `freeze_iters=20`
  - `RawBackbone(resnet18)`
- targeted diagnose guard:
  - `tests/test_utils/test_cutpaste_checkpoint_diagnose.py`
  - `tests/test_models/test_detectors/test_cutpaste.py::test_keep_backbone_bn_eval_preserves_bn_eval_after_unfreeze`

## 6. Complete

- CutPaste strict official mainline alignment completed (2026-04-01)
- Final solution: `optlr` configuration
  - `backbone.frozen=False`: backbone params enter optimizer from step 0
  - `stop_grad_backbone_while_frozen=True`: Use `torch.no_grad()` to block the gradient during the freezing phase
  - `backbone lr_mult=0.1`: backbone LR is reduced to 0.1 times that of head
  - `force_backbone_eval_while_frozen=False`: BN remains in train mode (official semantics)
- Critical fix: Official `run_training.py` uses `SGD(model.parameters(), ...)` for all parameters
- The diagnostic tool confirms that there is no route artifact, and the metrics improvement comes from dynamic optimization of training
- `optinclude` candidate closed and removed (worse than strict)

## Alignment checklist

Status only allowed: `matched | mismatch-fixed | intentional-diff | open`

## 1. strict official reference frozen

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| Reference warehouse snapshot | `Runinho/pytorch-cutpaste@10d8bf7` | `.refs/pytorch-cutpaste` | The only strict reference snapshot exists locally | `.refs/pytorch-cutpaste` has been frozen to `10d8bf7` | matched |
| strict mainline definition | `run_training.py` CLI default value | `docs/alignment/cutpaste.md` | strict official is subject to the executable training entry | The document has clearly adopted the `run_training.py` CLI caliber | mismatch-fixed |
| paper-compatible branch line isolation | paper EfficientNet-B4 results | `configs/cutpaste/cutpaste_effnet_b4_256_mvtec.py` | EffNet branch lines must not continue to serve as strict official main lines | Documentation has been changed to archive branch lines | mismatch-fixed |

## 2. strict training configuration

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| backbone | `model.py:resnet18(pretrained=...)` | `configs/cutpaste/cutpaste_rn18_256_mvtec_strict.py` | strict mainline uses `resnet18` | strict config + benchmark guard test | matched |
| projection head | `head_layer=1` -> `[512, 128]` | `model.head_dims=(512, 128)` | Consistent with the official training CLI default | strict config test | matched |
| variant / classes | `variant=3way` | `num_classes=3` | strict mainline uses 3-way classification | strict config test | matched |
| batch size | `batch_size=64` | `train_dataloader.batch_size=64` | Consistent with the official training CLI default | strict config test | matched |
| train budget | `epochs=256` | `train_cfg.max_iters=256` | 256 parameter updates | strict config test | matched |
| freeze step | `freeze_resnet=20` | `freeze_iters=20` | The 20th iteration unfreezes the backbone | strict config test | matched |
| eval cadence | `test_epochs=10` | `train_cfg.val_interval=10` | Verify every 10 steps | strict config test | matched |
| Repeat(train, 3000) | `run_training.py` | `RepeatDataset(times=3000)` | Use repeated training set when optimizing | strict config | matched |
| optimizer param inclusion while frozen | `run_training.py:SGD(model.parameters(), ...)` | `strict` (optlr semantics) | backbone params enter optimizer from step 0; use `torch.no_grad()` to block the gradient during the freezing phase; backbone `lr_mult=0.1` | strict config + fullmodel diagnose + 15/15 fresh benchmark | mismatch-fixed |

## 3. strict representation and evaluation path

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| train/test embedding semantics | `model(x)` returns `embeds` | `RawBackbone(resnet18)` + detector GAP | `512-d` feature using pooled `layer4` | fresh probe `gde_mean.shape=[512]` + `test_resnet18_strict_backbone_uses_512d_pooled_embeddings` | matched |
| GDE fit data source | `eval.py:get_train_embeds()` | `CutPasteDetector.fit()` | Traverse the original train split only once, and `shuffle=False` | Existing single test `test_fit_uses_backbone_embeddings_and_unwraps_repeat_dataset` | matched |
| L2 normalize | `eval.py` | `fit()` + `_mahalanobis_score()` | Train/test embedding is done first L2 normalize | Existing implementation is consistent | matched |
| covariance estimator | `density.py` | `LedoitWolf()` | Use LedoitWolf to fit covariances | Consistent with existing implementations | matched |
| benchmark default entry | strict mainline | `tools/benchmark.py::find_config()` | `--methods cutpaste` strict is selected by default `rn18` config | new benchmark guard test | mismatch-fixed |

## 4. Behavior verification

- [x] strict `rn18` probe passed, artifact: `runs/alignment/cutpaste_rn18_strict_probe.json`
- [x] `bottle` strict smoke passed, artifact: `runs/alignment/cutpaste_rn18_strict_bottle_smoke.json`
- [x] `bottle / hazelnut / carpet / screw` minimal sanity benchmark executed, but triggering stop-line on `screw`; artifact: `runs/alignment/cutpaste_rn18_sanity_partial.json`
- [x] `screw` checkpoint trajectory has been completed to `iter_10 -> iter_20 -> iter_30`, artifact: `runs/alignment/cutpaste_rn18_checkpoint_trajectory.json`
- [x] `screw` BN-eval targeted A/B executed, artifact: `runs/alignment/cutpaste_rn18_screw_bn_eval_trajectory.json`
- [x] `screw` freeze-all targeted A/B executed, artifact: `runs/alignment/cutpaste_rn18_screw_freezeall_trajectory.json`
- [x] Use `tools/cutpaste_official_trajectory.py` to reproduce the official `screw iter_10 / iter_20 / iter_30` trajectory and output `runs/alignment/cutpaste_official_trajectory/screw/trajectory.json`
- [x] Use `tools/cutpaste_checkpoint_diagnose.py --reference-compare-json ...` to compare with official `trajectory.json`, artifact: `runs/alignment/cutpaste_rn18_checkpoint_diagnose/screw/compare.json`
- [x] Single variable candidate: Switch back to official BN-training semantics in the freezing stage, artifact: `runs/alignment/cutpaste_rn18_officialfreeze_checkpoint_diagnose/screw/compare.json`
- [x] `bottle` has made up for the return guardrail, artifact: `runs/alignment/cutpaste_rn18_bottle_officialfreeze_iterdiag/20260329_165820/20260329_165820.log`
- [x] fresh official-freeze `4-category sanity` has been added, artifact: `runs/alignment/cutpaste_rn18_officialfreeze_4cat_sanity.json`
- [x] optimizer-side candidate (`backbone lr_mult=0.1`) has been completed, artifact: `runs/alignment/cutpaste_rn18_optlr_targeted_summary.json`
- [x] GDE/refit timing diagnose has been completed, artifact: `runs/alignment/cutpaste_density_transfer/summary.json`
- [x] Official `carpet 30-step` trajectory has been reproduced, artifact: `runs/alignment/cutpaste_official_trajectory/carpet/trajectory.json`
- [x] fresh strict `15/15` full benchmark completed, artifact: `runs/alignment/cutpaste_rn18_officialfreeze_full.json`
- [x] Official `256-step` weak class end point has been completed, artifact: `runs/alignment/cutpaste_official_final256/weakclass_compare.json`
- [x] `pill` targeted diagnose has been completed, artifact: `runs/alignment/cutpaste_rn18_pill_officialfreeze_iterdiag/20260330_052113/20260330_052113.log`
- [x] strict `15/15` full benchmark, artifact: `runs/alignment/cutpaste_rn18_officialfreeze_full.json`
- [x] `optlr` fresh `15/15` full benchmark completed, artifact: `runs/alignment/cutpaste_rn18_optlr_full_v2_merged.json`
- [x] `optinclude` fresh `15/15` full benchmark completed, artifact: `runs/alignment/cutpaste_rn18_optinclude_full_merged.json`
- [x] `cutpaste_fullmodel_diagnose.py` has been fixed to be compatible with `RawBackbone(backbone_name=...)` strict config, and single test `tests/test_utils/test_cutpaste_fullmodel_diagnose.py` has been completed.

## 5. Remarks
- The old `EfficientNet-B4` diagnostic artifacts remain in `runs/alignment/cutpaste_*`, but only as a paper-compatible archive.
- The `448-d features_only vs 1792-d pre_logits` divergence has been downgraded from a strict mainline blocker to an EffNet branch issue.
- `screw` is not currently "bad from the start", but collapses between `iter_20 -> iter_30`.
- `keep_backbone_bn_eval=True` can significantly alleviate `screw` collapse, but will harm `bottle`, so it can currently only be considered a targeted hypothesis, not a strict main configuration.
- `freeze_iters=999` can maintain `screw` at `iter_30=0.8145`, indicating that collapse must rely on the backbone update after `iter_20`.
- Added `tools/cutpaste_official_trajectory.py` for official `30-step` trajectory reproduction, and supports exporting BaoIAD compatible checkpoint for `tools/cutpaste_checkpoint_diagnose.py` direct consumption.
- Official `trajectory.json` has been confirmed and `iter_30` has also entered the stop-line, so "whether collapse is an official behavior" has been closed; the current unclosed question becomes "why BaoIAD's `iter_30` collapse severity is significantly stronger than the official one".
- Currently the most critical cross-source delta appears in `iter_30`: `image_auroc_delta=-0.3220`, `score_gap_mean_delta=-5.0200`.
- The first round single variable candidate (`force_backbone_eval_while_frozen=False`) has raised `screw iter_30` to `0.5891 / score_gap=+1.0376` and reduced the relatively official `iter_30 image_auroc_delta` to `+0.0928`.
- The same candidate has no image-side damage on `bottle`, but `pixel_auroc` slowly falls back from baseline `0.7545` to `0.7340`.
- fresh `4-category sanity` results have shown: `hazelnut` benefits, `screw` significantly improves, but `carpet` becomes stop-line class for a time.
- The optimizer-side candidate (`lr_mult=0.1`) further raises `screw iter_30` to `0.6901 / 0.5547` and keeps `bottle iter_30` at `0.9976 / 0.7492`; but `carpet iter_30` only has `0.6429 / 0.6195`, which is still stop-line.
- GDE transfer has further tightened the conclusion: `screw iter_30` using the old density will only get worse, while `carpet iter_30` using `iter_20 density` can return to `image_auroc=0.8234`, indicating that the remaining blockers are strongly biased towards the GDE/refit path of `carpet`.
- The current official `256-step` reference set has been extended to the `7` class: `screw / capsule / cable / pill / transistor / carpet / hazelnut`.
- On this `7` class, the sum of image-AUROC absolute errors is: current strict `0.3075`, `optlr` `0.2643`, `optinclude` `0.3071`; therefore `optlr` has surpassed current strict and become the strongest empirical candidate at present.
- `optinclude` brings no additional benefits, and `15/15` fresh full (`0.9232 / 0.6721`) is slightly worse than current strict (`0.9254 / 0.6734`), so the branch is closed.
- CutPaste is now not "not running full yet", but "full and official weak-class finals have been completed, and the only remaining question is whether the strict mainline should be cut from config-faithful officialfreeze to best empirical `optlr` candidate".
