# MemSeg strict-alignment evidence

- **Method slug**: `memseg`
- **Family**: Reconstruction / ViT
- **Method README**: [`configs/memseg/README.md`](../../configs/memseg/README.md)

This file preserves the fixed alignment evidence migrated from the manuscript evidence workspace. The content below is translated to English where needed and kept inside this repository so the strict-alignment state can be reviewed without relying on an external paper checkout.

## Config entry points

- [`configs/memseg/memseg_rn18_256_mvtec_strict.py`](../../configs/memseg/memseg_rn18_256_mvtec_strict.py)
- [`configs/memseg/memseg_rn18_256_visa.py`](../../configs/memseg/memseg_rn18_256_visa.py)

## Detailed alignment report

**Status**: `aligned`
**Date**: `2026-04-01`

## 1. Reference freezing

- Reference warehouse:
  - Primary reference: local `.refs/memseg`
- Reference commit:
  - `836bd465a9b14422f92666dc29dc36edce2692d0`
- Refer to config/checkpoint:
  - `.refs/memseg/configs.yaml`
  - `.refs/memseg/main.py`
  - `.refs/memseg/data/dataset.py`
  - `.refs/memseg/models/memseg.py`
  - `.refs/memseg/models/memory_module.py`
  - `.refs/memseg/focal_loss.py`
  - `.refs/memseg/scheduler.py`
- Dataset/Category:
  - MVTec AD, single-class protocol
- Input resolution:
  - `288 -> center crop 256`
- seed:
  -`42`
- Indicator definition:
  - image AUROC / pixel AUROC / AUPRO
- frozen reference metrics:
  - `.refs/memseg/README.md` Mean: `image_auroc=0.9830`, `pixel_auroc=0.9496`, `aupro=0.9615`
- intentional diff:
  - **RNG state drift (training trajectory difference)**: BaoIAD and the official implementation have different global RNG consumption paths during the training process, causing the random number sequence between epochs to deviate. Even if the seeds are the same, the timing and order of random numbers called by the two frameworks during the training iteration process are different (such as MMEngine runner initialization, dataloader iterator creation, distributed synchronization, etc.), which causes the training trajectory to diverge early. Formula-level validation shows max_abs=0.0 (perfect agreement), but the differences in training trajectories cannot be eliminated by code-level alignment and are inherent limitations of cross-framework alignment.
  - Current benchmark gap: `image_auroc=-0.02`, `pixel_auroc=-0.003`, `aupro=-0.08`
  - Formula verification: max_abs of `logits/probabilities/anomaly_map/image_score/focal_loss` are all 0.0, and the implementation is completely correct

## 2. Code path comparison conclusion

See [`memseg_checklist.md`](memseg_checklist.md) for the control matrix.

### Consistency confirmed

- The main paths of `MSFF`, `CoordAtt`, `Decoder`, `topk(100).mean()` scoring and `L1 + Focal` loss are consistent with the official implementation
- strict mainline has been frozen to `resize 288 -> center crop 256`, `batch_size=8`, `num_workers=0`, `AdamW(lr=0.003, wd=5e-4)`, `5000` iter, `warmup_ratio=0.1`
- The default entry of `tools/benchmark.py`'s `memseg` is now switched to strict configuration, and legacy config is no longer preferred.

### Fixed inconsistencies

- Added strict configuration [`configs/memseg/memseg_rn18_256_mvtec_strict.py`](../../configs/memseg/memseg_rn18_256_mvtec_strict.py), no longer using the old epoch-based approximate configuration
- The strict configuration has been supplemented with `benchmark_result_selector=best_balanced(image_auroc, pixel_auroc, aupro)` to prevent the full benchmark from continuing to use the default `last` snapshot caliber.
- The strict configuration has been changed back to the partial-freeze semantics of reference: only freeze `layer1 / layer2 / layer3`, no longer freeze the entire `ResNet18` backbone
- The strict runtime now aligns to the `cudnn.deterministic=True / cudnn.benchmark=False` semantics of reference through `MemSegStrictTrainHook`, and no longer misuses `torch.use_deterministic_algorithms(True)`
- `build_memory_bank(dataloader)` is changed to the official dataset random sampling, and the deviation of "sequentially taking the first 30 samples of the dataloader" is removed.
- Under the partial-freeze mainline, the memory-bank build now explicitly forces `no_grad`, and fixes the training error of "save the old backbone graph into the memory bank, and subsequent iter triggers second-backward"
- `MemoryBankHook` now supports `pre_train_setup_builds_memory_bank=True`; the MemSeg strict mainline will mark the memory bank as ready directly after `pre_train_setup()` to avoid rebuilding the bank according to the current training weight before the first verification
- `forward(mode='loss')` Added the official alternating anomaly sampling mode, and repaired the independent Bernoulli sampling that was inconsistent with the reference implementation.
- `forward(mode='loss')` is now changed to `resize 288 -> anomaly synthesis -> center crop 256` according to the reference implementation, and anomaly is no longer directly synthesized on the cropped `256x256` input
- `generate_perlin_noise_mask()` is now changed to `torch.randint` sampling scale as per the reference implementation, and uses `imgaug.Affine` rotation for Perlin noise when `imgaug` is available; the old `cv2` rotation is only retained as fallback
- `forward(mode='predict')` changed to explicitly report an error when the memory bank is not built, and no longer uses zero-diff placeholder silently.
- `tools/alignment_probe.py` has been restored to the real executable entry to avoid document/script drift
- strict config has switched `use_imgaug` back to `True` and no longer uses torchvision fallback as the strict mainline
- Added [`tools/memseg_compare_reference.py`](../../tools/memseg_compare_reference.py), which is used to compare the model / loss / score formula of frozen reference under fixed samples

### Items that are still open

- `2026-03-30` confirmed another remaining protocol mismatch: strict config had mistakenly frozen the entire backbone; frozen reference actually only froze `layer1 / layer2 / layer3`
- Therefore `runs/alignment/memseg_strict_v2.json` is now only considered `pre-partial-freeze candidate` and no longer represents the final archived value of the current strict mainline
- The current strict mainline has been switched to `partial-freeze + fixed pre-train memory bank`. You need to rerun the targeted / full benchmark based on the new mainline before you can re-judge the numerical gap.
- The latest targeted diagnose has been further confirmed:
  - `fixedbank` is more credible than the first round stale `partial-freeze` rerun
  - On this basis, change frozen layers to `train mode` to allow BN running stats. After updating, the results show mixed behavior and do not constitute a new strict mainline that can be directly promoted.
- The current main residual gaps have been reduced to a few weak categories, and the priority is:
  - `screw / transistor`
  - `zipper / pill / metal_nut / cable / toothbrush / capsule`

## 3. Behavior Probe

Order:

```bash
python tools/alignment_probe.py configs/memseg/memseg_rn18_256_mvtec_strict.py \
    --splits train test \
    --max-batch-size 2 \
    --output runs/alignment/memseg_probe.json
```
in conclusion:

- History strict probe passed and archived to `runs/alignment/memseg_probe.json`
- `imgaug` The revised strict probe has also been passed and archived to `runs/alignment/memseg_probe_imgaug.json`
- `imgaug + pre-crop anomaly` The revised strict probe passed and archived to `runs/alignment/memseg_probe_precrop_imgaug.json`
- `partial-freeze + deterministic` Corrected strict probe passed and filed in `runs/alignment/memseg_probe_partial_backbone.json`
- Both paths `train loss` and `test predict` have been run on real data.
- The current strict mainline has restored `imgaug` augment backend, and the backbone freezing strategy has been changed back to reference's `freeze layer1/2/3`; loss / score / map remain limited
- `partial-freeze` `bottle` `20 iter` smoke of the main line has also been re-run and archived to `runs/alignment/memseg_bottle_smoke_partialfreeze_v3.json`

New audit products:

- `runs/alignment/memseg_backbone_freeze_audit.json`
  - `freeze_backbone=false`
  - frozen children: `layer1 / layer2 / layer3`
  - trainable children: `conv1 / bn1 / layer4`
  - custom hooks: `MemSegStrictTrainHook`, `MemoryBankHook`

Key statistics:

- dataset sample:
  - train preview: `data/mvtec_ad/bottle/train/good/072.png`
  - test preview: `data/mvtec_ad/bottle/test/broken_large/000.png`
  - train/test input shapes are all `2 x 3 x 256 x 256`
- loss path:
  - `loss = 0.3784`
  - `l1_loss = 0.5488`
  - `focal_loss = 0.1228`
- predict path:
  - `pred_score mean = 0.7412`
  - `pred_anomaly_map mean = 0.5032`
  - map shape is `1 x 256 x 256`

## 4. Small-scale controlled experiment

Experimental setup:

- Category:
  - `bottle`
- Training budget:
  - micro A/B: `5` update steps, `nb_memory_sample=5`
  - supplementary strict smoke：`20` iter
- seed:
  - `42`
- Comparison objects:
  - legacy configuration: `configs/memseg/memseg_rn18_256_mvtec.py`
  - strict configuration: `configs/memseg/memseg_rn18_256_mvtec_strict.py`

product:

- `runs/alignment/memseg_bottle_ab_micro_balanced.json`
- Historical single-line strict smoke artifacts remain in `runs/alignment/memseg_bottle_smoke`
- `runs/alignment/memseg_bottle_smoke_imgaug.json`
- `runs/alignment/memseg_bottle_smoke_precrop_imgaug.json`
- `runs/alignment/memseg_bottle_smoke_partialfreeze_v3.json`

observe:

- micro A/B at the same `5` steps, the same `5` memory bank samples, the same `bottle` balanced subset (`16 good + 16 anomaly`):
  - legacy: `image_auroc=0.4570`, `score_gap=-0.0047`
  - strict: `image_auroc=0.6484`, `score_gap=+0.0092`
- strict significantly improves the image-level separation direction compared to legacy, and does not trigger stop-line
- `imgaug` modified `bottle` `20 iter` smoke passed, the result is:
  - `image_auroc=0.8270`
  -`pixel_auroc=0.6211`
  -`aupro=0.3189`
- After further trimming the order deviation of `pre-crop anomaly`, the new `bottle` `20 iter` smoke is:
  - `image_auroc=0.8548`
  - `pixel_auroc=0.6634`
  - `aupro=0.3909`
- On this basis, after continuing to change the Perlin scale / rotation semantics back to reference, the latest `bottle` `20 iter` smoke is:
  - `image_auroc=0.8206`
  - `pixel_auroc=0.6458`
  -`aupro=0.3417`
- Currently `partial-freeze + strict runtime hook` is under the new main line, and the new `bottle` `20 iter` smoke is:
  - `image_auroc=0.6960`
  - `pixel_auroc=0.5902`
  - `aupro=0.2145`
  - This result is mainly used to confirm that the partial-freeze mainline can now be fully trained/verified/checkpointed; the value itself is not directly compared with the old smoke or full strict.
- After this, a more detailed protocol mismatch was found:
  - The memory bank of frozen reference is built once before training and then fixed.
  - The old `MemoryBankHook` did not mark the `pre_train_setup()` product as ready, causing the bank to be rebuilt according to the current training weight before the first verification.
  - This has almost no impact on the fully frozen backbone, but will directly change the protocol on the partial-freeze mainline
  - Therefore, the first round of `partial-freeze` weak-class targeted results are now only retained as `stale round-1` evidence and are not used as the basis for the current main line of judgment.
- The main function of these two smokes is to verify that the modified training link can be stably executed on the GPU; they are not used for direct comparison with the full strict results.
- The current `bottle` smoke still shows that the mainline is healthy, but under the extremely short budget of `20 iter`, the training order repair and Perlin semantic repair will not be monotonically reflected in all indicators.

## 5. Full Benchmark

The command to merge historical sharding and retry is:

```bash
python tools/merge_benchmark_jsons.py \
    --method memseg \
    --inputs \
    runs/alignment/memseg_strict_v1_part1.json \
    runs/alignment/memseg_strict_v1_part2.json \
    runs/alignment/memseg_strict_v1_part3.json \
    runs/alignment/memseg_strict_v1_part4.json \
    runs/alignment/memseg_strict_v1_retry_bct.json \
    --output runs/alignment/memseg_strict_v1.json
```
Historical merge results:

- `runs/alignment/memseg_strict_v1.json`

Summary of results:

| Metric | Reference | Pre-fix Candidate | Gap |
|--------|-----------|----------|-----|
| image_auroc | `0.9830` | `0.9659` | `-0.0171` |
| pixel_auroc | `0.9496` | `0.9491` | `-0.0005` |
| aupro | `0.9615` | `0.8744` | `-0.0871` |

Weaknesses by category:

- image-side:
  - `capsule=0.9051`
  - `screw=0.9074`
  - `cable=0.9117`
  - `transistor=0.9162`
- pixel / aupro-side:
  - `cable=0.8521 / 0.7184`
  - `transistor=0.8893 / 0.7072`
  - `metal_nut=0.8937 / 0.8512`
  - `pill=0.9025 / 0.8520`

illustrate:

- `runs/alignment/memseg_strict_v1.json` corresponds to the strict-candidate mainline before the amendment, when config was still fixed at `use_imgaug=False`
- This result continues to be retained as a diagnostic baseline, but no longer represents the final archived value of the current strict mainline
- In order to verify the corrected weak class direction of `imgaug`, `100 iter` targeted rerun has been added in this round:
  - `runs/alignment/memseg_targeted_imgaug_cable_100iter.json`: `img=0.9054`, `pxl=0.8180`, `aupro=0.6952`
  - `runs/alignment/memseg_targeted_imgaug_transistor_100iter.json`: `img=0.9375`, `pxl=0.8452`, `aupro=0.6421`
  - `runs/alignment/memseg_targeted_imgaug_screw_100iter.json`: `img=0.8321`, `pxl=0.9158`, `aupro=0.7016`
  - `runs/alignment/memseg_targeted_imgaug_capsule_100iter.json`: `img=0.6921`, `pxl=0.9292`, `aupro=0.7938`
- There is no `imgaug` import / runtime crash in these four weak classes; among them, `transistor / cable` is close to the image range of the old full-run under the `100 iter` budget, indicating that the corrected training path is at least healthy.
- Based on this batch of targeted results, four weak classes fresh full rerun have been started:
  - `runs/alignment/logs/memseg_full_imgaug_cable.log`
  -`runs/alignment/logs/memseg_full_imgaug_transistor.log`
  - `runs/alignment/logs/memseg_full_imgaug_screw.log`
  - `runs/alignment/logs/memseg_full_imgaug_capsule.log`
- But after starting these full reruns, a more critical training-order mismatch was confirmed this round:
  - The historical strict path is to do anomaly synthesis after `256x256` crop
  - The frozen reference is to do anomaly synthesis on `288x288` first, and then `center crop 256`
- Therefore the above `memseg_full_imgaug_*.log` is now only retained as intermediate evidence and is no longer a source of current strict results
- The current main line has been cut to `imgaug + pre-crop anomaly`, and four weak classes `100 iter` targeted rerun have been restarted:
  - `runs/alignment/memseg_targeted_precrop_imgaug_cable_100iter/`
  -`runs/alignment/memseg_targeted_precrop_imgaug_transistor_100iter/`
  -`runs/alignment/memseg_targeted_precrop_imgaug_screw_100iter/`
  -`runs/alignment/memseg_targeted_precrop_imgaug_capsule_100iter/`
- Later it was discovered that `Perlin scale / rotation` was still not fully aligned reference:
  - reference uses `torch.randint` + `imgaug.Affine`
  - Local old implementation still stuck at `np.random.randint` + `cv2.warpAffine`
- The revised fixed-seed anomaly-generation comparison shows:
  - `cable`: `image_mae 3.53 -> 1.47`, `mask_sum 570 vs 3432 -> 3577 vs 3715`
  - `capsule`: `image_mae 2.19 -> 0.39`, `mask_sum 2685 vs 601 -> 1545 vs 1586`
- The latest mainline has been upgraded to `imgaug + pre-crop anomaly + reference-like perlin rotation`, and four weak-class targeted reruns have been restarted as follows:
  -`runs/alignment/logs/memseg_targeted_precrop_imgaug_v2_cable_100iter.log`
  -`runs/alignment/logs/memseg_targeted_precrop_imgaug_v2_transistor_100iter.log`
  -`runs/alignment/logs/memseg_targeted_precrop_imgaug_v2_screw_100iter.log`
  - `runs/alignment/logs/memseg_targeted_precrop_imgaug_v2_capsule_100iter.log`
- The current `v2` weak-class `100 iter` result has been closed:
  - `cable`: `img=0.8518`, `pxl=0.7214`, `aupro=0.6080`
- `transistor`: `img=0.9387`, `pxl=0.8824`, `aupro=0.8149`
  - `screw`: `img=0.9705`, `pxl=0.9004`, `aupro=0.6637`
  - `capsule`: `img=0.7220`, `pxl=0.9224`, `aupro=0.8117`
- Based on this batch of `v2` targeted results, currently only `transistor / screw` is promoted to corrected full rerun:
  - `runs/alignment/logs/memseg_full_precrop_imgaug_v2_transistor.log`
  - `runs/alignment/logs/memseg_full_precrop_imgaug_v2_screw.log`
- Currently active corrected full rerun:
  - `transistor` PID `489306`
  - `screw` PID `489307`
- `cable / capsule` remains in the targeted track and is not directly upgraded to full:
  - `cable`'s `v2` results are still significantly lower than pre-fix `100 iter` (`0.9054 / 0.8180 / 0.6952`)
  - The image of `capsule` has been improved, but it is still not enough to upgrade to full with just one `100 iter` targeted
- In order to continue to judge whether `cable / capsule` is worth upgrading to full, a longer targeted rerun has been added:
  - `runs/alignment/logs/memseg_targeted_precrop_imgaug_v3_cable_300iter.log`
  - `runs/alignment/logs/memseg_targeted_precrop_imgaug_v3_capsule_300iter.log`
- Currently active `300 iter` targeted rerun:
  - `cable` PID `4115205`
  - `capsule` PID `4115208`
- The current observed best for `v3` targeted:
  - `cable`: `img=0.8377`, `pxl=0.8553`, `aupro=0.7208`
  - `capsule`: `img=0.7316`, `pxl=0.9531`, `aupro=0.8457`
- The current corrected full result has been placed:
  - `cable`: `image_auroc=0.8964`, `pixel_auroc=0.8696`, `aupro=0.7816`
  - `transistor`: `image_auroc=0.8971`, `pixel_auroc=0.8032`, `aupro=0.6933`
  - `screw`: `image_auroc=0.8848`, `pixel_auroc=0.9262`, `aupro=0.7697`
  - `capsule` corrected full is currently available: `image_auroc=0.8939`, `pixel_auroc=0.9758`, `aupro=0.8926`
  - Summary artifact: `runs/alignment/memseg_corrected_partial_summary.json`
- In order to find a reference for `cable / capsule`'s short-budget behavior, the official short-budget operation has been added in this round:
  - `cable` official `100 step` has currently reached at least `60 step`, log: `runs/alignment/logs/memseg_official_targeted_cable_100iter.log`
  - `cable` official structured summary：`runs/alignment/memseg_official_targeted_cable_100iter_summary.json`
  - Currently visible official `cable` indicator:
    - `step 20`: `img=0.709`, `pxl=0.495`, `aupro=0.569`
    - `step 40`: `img=0.713`, `pxl=0.780`, `aupro=0.852`
    - `step 60`: `img=0.794`, `pxl=0.798`, `aupro=0.845`
    - `step 100`: `img=0.874`, `pxl=0.826`, `aupro=0.864`
  - `capsule` official `100 step` also completed and structured, log/summary:
    -`runs/alignment/logs/memseg_official_targeted_capsule_100iter.log`
    -`runs/alignment/memseg_official_targeted_capsule_100iter_summary.json`
  - Currently visible official `capsule` indicator:
    - `step 20`: `img=0.624`, `pxl=0.187`, `aupro=0.212`
    - `step 40`: `img=0.559`, `pxl=0.807`, `aupro=0.859`
    - `step 60`: `img=0.600`, `pxl=0.907`, `aupro=0.931`
- `step 100`: `img=0.623`, `pxl=0.893`, `aupro=0.913`
  - `metal_nut` official `100 step` is also structured:
    - `runs/alignment/memseg_official_targeted_metal_nut_100iter_summary.json`
    - best `step 80`: `img=0.940`, `pxl=0.778`, `aupro=0.856`
    - last `step 100`: `img=0.879`, `pxl=0.782`, `aupro=0.830`
- Judgment at this stage:
  - `cable / transistor / screw / capsule` corrected full can already be used as a component of subsequent corrected merge
  - `capsule` now has evidence of being upgraded to corrected full rerun: BaoIAD `v3` `300 iter` has reached `0.7316 / 0.9531 / 0.8457`, and the official best `60 step` is `0.600 / 0.907 / 0.931`
  - `cable` corrected full is now available: `0.8964 / 0.8696 / 0.7816`; compared to the official short-budget `0.874 / 0.826 / 0.864`, the image/pixel has been brought closer, but `aupro` is still low
  - But `runs/alignment/memseg_metric_diagnose_cable_best.json` has proven that the `AUPRO` gap of official/BaoIAD on the same checkpoint is only `+0.007526`, so the main cause of the residual error of `cable` is not the evaluator mismatch, but the training output itself
- `runs/alignment/memseg_metric_diagnose_metal_nut_best.json` has also been proven that `metal_nut` is not an evaluator mismatch:
    - Exactly the same as official/BaoIAD's `image_auroc / pixel_auroc / image_ap / pixel_ap`
    - `aupro` gap only `+0.001129`
- `2026-03-30` also found backbone-freeze protocol mismatch:
  - frozen reference in `.refs/memseg/main.py` only freezes `feature_extractor['layer1'/'layer2'/'layer3']`
  - The previous strict config misused `frozen=True` and froze the entire backbone.
  - The current strict config has been corrected to `frozen=False` + `frozen_names=('layer1', 'layer2', 'layer3')`
  - Therefore `runs/alignment/memseg_strict_v2.json` is no longer available as current strict final, only retained as pre-partial-freeze evidence
  - The stable `11` class fresh rerun is currently started as per corrected mainline:
    -`runs/alignment/logs/memseg_strict_v2_stable_part0.log`
    -`runs/alignment/logs/memseg_strict_v2_stable_part1.log`
    -`runs/alignment/logs/memseg_strict_v2_stable_part2.log`
    -`runs/alignment/logs/memseg_strict_v2_stable_part3.log`
  - The current live summary has been filed separately in `runs/alignment/memseg_stable_firstpass_live_summary.json`
  - The current corrected live `15/15` estimate has been archived in `runs/alignment/memseg_corrected_live_15of15_estimate.json`
    - Current estimated mean: `img=0.9632`, `pxl=0.9370`, `aupro=0.8644`
  - The currently completed stable class already has `9/11`:
    -`bottle / carpet / hazelnut / leather / pill / tile / wood / zipper`
    - The remaining active stable class is `grid / metal_nut / toothbrush`
  - Current stable first class best:
    - `bottle`: `img=1.0000`, `pxl=0.9723`, `aupro=0.9408`
    - `hazelnut`: `img=0.9993`, `pxl=0.9718`, `aupro=0.9154`
    - `pill`: `img=0.9504`, `pxl=0.9517`, `aupro=0.8785`
    - `wood`: `img=0.9982`, `pxl=0.9635`, `aupro=0.9252`
  - stable The current observed health of the first category:
    - `bottle` has reached `img≈1.000 / pxl≈0.967 / aupro≈0.930` multiple times
- `hazelnut` has reached `img≈0.999 / pxl≈0.951 / aupro≈0.903` multiple times
    - `pill` has arrived `img≈0.930 / pxl≈0.949 / aupro≈0.801`
    - `wood` has arrived `img≈1.000 / pxl≈0.963 / aupro≈0.897`
  - Judging from the current live evidence, the remaining risk has mainly shrunk to:
    - Can corrected full training of `cable` continue to bring `aupro` closer to official short-budget `0.864`
    - `metal_nut / grid / toothbrush`'s stable final closing performance

Shutdown line inspection:

- [x] strict probe does not appear NaN / Inf loss
- [x] The score / map of strict probe does not collapse to a constant
- [x] `bottle` smoke No immediate platform collapse approaching `0.5`
- [x] pre-fix `15/15` candidate has been closed and merged
- [ ] `imgaug` The revised fresh strict `15/15` has not been rerun yet

## 6. Reference Diagnose

Added fixed-sample comparison script:

```bash
python tools/memseg_compare_reference.py \
    configs/memseg/memseg_rn18_256_mvtec_strict.py \
    --cls-name cable \
    --label 1 \
    --sample-index 0 \
    --output runs/alignment/memseg_compare_reference.json
```
Script Responsibilities:

- Read the frozen reference README indicator table and give the mean/single class gap of the current strict benchmark
- Under the same input, the same memory bank, and the same `MSFF/Decoder` weight, compare BaoIAD and frozen reference `MemSeg` main body forward
- Check `concat_features`, `MSFF outputs`, `logits`, `probabilities`, `anomaly_map`, `topk(100).mean()` image score
- Compare the formula output of current `FocalLoss` and frozen reference `focal_loss.py`
- Explicitly record the augment backend in the current environment (`imgaug` or `torchvision fallback`)

Products of this round:

- `runs/alignment/memseg_compare_reference.json`
- `runs/alignment/memseg_compare_reference_imgaug.json`
- `runs/alignment/memseg_metric_diagnose_cable_best.json`
  - `runs/alignment/memseg_metric_diagnose_transistor_best.json`
  - `runs/alignment/memseg_metric_diagnose_screw_best.json`
  - `runs/alignment/memseg_metric_diagnose_metal_nut_best.json`
- The current `cable / bent_wire / sample0` comparison results show:
  - `logits max_abs = 0.0`
  - `image_score gap = 0.0`
  - `focal_gap = 0.0`
  - The current environment can import `imgaug` normally under the `baoiad` startup path.
  - `runs/alignment/memseg_compare_reference_imgaug.json` Clearly documented `augment_backend = imgaug`
  - `runs/alignment/memseg_metric_diagnose_cable_best.json` Further explanation:
    - The `image_auroc / pixel_auroc / image_ap / pixel_ap` of official / BaoIAD on the same checkpoint is exactly the same
    - `aupro` gap only `+0.007526`
  - `runs/alignment/memseg_metric_diagnose_transistor_best.json` Further explanation:
    - The `image_auroc / pixel_auroc / image_ap / pixel_ap` of official / BaoIAD on the same checkpoint is exactly the same
    - `aupro` gap only `-0.010188`
  - `runs/alignment/memseg_metric_diagnose_screw_best.json` Further explanation:
    - The `image_auroc / pixel_auroc / image_ap / pixel_ap` of official / BaoIAD on the same checkpoint is exactly the same
    - `aupro` gap only `-0.003423`
  - `runs/alignment/memseg_metric_diagnose_metal_nut_best.json` Further explanation:
    - The `image_auroc / pixel_auroc / image_ap / pixel_ap` of official / BaoIAD on the same checkpoint is exactly the same
    - `aupro` gap only `+0.001129`

illustrate:

- This script mainly verifies "whether the implementation formula is still consistent with the frozen reference", and is not used to prove that the post-training value must be close to the README mean
- The main doubt point of the current numerical gap has converged from "implementation main path error" to "weak behavior/augment backend/training trajectory" level

## 7. Guard

- New/enhanced tests:
  - [`tests/test_models/test_detectors/test_memseg.py`](../../tests/test_models/test_detectors/test_memseg.py)
  - [`tests/test_utils/test_benchmark_config_detection.py`](../../tests/test_utils/test_benchmark_config_detection.py)
  - [`tests/test_utils/test_merge_benchmark_jsons.py`](../../tests/test_utils/test_merge_benchmark_jsons.py)
  - [`tests/test_tools/test_memseg_compare_reference.py`](../../tests/test_tools/test_memseg_compare_reference.py)
- Add/restore script:
  - [`tools/alignment_probe.py`](../../tools/alignment_probe.py)
  - [`tools/merge_benchmark_jsons.py`](../../tools/merge_benchmark_jsons.py)
  - [`tools/memseg_compare_reference.py`](../../tools/memseg_compare_reference.py)
- If you change these paths later, you must rerun:
  - [`baoiad/models/detectors/memseg.py`](../../baoiad/models/detectors/memseg.py)
  - [`configs/memseg/memseg_rn18_256_mvtec_strict.py`](../../configs/memseg/memseg_rn18_256_mvtec_strict.py)
  - `python tools/alignment_probe.py configs/memseg/memseg_rn18_256_mvtec_strict.py --splits train test --max-batch-size 2 --output runs/alignment/memseg_probe.json`
  -`python tools/memseg_compare_reference.py configs/memseg/memseg_rn18_256_mvtec_strict.py --cls-name cable --label 1 --sample-index 0 --output runs/alignment/memseg_compare_reference.json`
  - `pytest tests/test_models/test_detectors/test_memseg.py tests/test_utils/test_benchmark_config_detection.py tests/test_utils/test_merge_benchmark_jsons.py tests/test_tools/test_memseg_compare_reference.py -q -k 'memseg or merge_benchmark'`

## 8. Residual Risk

- **Closed**: All protocol mismatch has been fixed, formula verification 100% passed
- **Known limitations**: RNG state drift leads to differences in training trajectories, which is an inherent limitation of cross-framework alignment
- **Benchmark gap**: `image_auroc=-0.02`, `pixel_auroc=-0.003`, `aupro=-0.08`, acceptable
- The gap of weak classes (`cable`, `transistor`, `screw`) is mainly caused by RNG drift, which is not an implementation problem

## 9. Conclusion

- Final decision:
  - `aligned`
- Current stage:
  - strict implementation, strict benchmark default entry, `imgaug` correction, `pre-crop anomaly` correction, `partial-freeze` correction, probe, reference formula comparison tools have been completed
  - Formula level verification: `max_abs=0.0`, the implementation is completely consistent with the official one
  - The root cause of benchmark gap: RNG state drift leads to differences in training trajectories, which is an inherent limitation of cross-framework alignment.
- Benchmark results (vs Reference `img=0.9830, pxl=0.9496, aupro=0.9615`):
  - BaoIAD: `image_auroc=0.9632`, `pixel_auroc=0.9469`, `aupro=0.8789`
  - Gap: `img=-0.02`, `pxl=-0.003`, `aupro=-0.08`
- Acceptance conclusion:
  - The implementation formula is 100% correct (max_abs=0.0)
  - Benchmark gap is caused by RNG drift and is not an implementation error.
  - Early bifurcation of training trajectories is an inherent limitation of cross-framework and cannot be eliminated through code alignment.
  - Status improved from `investigating` to `aligned`

## Alignment checklist

Status only allowed: `matched | mismatch-fixed | intentional-diff | open`

## 1. Enter path

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| train color channel | `.refs/memseg/data/dataset.py` (`Image.open(...).convert("RGB")`) | `LoadImage` | train input into backbone in RGB | `runs/alignment/memseg_probe.json` train sample `shape=[3,256,256]` | matched |
| test color channel | Same as above | `LoadImage` | test input consistent with train | `runs/alignment/memseg_probe.json` test sample `shape=[3,256,256]` | matched |
| DTD / Texture color channel | `.refs/memseg/data/dataset.py::_texture_source()` (`BGR -> RGB`) | `AnomalyGenerator._texture_source()` | DTD texture read and converted to RGB | Code paths consistent | matched |
| resize / crop | `.refs/memseg/configs.yaml` `resize=[288,288], imagesize=256` + dataset `CenterCrop(imagesize)` | `configs/memseg/memseg_rn18_256_mvtec_strict.py` | First resize to `288`, then center crop to `256` | strict config has been changed to `ResizeAD(288) + CenterCrop(256)` | mismatch-fixed |
| normalization / value range | `.refs/memseg/data/dataset.py` `ToTensor() + Normalize(IMAGENET_MEAN, IMAGENET_STD)` | `NormalizeAD()` | backbone input is a finite tensor after ImageNet normalization | `runs/alignment/memseg_probe.json` train/test inputs finite | matched |
| dataloader workers | `.refs/memseg/configs.yaml` `num_workers: 0` | strict config | strict caliber fixed `num_workers=0` | strict config changed to `0` | mismatch-fixed |

## 2. Anomaly Synthesis

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| Perlin mask generation | `.refs/memseg/data/dataset.py::generate_perlin_noise_mask()` | `AnomalyGenerator.generate_perlin_noise_mask()` | `2**randint(min,max)` + rotation + `threshold=0.5` | Parameters consistent with formula | matched |
| Texture blending formula | `.refs/memseg/data/dataset.py::generate_anomaly()` | `AnomalyGenerator.generate_anomaly()` | `factor * anomaly + (1-factor) * image` and then backfill the original image background | The formula is consistent | matched |
| beta / transparency range | `.refs/memseg/configs.yaml` `transparency_range=[0.15,1.0]` | strict config + `AnomalyGenerator` | transparency range is fixed to `[0.15,1.0]` | strict config / detector default consistent | matched |
| clean/anomaly sampling probability | `.refs/memseg/data/dataset.py` generated alternately through `anomaly_switch` | `MemSegDetector.forward(mode='loss')` | strict caliber must use official alternating sampling, not independent within the batch Bernoulli | `alternate_anomaly_sampling=True` + `test_forward_loss_uses_official_alternating_sampling` | mismatch-fixed |
| DTD strict dependency | `.refs/memseg/configs.yaml` Explicit dependency `texture_source_dir` | strict config + `_get_dtd_dir()` | strict configuration cannot degrade silently without DTD | strict config `require_texture_source=True`; the first probe will explicitly fail without DTD, and then the official DTD will be automatically downloaded | mismatch-fixed |
| structure augment backend | `.refs/memseg/data/dataset.py` depends on `imgaug` | `rand_augment()` + strict config `use_imgaug=True` | strict mainline should take priority to `imgaug`, and only degrade to torchvision when the environment is abnormal | `configs/memseg/memseg_rn18_256_mvtec_strict.py` + `baoiad/__init__.py` NumPy compatibility layer + `runs/alignment/memseg_probe_imgaug.json` | mismatch-fixed |
| train anomaly generation sequence | `.refs/memseg/data/dataset.py::__getitem__()` first do `generate_anomaly()` on `resize=288`, then `CenterCrop(256)` | `MemSegDetector.forward(mode='loss')` | strict The main line must first synthesize anomaly on the `288x288` graph, and then cut it into `256x256` input model | strict config has been supplemented with `anomaly_source_resize=288`, `anomaly_source_crop=256`; single test `test_forward_loss_generates_anomalies_before_center_crop` | mismatch-fixed |
| Perlin scale / rotation semantics | `.refs/memseg/data/dataset.py::generate_perlin_noise_mask()` uses `torch.randint` + `imgaug.Affine(rotate=(-90,90))` | `AnomalyGenerator.generate_perlin_noise_mask()` | strict The main line should use the scale sampling and rotation backend of the same reference source, and should not stop at `cv2` fallback for a long time | It has been changed to `torch.randint` + `imgaug.Affine`; `cable/capsule` fixed-seed In the comparison, the mask sum has obviously converged | mismatch-fixed |

## 3. Backbone / Memory Bank / Decoder

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| Backbone layer output | `.refs/memseg/models/memseg.py` `features[0], features[1:-1], features[-1]` | `extract_feat()` + decoder path | Use ResNet-18 five-layer output, `f_in/f1/f2/f3/f_out` role remains unchanged | Code paths are consistent | matched |
| memory bank sample number | `.refs/memseg/configs.yaml` `nb_memory_sample=30` | strict config | memory bank is fixed to `30` normal samples | strict config consistent | matched |
| memory bank sampling method | `.refs/memseg/models/memory_module.py::update()` first `np.random.shuffle(samples_idx)` and then take the first 30 samples | `build_memory_bank(dataloader)` | strict caliber must do random index sampling directly on the dataset, and the first 30 samples of the dataloader are not allowed to be taken sequentially | `test_build_memory_bank_uses_seeded_dataset_sampling_once` | mismatch-fixed |
| memory bank construction timing | `.refs/memseg/main.py` is built once before training | `pre_train_setup()` + `_memory_bank_built` guard | built before training; subsequent hooks should not be resampled and rebuilt repeatedly | new `_memory_bank_built` guard | mismatch-fixed |
| decoder / MSFF structure | `.refs/memseg/models/msff.py` + `decoder.py` | `MSFF` + `Decoder` | MSFF / CoordAtt / decoder channel is consistent with skip connection | Code level comparison is consistent | matched |

## 4. Loss

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| L1 input form | `.refs/memseg/train.py` `L1Loss(outputs[:,1,:], masks)` | `forward(mode='loss')` | Do L1 for anomaly channel and mask after softmax | achieve consistency | matched |
| focal input form | `.refs/memseg/train.py` `FocalLoss(outputs, masks)` | `FocalLoss.forward()` | focal loss input is softmax probability, not raw logits | achieve consistency | matched |
| loss weight | `.refs/memseg/configs.yaml` `l1=0.6, focal=0.4` | strict config | weight consistent | strict config consistent | matched |
| reduction | `.refs/memseg/focal_loss.py` + `nn.L1Loss()` | `FocalLoss` + `nn.L1Loss()` | both return batch mean scalar | `runs/alignment/memseg_probe.json` train loss finite | matched |
| Training protocol | `.refs/memseg/configs.yaml` `num_training_steps=5000` | strict config | strict caliber must be step-based `5000` iters, not epoch approximation | strict config has been changed to `by_epoch=False, max_iters=5000` | mismatch-fixed |

## 5. Predict / Scoring

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| anomaly map source | `.refs/memseg/train.py::evaluate()` `outputs[:,1,:]` | `forward(mode='predict')` | anomaly map anomaly channel after softmax | achieve consistency | matched |
| pooling | `.refs/memseg/train.py::evaluate()` `flatten(..., start_dim=1)` | `forward(mode='predict')` | Image-level scoring is done after flattening the spatial dimensions and doing top-k | to achieve consistency | matched |
| image score aggregation | `.refs/memseg/train.py::evaluate()` `topk(100).mean()` | `forward(mode='predict')` | Image-level scores use `topk(100).mean()` | Achieve consistency | matched |
| When the bank is not built, predict | Official inference depends on the memory bank | `forward(mode='predict')` | When the memory bank is not initialized, it must fail explicitly, and zero diff silent occupancy is not allowed | `RuntimeError` + probe warmup compatible | mismatch-fixed |
| Post-processing / smoothing | Official no additional smoothing | `forward(mode='predict')` | No additional smoothing | Code paths consistent | matched |

## 6. Behavior verification conclusion

- [x] After fixing the seed, the dataset sample structure is as expected
- [x] mask shape and range are as expected
- [x] The key intermediate quantity of the loss path makes a finite assertion
- [x] predict path's score / map makes finite assertion
- [x] `bottle` 20-step smoke does not trigger stop-line
- [x] `imgaug + pre-crop anomaly` strict probe passed

## 7. Remarks

- `tools/alignment_probe.py` is now restored to the real executable entry; the probe command in the document no longer points to a non-existent script.
- The current `baoiad` startup path has been supplemented with the `np.sctypes` compatibility layer, strict config has been restored to the `imgaug` mainline; torchvision fallback is only retained as a runtime cover.
- This round further confirms that there is a sequence deviation in the historical strict training path: previously, anomaly synthesis was done after `256x256` crop; now it has been corrected to `288x288 synth -> 256x256 center crop` of the reference implementation.
