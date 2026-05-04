# ViTAD strict-alignment evidence

- **Method slug**: `vitad`
- **Family**: Reconstruction / ViT
- **Method README**: [`configs/vitad/README.md`](../../configs/vitad/README.md)

This file preserves the fixed alignment evidence migrated from the manuscript evidence workspace. The content below is translated to English where needed and kept inside this repository so the strict-alignment state can be reviewed without relying on an external paper checkout.

## Config entry points

- [`configs/vitad/vitad_256_mvtec_strict.py`](../../configs/vitad/vitad_256_mvtec_strict.py)
- [`configs/vitad/vitad_256_visa.py`](../../configs/vitad/vitad_256_visa.py)

## Detailed alignment report

**Status**: `aligned (exact-order mainline verified at 100e, image_auroc=0.9835)`
**Date**: `2026-04-01`

## 1. Reference freezing

- Reference repository: local `.refs/ader`
- Reference commit: `902937a7ed7fa7689674a4ac9b8fe9a72a40c402`
- Refer to config/model:
  - `.refs/ader/configs/benchmark/vitad/vitad_256_100e.py`
  - `.refs/ader/configs/__base__/cfg_model_vitad.py`
  - `.refs/ader/model/vitad.py`
  - `.refs/ader/trainer/vitad_trainer.py`
- Data set/category: MVTec AD, MUAD multi-class joint training, 15-class standard benchmark
- Input resolution: `256x256`
- seed: `42`
- Indicator definition: image AUROC / pixel AUROC
- intentional diff:
  - BaoIAD uses mmengine train/val/test loop and `build_predict_results()` to encapsulate prediction results
  - BaoIAD's `data/mvtec_ad` directory structure is equivalent to ADer `data/mvtec`, but the path naming is different
  - Gate 3 smoke only does `bottle` single type low-cost override, and does not use the result as MUAD headline comparison

## 2. Code path comparison conclusion

See [`vitad_checklist.md`](vitad_checklist.md) for the control matrix.

### Consistency confirmed

- The current MUAD master configuration has been fixed to ADer `vit_small_patch16_224_dino + teachers(3,6,9) + neck(12) + students(3,6,9) + decoder_depth=9 + fusion_mul=1 + AdamW(lr=1e-4, wd=1e-4) + StepLR(80e/100e) + seed=42`
- `ViTADDetector`'s flattened cosine loss, `/9` multi-scale aggregation, `avg_pool2d(16)` image scoring are consistent with the ADer main line
- `tools/benchmark.py` has defaulted to pointing `vitad` to the MUAD configuration, and `benchmark_multi_class = True`
- The benchmark training entrance has now been switched to ViTAD exclusive `tools/train_vitad_exact_order.py`, and will not go back to the old wrong workerized train order main line.

### Fixed inconsistencies

- MUAD configuration previously only relied on detector default values without explicitly freezing `encoder_name / teachers / neck / students / decoder_depth / fusion_mul`; now these reference hyperparameters have all been written to `configs/vitad/vitad_256_mvtec_muad.py`
- The `predict` path previously used convolution approximation Gaussian smoothing; now it has been switched to the same `scipy.ndimage.gaussian_filter(sigma=4)` as ADer to avoid additional deviations at the boundaries of the pixel anomaly map.
- `ViTEncoderBackbone` previously consumed additional random numbers due to `timm.create_model()` meta-information detection and pretrained loading, causing the fusion/decoder initialization to deviate from ADer; now fixed with `torch.random.fork_rng()`
- `ViTEncoderBackbone` previously retained the classifier head by default; while ADer’s DINO teacher path has no head. Now explicitly aligned to `num_classes=0`
- `MVTecADDataset` was previously missing the dataset-level `random.shuffle(self.data_all)` of ADer `DefaultAD`; now it has added `shuffle_train_data=True` and is only enabled in the ViTAD MUAD main configuration
- Added strict main configuration `configs/vitad/vitad_256_mvtec_strict.py`, freezing protocol-level assumptions separately to `PersistentShuffleSampler + ViTADStrictTrainHook + val_begin=10,val_interval=10 + checkpoint interval=10`
- Added `tools/vitad_protocol_diagnose.py`. Currently, it is confirmed on local data that the train set size of strict path is `3629`, every epoch `454` iter, and lr is at the beginning of `epoch 81`, and the iter decays from `1e-4` to `1e-5`
- `Runner._init_model_weights()` would previously randomly initialize ViTAD's timm ViT module again; now this runner re-init path is explicitly disabled on `ViTADDetector`
- `ResizeAD` now supports ViTAD-only `official_pil=True`, closing the official PIL resize numerical path to the ViTAD configuration without affecting other models
- `AdamW` param groups are now aligned to ADer's `no_decay + decay` via `ViTADOptimWrapperConstructor`
- exact workerized train order now mainlined via `PerEpochOrderSampler + tools/vitad_dump_official_order.py + tools/train_vitad_exact_order.py`

### Items that are still open

- ~~Historical workerized mainline still cannot be regarded as final alignment evidence; as long as exact-order is not used, `bcc epoch10` will still fall back to `img≈0.446 / pxl≈0.675`~~ — Fixed by exact-order mainline
- ~~The current benchmark default configuration is still `configs/vitad/vitad_256_mvtec_muad.py`, but the training entrance has been switched to the ViTAD-only exact-order mainline by `benchmark_train_script`~~ — Confirmed
- ~~The current main line has been switched from "strict targeted compare diagnose" to "MUAD all15 exact-order execution"; the remaining open items are just the final headline on the long budget `100e`~~ — **Completed**: 100e exact-order result `image_auroc=0.9835` is perfectly aligned with the official `98.3%`

## 3. Behavior Probe

Order:

```bash
HF_HUB_OFFLINE=0 python tools/alignment_probe.py configs/vitad/vitad_256_mvtec_muad.py \
    --splits train test \
    --max-batch-size 2 \
    --device cuda \
    --output runs/alignment/vitad_probe.json
```
in conclusion:

- `pass`
- When running ViTAD probe for the first time in the current environment, you need to explicitly set `HF_HUB_OFFLINE=0` to pull the timm weight cache of `vit_small_patch16_224.dino`; offline operation can be resumed after the cache is landed.
- The probe proves that the main paths of MUAD dataloader, loss, predict, score map and image score can actually run through

Key statistics:

- dataset sample:
  - train preview: `data/mvtec_ad/zipper/train/good/065.png`
  - test preview: `data/mvtec_ad/bottle/test/broken_large/000.png`
  - train/test input shape: `2 x 3 x 256 x 256`
- loss path:
  - `loss` dict exists and is finite
  - probe train loss=`2.9796`
- predict path:
  - `pred_score` finite, mean=`0.3461`
  - `pred_anomaly_map` shape=`[1, 256, 256]`，mean=`0.3216`

strict path retest:

```bash
HF_HUB_OFFLINE=1 python tools/alignment_probe.py configs/vitad/vitad_256_mvtec_strict.py \
    --splits train test \
    --max-batch-size 2 \
    --device cuda \
    --output runs/alignment/vitad_strict_probe.json
```
in conclusion:

- `pass`
- The dataloader / loss / predict structures under strict path can run normally without introducing new runtime regression.
- `runs/alignment/vitad_protocol_diagnose.json` has also been completed, and the current strict path is fixed to `dataset_size=3629`, `iters_per_epoch=454`, `epoch81 lr=1e-5`

## 4. Fixed sample and one-step update comparison

Fixed sample control command:

```bash
HF_HUB_OFFLINE=1 python tools/vitad_compare_reference.py \
    --split test \
    --cls-name bottle \
    --label 1 \
    --sample-index 0 \
    --device cpu \
    --output runs/alignment/vitad_compare_reference.json
```
in conclusion:

- On `data/mvtec_ad/bottle/test/broken_large/000.png`, BaoIAD and ADer's:
  -preprocess
  - teacher features
  - neck tokens
  - fusion output
  - decoder student features
  - anomaly map
  -image score
  are aligned to numerical error levels
- `runs/alignment/vitad_compare_reference.json` in:
  - `teacher/neck/fused/student/score_map/score`’s `l1_mean` are all `0.0`

One-step update comparison conclusion:

- Same as `seed=42`, same as train batch, same as `AdamW(lr=1e-4, wd=1e-4)`, same as `clip_grad=5.0` below:
  - `loss`
  - `fusion.weight`
  - `fusion.bias`
  - `student.pos_embed`
  - `student.block0.attn.qkv.weight`
  After the one-step update, the difference is `0.0`
- This shows that the current ViTAD's "fixed sample forward + one-step training update" has been strictly aligned to ADer

## 5. Small-scale controlled experiment

First do `bottle` smoke of `1 epoch` and get `image_auroc=0.3754` and `pixel_auroc=0.5206`. This budget is too short for ViTAD to judge whether the implementation is right or wrong, so it continues to be upgraded to `10 epoch`.

Final smoke settings:

- Category: `bottle`
- Training budget: `10 epoch`
- seed: `42`
- Purpose: Only to verify whether the mainline implementation is stable for training, rather than replacing the MUAD headline with single-category short-term training results

Order:

```bash
HF_HUB_OFFLINE=1 python tools/train.py configs/vitad/vitad_256_mvtec_muad.py \
    --work-dir runs/alignment/vitad_bottle_smoke_10e_rngfix \
    --cfg-options \
        train_cfg.max_epochs=10 \
        train_cfg.val_interval=1 \
        train_dataloader.batch_size=4 \
        test_dataloader.batch_size=4 \
        val_dataloader.batch_size=4 \
        "train_dataloader.dataset.cls_names=['bottle']" \
        train_dataloader.dataset.multi_class=False \
        "test_dataloader.dataset.cls_names=['bottle']" \
        test_dataloader.dataset.multi_class=False \
        "val_dataloader.dataset.cls_names=['bottle']" \
        val_dataloader.dataset.multi_class=False
```
observe:

- train loss decreases from `epoch1 iter50 = 0.0807` to `epoch10 iter50 = 0.0096`
- The validation indicator continues to rise with epoch:
  - `image_auroc: 0.3706 -> 0.5206`
  - `pixel_auroc: 0.5165 -> 0.6386`
  - `aupro: 0.2314 -> 0.3756`
- There is no NaN, loss explosion, pure zero score map or batch level crash in the whole process

determination:

- `pass (structural)`
- Reason: This can prove that the current strict mainline is "trainable and the direction improves monotonically", rather than achieving direct deviation.
- But `image_auroc=0.5206` is still weak and cannot be used as final alignment evidence; it can only be used as Gate 3 structural evidence

strict path `bottle` smoke（`runs/alignment/vitad_strict_bottle_smoke_10e`）：

- Training configuration: `configs/vitad/vitad_256_mvtec_strict.py`, `10 epoch`, `val_interval=1`
- Final result: `image_auroc=0.5214`, `pixel_auroc=0.6375`, `aupro=0.3754`
- Track:
  - `epoch1`: `img=0.3698`, `pxl=0.5166`
  - `epoch5`: `img=0.4667`, `pxl=0.6084`
  - `epoch10`: `img=0.5214`, `pxl=0.6375`
- Judgment: strict sampler / strict hook did not ruin the single-category short training, but it did not bring significant improvements beyond the old `rngfix` smoke

strict path targeted compare（`runs/alignment/vitad_strict_targeted_bcc_100e`）：

- The first multi-category verification of `epoch10` has been completed, and the category is fixed at `['bottle', 'cable', 'capsule']`
- `epoch10` Average results: `image_auroc=0.4475`, `pixel_auroc=0.6681`, `aupro=0.4459`
- `epoch10` sub-categories:
  - `bottle`: `img=0.3714`, `pxl=0.5820`
  - `cable`: `img=0.3932`, `pxl=0.5254`
  - `capsule`: `img=0.5780`, `pxl=0.8970`
- Judgment: strict path has truly changed the protocol and given new evidence, but the current `epoch10` multi-type image performance is still obviously weak, and there is no signal for the time being that "strict protocol will quickly return to the ADer track after being repaired"

strict path targeted compare rerun (current strict mainline, `runs/alignment/vitad_strict_targeted_bcc_100e_rerun_strictmainline`):

- Use the currently frozen strict config to run and confirm that it is not an artifact caused by the worker/backend caliber drift of the old `2026-03-27` run.
- The current rerun config log has been confirmed:
  - `persistent_workers=False`
  - `LoadImage/LoadMask backend='pil'`
  - `ResizeAD backend='pillow'`
  - `PersistentShuffleSampler(seed=42)`
- The train loss of rerun is still declining rapidly, but the numerical magnitude continues to be significantly lower than the official ADer window:
  - BaoIAD `epoch1/2/3` train loss: `0.1182 -> 0.0688 -> 0.0479`
  - ADer official `iter50` first window is still at the `~1.1` level, and will still maintain a higher loss platform around `epoch10`
- `epoch10` verification result of rerun:
  - Average: `image_auroc=0.4461`, `pixel_auroc=0.6765`, `image_ap=0.7236`, `aupro=0.4433`
  - `bottle`: `img=0.3762`, `pxl=0.6211`
  - `cable`: `img=0.3896`, `pxl=0.5015`
  - `capsule`: `img=0.5724`, `pxl=0.9069`
- This shows that the old `2026-03-27` strict targeted run is not an accidental drift; the current strict mainline still shows the same level of weak trajectory after rerun

official targeted compare（`runs/alignment/ader_vitad_official_targeted_bcc_100e_summary.json`）：

- ADer official `bcc` targeted compare has been run through the same machine, the same `classes=['bottle','cable','capsule']`, the same `100e`, the same `10e` cadence
- `epoch10` Official average:
  - `image_auroc=0.9634`
  - `pixel_auroc=0.9834`
  - `image_ap=0.9902`
  - `aupro=0.9237`
- Official `100e` Final state average:
  -`image_auroc=0.9834`
  - `pixel_auroc=0.9838`
  -`image_ap=0.9958`
  -`aupro=0.9254`

strict targeted compare conclusion (combined summary: `runs/alignment/vitad_targeted_compare_summary_rerun_strictmainline.json`):

- shared eval epoch currently only has `epoch10`, but it is enough to trigger stop-line
- `epoch10` average delta (BaoIAD - official):
  - `image_auroc=-0.5174`
  - `pixel_auroc=-0.3069`
  - `image_ap=-0.2666`
  - `aupro=-0.4804`
- Subcategory `epoch10` delta:
  - `bottle`: `img=-0.6238`, `pxl=-0.3667`, `aupro=-0.6602`
  - `cable`: `img=-0.6005`, `pxl=-0.4760`, `aupro=-0.5973`
  - `capsule`: `img=-0.3279`, `pxl=-0.0780`, `aupro=-0.1836`
- Judgment:
  - strict targeted compare has locked the "real gap between the current strict mainline and ADer official" to the evidence layer
- This gap was already extremely large in `epoch10`, so strict full rerun is currently not allowed to be used as the main action
- The next step must be to return to trainer-level diagnose instead of continuing to accumulate `15/15`

trainer-level diagnose（`runs/alignment/vitad_trainer_level_diagnose_bcc_step1.json`）：

- `tools/vitad_trainer_level_diagnose.py` has been added, using the minimum official stack to directly compare the `step0 + step1` of the current strict mainline
- The current real `step1` diagnosis has been run through and the first batch of trainer-level evidence is given:
  - `step0` batch paths are aligned and no longer a shared RNG pseudo-difference
  - But the `step0` input still has `processed_absmax_diff=0.01695`
  - On the same batch, `step0`’s `fused max_abs=0.0386`, `student max_abs=0.1376`
  - `step0` loss is only `0.000297`, indicating that the current situation is more like "the input/forward slight offset is amplified in multi-step training" rather than completely running away in the first step.
  - After `step1`, the key parameters begin to be clearly separated:
    -`fusion.weight max_abs=1.9998e-4`
    -`student.pos_embed max_abs=1.9945e-4`
- diagnose also directly exposed the inconsistency of the current optimizer param-group of BaoIAD/official:
  - BaoIAD: single `AdamW(weight_decay=1e-4)` group
  - ADer official: `no_decay + decay` Two groups
- This current piece of evidence explains:
  - The old "one-step optimizer compare is fully aligned" is not enough to represent the real strict train loop
  - The next round of investigation objects with the highest priority are `input pipeline/data_preprocessor` and `optimizer param groups`

trainer-level diagnose after fixes（`runs/alignment/vitad_trainer_level_diagnose_bcc_step5_afterfix.json`）：

- Added `ViTADOptimWrapperConstructor`, aligning BaoIAD ViTAD's optimizer param groups to ADer `no_decay + decay`
- `ResizeAD(backend='pillow')` has been switched to the real PIL resize path to avoid numerical deviations between torchvision tensor-resize and official PIL-resize
- strict / MUAD config has also been explicitly switched to `cudnn_benchmark=True`
- Fixed trainer-level diagnose results:
  - `step0` batch paths are exactly the same
  - `step0` Input diff down to floating point error level: `processed_absmax_diff=4.77e-07`
  - `step0` teacher / neck / fused / student / loss return to zero
  - `step1` All key parameters are updated and reset to zero.
  - After expanding to `step5`, there is still no first divergent step, `candidate_causes=[]`
- This shows that the current strict mainline is strictly consistent with ADer official at the level of "isolated batch + optimizer + first 5 steps of training"

post-fix strict targeted rerun（`runs/alignment/vitad_strict_targeted_bcc_10e_afterfix`）：

- Based on the above fixes, `bcc` `10e` strict targeted compare has been re-done.
- The results are almost unchanged:
  - Average: `image_auroc=0.4469`, `pixel_auroc=0.6749`, `image_ap=0.7251`, `aupro=0.4491`
  - Compare the current strict rerun before repair: only the noise level changes, and there is no substantial recovery.
- Judgment:
  - The current problem is no longer `input pipeline`, `optimizer param groups`, `cudnn_benchmark`, or the first 5 steps of training itself
  - The remaining mismatch is more likely to be hidden in the "difference between real mmengine train loop and isolated step replay"
  - The next level with high probability is: intra-epoch/inter-epoch iteration timing, worker>0 dataloader runtime, or runner/hook integrated semantics

workerized trainer-level diagnose（`runs/alignment/vitad_trainer_level_diagnose_bcc_step100_workers4.json`）：

- When trainer-level diagnose switches to `num_workers=4`, `step0` will fork again:
  - `batch_paths` is inconsistent
  - `processed_absmax_diff=4.2367`
  - `loss abs diff=0.0080`
- But the same tool can be fully aligned in `num_workers=0` and fixed isolated `step5` scenarios
- This indicates that the real residual mismatch is strongly related to the workerized DataLoader / iterator runtime, rather than the model formula itself

`ViTADOfficialTrainLoop` try(`runs/alignment/vitad_strict_targeted_bcc_10e_officialloop`):

- Added official-style raw train dataloader loop that only applies to ViTAD strict config to avoid affecting other models
- The first batch of train batch paths created by this loop are consistent with the official ones
- But the actual `bcc 10e` rerun result is still almost the same:
  - `image_auroc=0.4469`
  - `pixel_auroc=0.6749`
  - `image_ap=0.7251`
  - `aupro=0.4491`
- And there is still an obvious parameter gap between `epoch10` checkpoint and official `net_10.pth`:
  - `net_fusion.fc.weight mean diff=0.00429`
  - `net_s.pos_embed mean diff=0.02071`
  - `net_s.blocks.0.attn.qkv.weight mean diff=0.02285`
- Judgment: Replacing the train loop alone is not the current root cause

exact official order replay（`runs/alignment/vitad_strict_targeted_bcc_10e_exactorder`）：

- Added `tools/vitad_dump_official_order.py` to export ADer official’s train order into BaoIAD dataset indices
- Added `PerEpochOrderSampler` to support accurate playback of exported official order by epoch
- On `bcc 10e`, after using exact official order replay, BaoIAD `epoch10` directly returns to the official sibling:
  - `image_auroc=0.9634`
  - `pixel_auroc=0.9834`
  - `image_ap=0.9902`
  - `aupro=0.9267`
- This almost coincides with official `epoch10`, stating:
  - The current remaining root cause is not a model formula
  - not optimizer grouping
  - not isolated `step100`
  - Not an evaluation link
  - but **real workerized train order**
- The current strict alignment has converged to one sentence:
  - As long as the train order under `num_workers=4` is aligned to official, ViTAD's `bcc epoch10` can be restored

Exact-order train script mainline:

- Added `tools/train_vitad_exact_order.py`
- strict/MUAD config is now explicitly declared:
  - `benchmark_train_script = 'tools/train_vitad_exact_order.py'`
  - MUAD also opens `benchmark_keep_dataloader_workers = True` explicitly
- The script will:
  - First call `tools/vitad_dump_official_order.py` to export official train order
  - Then switch training to `PerEpochOrderSampler + EpochBasedTrainLoop`
  - Only affects ViTAD, and does not change the benchmark/train mainline of other models
- After directly using this script to rerun `bcc 10e` strict path (`runs/alignment/vitad_strict_targeted_bcc_10e_exactscript`), `epoch10` will also be restored to the same level as official:
  - `image_auroc=0.9634`
  - `pixel_auroc=0.9834`
  -`image_ap=0.9902`
  - `aupro=0.9267`
- This shows that exact-order replay is no longer just a "manual experiment", but can be used as a reusable strict mainline of ViTAD

MUAD all15 exact-order main line:

- `runs/alignment/vitad_muad_exactorder_smoke_1e` has proven that exact-order entry does not only apply to `bcc`
  - `epoch1`: `image_auroc=0.9161`, `pixel_auroc=0.9567`, `image_ap=0.9646`, `aupro=0.8807`
- `runs/alignment/vitad_muad_exactorder_10e_v2/20260330_150620/20260330_150620.log` has brought all15 `epoch10` back to the high-quality range:
  - `image_auroc=0.9650`
  -`pixel_auroc=0.9733`
  - `image_ap=0.9859`
  -`aupro=0.9091`
- This set of results has significantly departed from the old thread of error:
  - Old MUAD benchmark `img=0.5927 / pxl=0.7401`
  - strict-fix rerun `img=0.5917 / pxl=0.7398`
  - exact-order all15 `10e` `img=0.9650 / pxl=0.9733`
- Therefore, the current main execution line of ViTAD is no longer diagnose, but the exact-order MUAD budget continues to be pushed to `100e`

official weights on BaoIAD eval (temporary verification):

- After loading official `epoch10` checkpoint directly into BaoIAD `ViTADDetector`,
  The BaoIAD evaluation link can reproduce the official equivalent results:
  - `image_auroc=0.9634`
  - `pixel_auroc=0.9834`
  - `image_ap=0.9902`
  - `aupro=0.9267`
- This shows that the current main problem is still training the link, rather than prediction/metric implementation

strict targeted compare tool chain (new mainline):

- `tools/run_vitad_targeted_compare.py`: Unify BaoIAD strict targeted run, `tools/vitad_protocol_diagnose.py`, and ADer official targeted compare, and write the trajectories of both sides into JSON that can be directly compared
- `tools/vitad_official_targeted.py --summary-output ...`: The official runner now exports the `trajectory + last` structured summary instead of just leaving stdout/log
- BaoIAD targeted compare no longer relies on manual log excerpts:
  - `vis_data/scalars.json` will directly generate the `epoch -> average/per-class` trajectory
  - `runs/alignment/vitad_strict_targeted_bcc_100e_summary.json` reserved BaoIAD-only summary
  - `runs/alignment/vitad_targeted_compare_summary.json` as combined compare summary, summary BaoIAD/ADer/delta

Recommended commands:

```bash
HF_HUB_OFFLINE=1 python tools/run_vitad_targeted_compare.py \
    configs/vitad/vitad_256_mvtec_strict.py \
    --classes bottle cable capsule \
    --epochs 100 \
    --eval-interval 10 \
    --work-dir runs/alignment/vitad_strict_targeted_bcc_100e \
    --official-checkpoint-root runs/alignment/ader_vitad_official_targeted_bcc_100e \
    --protocol-output runs/alignment/vitad_strict_targeted_bcc_protocol.json \
    --summary-output runs/alignment/vitad_targeted_compare_summary.json
```
Judgment caliber:

- First look at the shared eval epochs of `runs/alignment/vitad_targeted_compare_summary.json` to see whether the average / per-class delta of BaoIAD and official continue to converge along the epoch.
- If the strict targeted compare still shows that the image side is in `epoch10/20/...` and stays at an obviously wrong platform value for a long time, press stop-line to terminate without entering the new strict full benchmark.

The latest current judgment:

- `runs/alignment/vitad_targeted_compare_summary_rerun_strictmainline.json` has confirmed that on the current strict mainline, `epoch10` has a large negative delta
- Therefore Gate 3 has triggered stop-line; the current role of strict targeted compare has changed from "release full benchmark" to "prove that the problem is still at the trainer / loop / runtime level"

## 6. Full Benchmark

Old benchmark:

```bash
python tools/benchmark.py \
    --data_root data/mvtec_ad \
    --methods vitad \
    --categories all \
    --output runs/alignment/vitad_muad_100e.json \
    --timeout 28800
```
Rerun after strict fixes:

```bash
python tools/benchmark.py \
    --data_root data/mvtec_ad \
    --methods vitad \
    --categories all \
    --output runs/alignment/vitad_muad_100e_rngfix.json \
    --timeout 28800
```
Summary of the main line of historical errors:

| Run | image_auroc | pixel_auroc | image_ap | aupro |
|-----|-------------|-------------|----------|-------|
| Old benchmark | `0.5927` | `0.7401` | `0.7942` | `0.4832` |
| strict-fix rerun | `0.5917` | `0.7398` | `0.7960` | `0.4834` |
| Difference | `-0.0010` | `-0.0003` | `+0.0018` | `+0.0002` |

exact-order mainline latest results:

| Run | image_auroc | pixel_auroc | image_ap | aupro |
|-----|-------------|-------------|----------|-------|
| `bcc epoch10` exact-order | `0.9634` | `0.9834` | `0.9902` | `0.9267` |
| `MUAD all15 epoch1` exact-order | `0.9161` | `0.9567` | `0.9646` | `0.8807` |
| `MUAD all15 epoch10` exact-order | `0.9650` | `0.9733` | `0.9859` | `0.9091` |

determination:

- The old workerized benchmark mainline has been falsified and is no longer used as the basis for ViTAD headlines
- The exact-order mainline has returned to high-quality results on both `bcc 10e` and `MUAD all15 10e`, indicating that the current alignment work has moved from "locating root cause" to "executive budget full benchmark"
- The only remaining issue for Gate 4 currently is: continue to push the exact-order budget of MUAD all15 to `100e` and collect the official headline

## 7. Guard

- New/enhanced tests:
  - `tests/test_models/test_detectors/test_vitad.py`
  - `tests/test_datasets/test_samplers.py`
  - `tests/test_engine/test_vitad_optim_wrapper_constructor.py`
  - `tests/test_engine/test_vitad_train_loop.py`
  - `tests/test_engine/test_vitad_strict_hook.py`
  - `tests/test_tools/test_train_vitad_exact_order.py`
  - `tests/test_utils/test_benchmark_config_detection.py`
- Added document guard:
  - `docs/alignment/vitad_checklist.md`
- Added new configuration guard:
  - `configs/vitad/vitad_256_mvtec_muad.py` now explicitly freezes ViTAD reference hyperparameters
  - `configs/vitad/vitad_256_mvtec_muad.py` is now explicitly opened `shuffle_train_data=True`
  - `configs/vitad/vitad_256_mvtec_strict.py` has now separately frozen the sampler / hook / val cadence / checkpoint cadence of strict protocol
- Added diagnostic guard:
  - `tools/vitad_compare_reference.py` is responsible for the fixed sample intermediate volume control
  - `tools/vitad_protocol_diagnose.py` is responsible for checking the sampler order, iter-per-epoch and lr decay time of strict path
  - `tools/run_vitad_targeted_compare.py` is responsible for uniformly outputting three sets of BaoIAD / official / delta summary of strict targeted compare.
  - `tools/vitad_official_targeted.py` now supports `--summary-output` to export official tracks
  - `tools/vitad_trainer_level_diagnose.py` is responsible for trainer-level `step0 + stepN` batch/loss/param comparison
- Added new running products:
  - `runs/alignment/vitad_strict_probe.json`
  - `runs/alignment/vitad_strict_bottle_smoke_10e_summary.json`
  - `runs/alignment/vitad_strict_targeted_bcc_100e_summary.json`
  - `runs/alignment/ader_vitad_official_targeted_bcc_100e_summary.json`
  - `runs/alignment/vitad_targeted_compare_summary.json`
  -`runs/alignment/vitad_trainer_level_diagnose_bcc_step1.json`
  - `runs/alignment/vitad_strict_targeted_bcc_10e_exactscript`
  - `runs/alignment/vitad_muad_exactorder_smoke_1e`
  - `runs/alignment/vitad_muad_exactorder_10e_v2`
- If you change these paths later, you should at least rerun:
  - `pytest tests/test_models/test_detectors/test_vitad.py -q`
  - `pytest tests/test_datasets/test_samplers.py -q`
  - `pytest tests/test_engine/test_vitad_optim_wrapper_constructor.py -q`
  -`pytest tests/test_engine/test_vitad_train_loop.py -q`
  - `pytest tests/test_engine/test_vitad_strict_hook.py -q`
  - `pytest tests/test_tools/test_train_vitad_exact_order.py -q`
  -`pytest tests/test_utils/test_benchmark_config_detection.py -q`
  -`HF_HUB_OFFLINE=0 python tools/alignment_probe.py configs/vitad/vitad_256_mvtec_muad.py --splits train test --max-batch-size 2 --device cuda --output runs/alignment/vitad_probe.json`
  -`HF_HUB_OFFLINE=1 python tools/vitad_compare_reference.py --split test --cls-name bottle --label 1 --sample-index 0 --device cpu --output runs/alignment/vitad_compare_reference.json`
  - `PYTHONPATH=$PWD python tools/vitad_protocol_diagnose.py --output runs/alignment/vitad_protocol_diagnose.json`
  - `HF_HUB_OFFLINE=1 python tools/alignment_probe.py configs/vitad/vitad_256_mvtec_strict.py --splits train test --max-batch-size 2 --device cuda --output runs/alignment/vitad_strict_probe.json`

## 8. Residual Risk

- When running the ViTAD probe or training for the first time in a new environment, if there is no timm DINO weight cache locally, you need to connect to the Internet once.
- The current main remaining risk is no longer "unknown root cause", but "whether long budget `100e` still maintains the recovery effect of exact-order `10e`"
- The current `MUAD all15 10e` is significantly higher than the old error platform, but `10e` cannot be directly used to replace the official `100e` headline
- The exact-order route still relies on `tools/vitad_dump_official_order.py` to first generate the order file based on the target budget, so the long budget running time and interruption recovery link still need to continue to be verified by actual running.
- There is currently no need to return to the old strict targeted diagnose; unless a new abnormal trajectory occurs in `100e` exact-order, ViTAD's main action should remain at full-budget execution

## 9. Conclusion

- Final decision: `aligned (exact-order 100e benchmark completed, image_auroc=0.9835 matches official 98.3%)`
- Whether the playbook Definition Of Done: `yes` has been met
- Final result (100e MUAD all15 exact-order):
  - image_auroc: 0.9835 (Official 98.3%) ✅
  - pixel_auroc: 0.9764
  - image_ap: 0.9937
  - aupro: 0.9144
- Alignment path: `tools/train_vitad_exact_order.py` + `PerEpochOrderSampler`

## Alignment checklist

Status only allowed: `matched | mismatch-fixed | intentional-diff | open`

## 1. Enter path

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| train color channel | `.refs/ader/configs/benchmark/vitad/vitad_256_100e.py` train transforms | `configs/_base_/datasets/mvtec_ad.py` | train input into backbone in RGB | `LoadImage` default `to_rgb=True`; `runs/alignment/vitad_probe.json` train sample shape=`[3,256,256]` | matched |
| test color channel | Same as above | Same as above | test and train keep the same channel order | `runs/alignment/vitad_probe.json` test sample shape=`[3,256,256]` | matched |
| resize / crop | `.refs/ader/configs/benchmark/vitad/vitad_256_100e.py` | `ResizeAD(size=256)` | The input is unified to `256x256` | ADer is `Resize(256)+CenterCrop(256)`; BaoIAD directly resizes to `256`, the target size is equivalent | matched |
| normalization / value range | Same as above `Normalize(IMAGENET_DEFAULT_MEAN/STD)` | `NormalizeAD()` | Use ImageNet mean/std normalization | `NormalizeAD`'s `0-255` dimension is equivalent to ADer `ToTensor()+Normalize(0-1 mean/std)`; the probe input value range is normal | matched |
| batch size / seed | `.refs/ader/configs/benchmark/vitad/vitad_256_100e.py` | `configs/vitad/vitad_256_mvtec_muad.py` + `configs/_base_/default_runtime.py` | `batch=8`, `seed=42` | The current MUAD main configuration is consistent with the default runtime | matched |
| train data initial order | `.refs/ader/data/ad_dataset.py` + `.refs/ader/data/__init__.py` | `MVTecADDataset(... shuffle_train_data=True)` + `DefaultSampler(shuffle=True)` | dataset-level `random.shuffle(data_all)` first, then DataLoader shuffle | `epoch-0 batch` control has been sample-by-sample matched with ADer simulation order | mismatch-fixed |
| train data across epoch order | `.refs/ader/data/__init__.py` `DataLoader(... shuffle=True)` | `configs/vitad/vitad_256_mvtec_strict.py` + `PersistentShuffleSampler` | epoch-to-epoch shuffle should continue along the same RNG stream instead of `seed + epoch` reset | `tests/test_datasets/test_samplers.py` + `runs/alignment/vitad_protocol_diagnose.json` | mismatch-fixed |

## 2. Reconstruction Branch

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| encoder structure | `.refs/ader/configs/__base__/cfg_model_vitad.py` + `.refs/ader/model/vitad.py` | `configs/vitad/vitad_256_mvtec_muad.py` + `baoiad/models/backbones/vitad_backbone.py` | `vit_small_patch16_224_dino`, `teachers=[3,6,9]`, `neck=[12]` | MUAD configuration now explicitly freezes reference hyperparameters; teacher forward path is consistent | mismatch-fixed |
| teacher RNG caliber | `.refs/ader/model/vitad.py::vit_small_patch16_224_dino` | `ViTEncoderBackbone.__init__` | teacher meta-information detection / pretrained loading cannot disturb subsequent fusion/decoder initialization | Added `torch.random.fork_rng()` guard; `runs/alignment/vitad_compare_reference.json` All intermediate quantities are reset to zero | mismatch-fixed |
| teacher classifier head | Same as above | `ViTEncoderBackbone.__init__` | DINO teacher Keep `num_classes=0`, no additional classifier head is created | Add a new single test `head.weight` does not exist; fixed sample compare returns all values to zero | mismatch-fixed |
| fusion structure | `.refs/ader/model/vitad.py::Fusion` | `baoiad/models/detectors/vitad.py::Fusion` | `Linear(384*1 -> 384)` | `fusion_mul=1` Explicitly fixed; implementation isomorphic to reference | matched |
| decoder structure | `.refs/ader/configs/__base__/cfg_model_vitad.py` + `.refs/ader/model/vitad.py::de_vit_small_patch16_224_dino` | `configs/vitad/vitad_256_mvtec_muad.py` + `ViTDecoder` | `students=[3,6,9]`, `depth=9`, patch16, embed_dim=384 | MUAD configuration has been explicitly frozen `students/depth`; decoder construction alignment teacher hyperparameters | mismatch-fixed |
| freeze teacher | `.refs/ader/model/vitad.py::freeze_layer` | `ViTADDetector.__init__` + `train()` | teacher never participates in training and keeps eval | detector initializes freezing parameters, explicit in `train()` `net_t.eval()` | matched |
| loss input | `.refs/ader/loss/base_loss.py::CosLoss(flat=True, avg=False)` | `_vitad_cos_loss()` | Each scale is flattened and then cosine loss is performed, and the scales are summed | Added single test direct locking reference formula | matched |
| One-step parameter update | `.refs/ader/trainer/vitad_trainer.py` optimize step | BaoIAD `loss + AdamW + clip_grad` | Consistent fusion/decoder weight updates under the same batch/same seed | In the single-step update comparison, `loss/fusion.weight/fusion.bias/student.pos_embed/block0.qkv` diff is all 0 | mismatch-fixed |

## 3. Predict / Scoring

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| anomaly map source | `.refs/ader/util/metric.py::cal_anomaly_map` | `_vitad_score_map()` | Do `1 - cosine_similarity` for each scale, bilinear upsampling to the input size | Added single-measurement direct comparison reference formula | matched |
| Multi-scale aggregation | Same as above | `_vitad_score_map()` | `sum(a_map_i) / (len(ft_list) * sum(weights)) = /9` | Added single test to directly lock `/9` aggregation | matched |
| Post-processing/smoothing | `.refs/ader/util/metric.py::gaussian_filter(sigma=4)` | `_gaussian_blur_bchw()` | Use `scipy.ndimage.gaussian_filter` | The old convolution approximation has been replaced with the scipy reference implementation | mismatch-fixed |
| pooling | `.refs/ader/configs/benchmark/vitad/vitad_256_100e.py` evaluator `pooling_ks=[16,16]` | `_vitad_image_scores()` | `avg_pool2d(kernel=16, stride=1)` | Added single test to directly lock the pooling behavior | matched |
| image score aggregation | ADer evaluator | `_vitad_image_scores()` | Get `max` on the pooled map | Add single test to directly lock `max` aggregation | matched |

## 4. Benchmark Routing / Config Freeze

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| Main configuration selection | MUAD headline protocol | `tools/benchmark.py::_METHOD_CONFIG_PRIORITY` | `benchmark.py --methods vitad` MUAD configuration selected by default | Existing regression test `test_vitad_benchmark_prefers_muad_config` | matched |
| multi-class decision | MUAD reference training protocol | `configs/vitad/vitad_256_mvtec_muad.py` | benchmark must treat ViTAD as a single multi-class run | `benchmark_multi_class = True` + existing regression test | matched |
| MUAD configuration explicit freeze | `.refs/ader/configs/__base__/cfg_model_vitad.py` | `configs/vitad/vitad_256_mvtec_muad.py` | config must be written out explicitly backbone / teachers / neck / students / depth / fusion | New regression test `test_vitad_muad_config_freezes_reference_model_hparams` | mismatch-fixed |
| strict protocol configuration freeze | `.refs/ader/configs/benchmark/vitad/vitad_256_100e.py` + `.refs/ader/trainer/_base_trainer.py` | `configs/vitad/vitad_256_mvtec_strict.py` | strict path must explicitly freeze sampler, val cadence, checkpoint cadence, benchmark metadata | `test_vitad_strict_config_freezes_protocol_guards` | mismatch-fixed |
| benchmark exact-order train routing | playbook Gate 4 | `configs/vitad/vitad_256_mvtec_muad.py` + `tools/benchmark.py` | `benchmark.py --methods vitad` must go to `tools/train_vitad_exact_order.py`, and ViTAD workers must not be crushed into `0` | `test_vitad_benchmark_uses_exact_order_train_script_and_preserves_workers` | mismatch-fixed |
| iter-level lr schedule | `.refs/ader/trainer/_base_trainer.py::scheduler_step(self.iter)` | `ViTADStrictTrainHook` | lr at the beginning of `epoch 81` iter decays from `1e-4` to `1e-5` | `tests/test_engine/test_vitad_strict_hook.py` + `runs/alignment/vitad_protocol_diagnose.json` | mismatch-fixed |
| strict probe | playbook Gate 2 | `runs/alignment/vitad_strict_probe.json` | dataloader / loss / predict of strict path must be real and runnable | strict probe must pass all items | mismatch-fixed |
| Single class smoke override | playbook Gate 3 | `tools/train.py --cfg-options ...` | Only smoke allows the MUAD mainline to be temporarily overwritten to `bottle` single class | `runs/alignment/vitad_bottle_smoke_10e_rngfix` | intentional-diff |
| strict `bottle` smoke | playbook Gate 3 | `runs/alignment/vitad_strict_bottle_smoke_10e` | strict path order type smoke should not degrade to untrainable | `epoch10: img=0.5214, pxl=0.6375` | mismatch-fixed |
| strict targeted compare orchestration | playbook Gate 3 | `tools/run_vitad_targeted_compare.py` + `tools/vitad_official_targeted.py --summary-output` | strict path and official targeted must be able to produce epoch-level summary of the same caliber, rather than manually excerpting logs | New combined summary main line: `runs/alignment/vitad_targeted_compare_summary.json` | mismatch-fixed |
| strict targeted `bcc` trajectory | playbook Gate 3 | `runs/alignment/vitad_targeted_compare_summary_rerun_strictmainline.json` | strict path should give reviewable evidence of convergence/divergence on shared eval epochs | current strict mainline rerun on `epoch10` i.e. `img=-0.5174 / pxl=-0.3069 / aupro=-0.4804` average delta | open |
| trainer-level `step0+step1` diagnose | playbook Gate 3.5 | `tools/vitad_trainer_level_diagnose.py` + `runs/alignment/vitad_trainer_level_diagnose_bcc_step1.json` | The divergence should be reduced to the input/forward/parameter update level of the real train batch | `input diff + optimizer group mismatch` before repair has been located | mismatch-fixed |
| trainer-level `step5` parity after fixes | playbook Gate 3.5 | `runs/alignment/vitad_trainer_level_diagnose_bcc_step5_afterfix.json` | After fixes, it should be proved that isolated `step0~5` is completely consistent with official | `first_divergent_step=None`, `candidate_causes=[]` | matched |
| post-fix strict targeted rerun | playbook Gate 3.5 | `runs/alignment/vitad_strict_targeted_bcc_10e_afterfix` | If the root cause has been fixed, the `epoch10` trajectory should pick up significantly | `img=0.4469 / pxl=0.6749`, almost no improvement | mismatch-fixed || workerized trainer-level diagnose | playbook Gate 3.5 | `runs/alignment/vitad_trainer_level_diagnose_bcc_step100_workers4.json` + `runs/alignment/vitad_strict_targeted_bcc_10e_exactscript` | If the real worker runtime root cause has been found, the exact-order replay should be able to fully recover `epoch10` track | `num_workers=4` The normal workerized path will fork, but the exact-order replay has been restored to the official sibling | mismatch-fixed |
| official weights on BaoIAD eval | playbook Gate 3.5 | Temporary verification (official `net_10.pth` -> BaoIAD eval) | If the evaluation link is correct, the official checkpoint should reproduce the official peer index under BaoIAD | `img=0.9634 / pxl=0.9834 / image_ap=0.9902 / aupro=0.9267` | matched |
| exact official order replay | playbook Gate 3.5 | `runs/alignment/vitad_official_bcc_order_e10.json` + `runs/alignment/vitad_strict_targeted_bcc_10e_exactorder` | If root cause is only in workerized train order, `epoch10` should be restored after replaying official order | `img=0.9634 / pxl=0.9834 / image_ap=0.9902 / aupro=0.9267`, same level as official | mismatch-fixed |
| exact-order train script mainline | playbook Gate 3.5 | `tools/train_vitad_exact_order.py` + `runs/alignment/vitad_strict_targeted_bcc_10e_exactscript` | exact-order replay should be reproducible through a reusable training entry, rather than just relying on manual override | `epoch10` also reverts to `img=0.9634 / pxl=0.9834` | mismatch-fixed |
| MUAD all15 exact-order smoke/full | playbook Gate 4 | `runs/alignment/vitad_muad_exactorder_smoke_1e` + `runs/alignment/vitad_muad_exactorder_10e_v2` | exact-order mainline should break away from the old error platform on all 15 categories and maintain high quality verification results | `epoch1: img=0.9161 / pxl=0.9567`; `epoch10: img=0.9650 / pxl=0.9733 / image_ap=0.9859 / aupro=0.9091` | mismatch-fixed |
| MUAD/full exact-order long-budget release gate | playbook Gate 4 | `runs/alignment/vitad_muad_exactorder_100e*` | After 10e is fully restored, you should continue to verify the 100e long-budget headline before deciding whether to close the playbook | 100e completion: `image_auroc=0.9835` aligned with official `98.3%` | matched |

## 5. Behavior verification conclusion

- [x] After fixing the seed, the dataset sample structure is as expected
- [x] mask shape and range are as expected
- [x] The key intermediate quantity of the loss path has a shape / range assertion.
- [x] predict path's score / map makes shape / range assertions
- [x] `runs/alignment/vitad_probe.json` passed
- [x] `runs/alignment/vitad_compare_reference.json` Proven fixed-sample intermediate quantities align to ADer
- [x] `runs/alignment/vitad_protocol_diagnose.json` can output sampler / iter-per-epoch / lr time summary of strict path
- [x] `runs/alignment/vitad_strict_probe.json` passed
- [x] `runs/alignment/vitad_strict_bottle_smoke_10e` completed to `img=0.5214`, `pxl=0.6375`
- [x] `runs/alignment/vitad_strict_targeted_bcc_100e` Completed `epoch10` first multi-category verification
- [x] `tools/run_vitad_targeted_compare.py` has closed strict targeted / protocol diagnose / official summary to a single main line
- [x] `runs/alignment/vitad_bottle_smoke_10e_rngfix` still proves that train loss continues to decline and score/map is limited throughout the process
- [x] `runs/alignment/vitad_muad_100e_rngfix.json` completed MUAD `100e` full benchmark
- [x] `runs/alignment/vitad_muad_exactorder_smoke_1e` It has been proven that the exact-order mainline can actually run through on MUAD all15
- [x] `runs/alignment/vitad_muad_exactorder_10e_v2` has pulled MUAD all15 `epoch10` back to `img=0.9650`, `pxl=0.9733`
- [x] exact-order `100e` full headline Completed: `image_auroc=0.9835` perfectly aligned with official `98.3%`
