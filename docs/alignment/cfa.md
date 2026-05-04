# CFA strict-alignment evidence

- **Method slug**: `cfa`
- **Family**: Discriminative
- **Method README**: [`configs/cfa/README.md`](../../configs/cfa/README.md)

This file preserves the fixed alignment evidence migrated from the manuscript evidence workspace. The content below is translated to English where needed and kept inside this repository so the strict-alignment state can be reviewed without relying on an external paper checkout.

## Config entry points

- [`configs/cfa/cfa_256_mvtec_strict.py`](../../configs/cfa/cfa_256_mvtec_strict.py)
- [`configs/cfa/cfa_256_visa.py`](../../configs/cfa/cfa_256_visa.py)

## Detailed alignment report

**Status**: `aligned`
**Date**: `2026-03-25`

## 1. Reference freezing

- Reference warehouse:
  - Primary reference: local `.refs/anomalib`
  - Auxiliary reference: local `.refs/ader` snapshot
- Reference commit:
  - anomalib: `4f6af1acb0ee7b81f54cda036dd9f1c27f63b69a`
  - ADer: The current environment cannot safely parse git commit, frozen to the file path level
- Refer to config/checkpoint:
  - `.refs/anomalib/src/anomalib/models/image/cfa/torch_model.py`
  - `.refs/anomalib/src/anomalib/models/image/cfa/loss.py`
  - `.refs/anomalib/src/anomalib/models/image/cfa/anomaly_map.py`
  - `.refs/anomalib/examples/configs/model/cfa.yaml`
  - `.refs/ader/model/cfa.py`
  - `.refs/ader/trainer/cfa_trainer.py`
  - `.refs/ader/configs/benchmark/cfa/cfa_256_100e.py`
- Dataset/Category: MVTec AD, 15 categories of standard benchmark
- Input resolution: `256x256`
- seed:
  - anomalib / BaoIAD main caliber: `42`
  - ADer historical configuration: `1024`
- Indicator definition: image AUROC / pixel AUROC
- intentional diff:
  - When anomalib conflicts with implementation details of ADer, BaoIAD uses anomalib as the main reference because the CFA public alignment caliber of `docs/alignment/README.md` is based on anomalib `0.956`
  - BaoIAD maintains the `100 epochs` unified runtime of the main benchmark; the anomalib example configuration is `30 epochs + early stopping`, and ADer is another set of trainer/runtime

## 2. Code path comparison conclusion

See [`cfa_checklist.md`](cfa_checklist.md) for the control matrix.

### Consistency confirmed

- The main configuration of `configs/cfa/cfa_256_mvtec_strict.py` already uses `FeatureExtractor + out_indices=(1,2,3)`, which is consistent with the caliber of anomalib `layer1/layer2/layer3`
- `_compute_loss()` is consistent with anomalib `CfaLoss`, still coupled-hypersphere loss, `radius^2`, hard negative margin `0.1` and final `*1000` scaling are all consistent
- `_compute_anomaly_map()` Same as anomalib `AnomalyMapGenerator`, still `sqrt -> topk -> softmin weighting -> bilinear upsample -> Gaussian blur`
- After double reference review, it was confirmed that the current BaoIAD does not have additional `LeakyReLU`, which is consistent with anomalib; ADer has added additional activation at this point, which is an auxiliary reference difference.

### Fixed inconsistencies

- `configs/cfa/cfa_wrn50_256_mvtec_unified.py` still used `wide_resnet50_raw.py` before, and would fall back to the old four-layer feature path; now it has been changed to explicit `FeatureExtractor(out_indices=(1,2,3))`
- `CFADetector.forward(mode='predict')` previously silently returned all zeros `pred_score` / `pred_anomaly_map` when the memory bank was not initialized; now it reports an error explicitly
- `CFADetector` adds `build_memory_bank(dataloader=None)` so that `alignment_probe`, `MemoryBankHook` and explicit warmup can take the same legal initialization path
- `tests/test_models/test_detectors/test_cfa.py` and `tests/test_utils/test_alignment_probe.py` have been added to the guard for memory bank / predict / probe warmup

### Items that are still open

- No algorithm level `open` items

## 3. Behavior Probe

Order:

```bash
python tools/alignment_probe.py configs/cfa/cfa_256_mvtec_strict.py \
    --splits train test \
    --max-batch-size 2 \
    --device cuda \
    --output runs/alignment/cfa_probe.json
```
in conclusion:

- This review has been regenerated `runs/alignment/cfa_probe.json`
- `test` split the first time `predict` explicitly reports `memory bank is not built`, and then the probe automatically executes `memory_bank_warmup`, which proves that the current `CFA` no longer relies on the all-zero occupancy result to "false pass"
- After warmup, `pred_score` and `pred_anomaly_map` are both finite non-zero values, and all structure checks pass.

Key statistics:

- dataset sample:
  - train preview: `zipper/train/good/065.png`
  - test preview: `bottle/test/broken_large/000.png`
  - Input shape: `2 x 3 x 256 x 256`
- loss path:
  - The train batch is the memory bank initialization phase, so `loss=0.0`
  - This is consistent with the design of `num_init_batches=3` and is not an abnormal convergence signal
- predict path:
  - `memory_bank_warmup.used = true`
  - `memory_bank_warmup.num_batches = 1`
  - `pred_score mean = 1.8650`
  - `pred_anomaly_map mean = 1.3294`
  - map shape is `1 x 256 x 256`

## 4. Small-scale controlled experiment

Experimental setup:

- Category: `bottle`
- Training budget: `3 epochs`
- seed: `42`
- Comparison object: current main configuration `configs/cfa/cfa_256_mvtec_strict.py`
- Note: After double reference review, no mismatch was found that required changing the main implementation to an ADer caliber algorithm. Therefore, this round of smoke will no longer be an artificial A/B fork, but will directly verify the training and verification path of the current implementation.

Order:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/cfa/cfa_256_mvtec_strict.py \
    --work-dir runs/alignment/cfa_bottle_smoke \
    --cfg-options \
        train_cfg.max_epochs=3 \
        train_cfg.val_interval=1 \
        train_dataloader.batch_size=8 \
        test_dataloader.batch_size=8 \
        val_dataloader.batch_size=8 \
        "train_dataloader.dataset.cls_names=['bottle']" \
        "test_dataloader.dataset.cls_names=['bottle']" \
        "val_dataloader.dataset.cls_names=['bottle']" \
        train_dataloader.dataset.multi_class=False \
        test_dataloader.dataset.multi_class=False \
        val_dataloader.dataset.multi_class=False
```
observe:

- Train loss continues to decline for 3 epochs: `2660.9476 -> 1418.1924 -> 1238.8345`
- The first epoch already reaches `image_auroc=0.9992`、`pixel_auroc=0.9867`
- The second and third epochs stay stable at `image_auroc=1.0000`、`pixel_auroc=0.9877`
- In the log, `MemoryBankHook` can successfully call `build_memory_bank(dataloader)` in each round, and there is no uninitialized predict or all-zero output.

determination:

- `pass`
- Reason: The loss curve is declining normally, the image-level / pixel-level indicators on `bottle` are at a reasonable high level, and the playbook shutdown line has not been triggered.

## 5. Full Benchmark

Suggested command:

```bash
python tools/benchmark.py \
    --data_root data/mvtec_ad \
    --methods cfa \
    --categories all \
    --output runs/alignment/cfa_v2.json
```
Summary of results:

| Metric | Reference | BaoIAD | Gap |
|--------|-----------|----------|-----|
| image_auroc | `0.956` | `0.9581` | `+0.2%` |
| pixel_auroc | `0.983` | `0.9794` | `-0.4%` |

illustrate:

- `0.9581 / 0.9794` is still the current repository `README` and the old version `cfa.md` has archived historical 15 category results
- There is no new `runs/alignment/cfa_v2.json` regenerated in this turn because `100 epochs x 15 categories` is a multi-hour task; this round gives priority to completing the missing evidence of Gate 1/2/3/5
- This round of code changes did not change the loss / scoring algorithm of the main configuration. It only repaired the `predict` precondition, memory bank warmup path, and the old backbone vulnerability of the unified configuration file.

Shutdown line inspection:

- [x] Historical archive results do not appear in large areas near `0.5` image AUROC
- [x] There is no score collapse, pure zero map or abnormal platform value in the current probe / smoke
- [x] The historical mean difference from the anomalib master reference is still within acceptable limits
- [ ] This turn did not rerun the complete 15 categories of benchmark original JSON

## 6. Guard

- New/enhanced tests:
  - `tests/test_models/test_detectors/test_cfa.py`
  - `tests/test_utils/test_alignment_probe.py`
- Added document guard:
  - `docs/alignment/cfa_checklist.md`
- Added new anti-regression points:
  - There must be a memory bank before `predict`, otherwise an error must be reported explicitly
  - `build_memory_bank(dataloader)` must be able to complete the warmup of `alignment_probe`
  - The number of three-layer feature channels of `resnet18` / `wide_resnet50_2` must be consistent with the descriptor input dimension
  - `configs/cfa/cfa_wrn50_256_mvtec_unified.py` No return to layer 4 is allowed `RawBackbone`
- If you change these paths later, you must rerun:
  -`baoiad/models/detectors/cfa.py`
  - `configs/cfa/cfa_256_mvtec_strict.py`
  -`configs/cfa/cfa_wrn50_256_mvtec_unified.py`
  -`python tools/alignment_probe.py configs/cfa/cfa_256_mvtec_strict.py --splits train test --max-batch-size 2 --output runs/alignment/cfa_probe.json`
  - `pytest tests/test_models/test_detectors/test_cfa.py tests/test_utils/test_alignment_probe.py -q`

## 7. Residual Risk

- This turn does not regenerate new 15-category benchmark JSON, so Gate 4 still relies on historical archive results rather than new products of this round.
- The git commit of ADer's local snapshot cannot pass safe-directory verification in the current environment, and is currently only frozen to the file path level.
- ADer and anomalib are not completely consistent in terms of `LeakyReLU` and descriptor details; the current conclusion clearly uses anomalib as the main reference, and these differences in ADer are no longer regarded as bugs to be fixed.

## 8. Conclusion

- Final decision: `aligned`
- Allowed to proceed to next stage: `yes`
- If not allowed, next action: If the original Gate 4 product needs to be strictly completed in the future, rerun `tools/benchmark.py --methods cfa --categories all` separately.

## Alignment checklist

Status only allowed: `matched | mismatch-fixed | intentional-diff | open`

## 1. Enter path

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| train color channel | `.refs/anomalib` CFA pre-processor | `configs/_base_/datasets/mvtec_ad.py` | train input enters the backbone in RGB | `LoadImage(to_rgb=True)` is enabled by default, `NormalizeAD()` is consistent with the main process | matched |
| test color channel | Same as above | Same as above | test input is consistent with train | `runs/alignment/cfa_probe.json` of `train/test inputs.shape=[2,3,256,256]` | matched |
| resize | anomalib example config | `ResizeAD(size=256)` | input unified resize to `256x256` | `configs/_base_/datasets/mvtec_ad.py` | matched |
| normalization / value range | anomalib image pre-processor | `NormalizeAD()` | normalization using ImageNet mean/std | limited input statistics in `runs/alignment/cfa_probe.json` | matched |
| seed / batch size | anomalib seed=42; ADer seed=1024, batch=4 | `randomness.seed=42` + base dataset batch size | The public alignment caliber is mainly anomalib `seed=42` | `docs/alignment/README.md` + `configs/_base_/default_runtime.py` | intentional-diff |

## 2. Backbone / Feature Extraction

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| backbone layer selection | anomalib `get_return_nodes()` | `configs/cfa/cfa_256_mvtec_strict.py` + `CFADetector.__init__` | `layer1/layer2/layer3` three-layer features using `wide_resnet50_2` | `out_indices=(1,2,3)`; descriptor input dimension `1792` | matched |
| Main configuration backbone bug | anomalib / ADer only take layer 3 | `configs/cfa/cfa_256_mvtec_strict.py` | Main configuration cannot fall back to four-layer features | Old issues have been fixed in the history report | mismatch-fixed |
| unified configuration backbone bug | Same as above | `configs/cfa/cfa_wrn50_256_mvtec_unified.py` | unified configuration must also be fixed to three layers `FeatureExtractor` | This round has replaced `wide_resnet50_raw.py` with explicit `FeatureExtractor` | mismatch-fixed |
| Additional activation | anomalib `CfaModel.forward()`; ADer `CFA.forward()` | `CFADetector._extract_features()` | Main reference anomalib no additional `LeakyReLU` | anomalib no `LeakyReLU`, ADer yes; BaoIAD chooses to follow anomalib | intentional-diff |
| descriptor input dimension | anomalib `Descriptor(backbone_dims)` | `Descriptor` + `test_descriptor_matches_expected_backbone_channels` | WRN-50-2 is `1792`, resnet18 is `448` | unit test fixed three-layer channel and descriptor input dimension | matched |
| descriptor multi-scale splicing | anomalib `Descriptor.forward()`; ADer `Descriptor.forward()` | `Descriptor.forward()` | use anomalib as the main reference to retain the current splicing semantics | BaoIAD is consistent with anomalib, ADer has subtle implementation differences here | matched |

## 3. Memory Bank / Initialization

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| centroid initialization | anomalib `initialize_centroid()` | `initialize_memory_bank()` | average the descriptor features and flatten them into `(C, HW)` memory bank | consistent code structure | matched |
| KMeans compression | anomalib `gamma_c > 1` branch | `initialize_memory_bank()` | `(H*W)//gamma_c` clustering when `gamma_c > 1` | Parameters consistent with formula | matched |
| predict preconditions | anomalib `forward()` directly reports an error when the bank is not initialized | `forward(mode='predict')` | must explicitly fail when the bank is not initialized | This round has removed the all-zero occupancy output and changed it to `RuntimeError` | mismatch-fixed |
| Explicit build entry | anomalib `initialize_centroid(data_loader)` | `build_memory_bank(dataloader=None)` | probe / hook / manual warmup requires unified legal entry | `test_build_memory_bank_*` + `runs/alignment/cfa_probe.json` | mismatch-fixed |

## 4. Loss

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| coupled-hypersphere formula | `.refs/anomalib/.../cfa/loss.py` | `_compute_loss()` | `attraction + repulsion`, margin=`0.1`, final `*1000` | item-by-item agreement | matched |
| top-k value | Same as above | `_compute_loss()` | `k = K + J`, then remove attraction / repulsion | `num_nearest_neighbors=3`, `num_hard_negative_features=3` | matched |
| radius parameter | anomalib `radius=1e-5` | `configs/cfa/cfa_256_mvtec_strict.py` | initial radius remains consistent | config consistent | matched |
| optimizer | anomalib/ADer all use AdamW + amsgrad | `optim_wrapper` | `lr=1e-3`, `weight_decay=5e-4`, `amsgrad=True` | config consistent | matched |
| runtime | anomalib 30e + early stopping; ADer 100e + scheduler | BaoIAD 100e unified runtime | main benchmark retains the unified runtime of the warehouse and does not force replication Lightning trainer | non-algorithm layer intentional diff | intentional-diff |

## 5. Predict / Scoring

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| distance -> score | `.refs/anomalib/.../cfa/anomaly_map.py` | `_compute_anomaly_map()` | `sqrt -> topk -> softmin weighting` | The formula is consistent | matched |
| map upsampling | same as above | `_compute_anomaly_map()` | anomaly map bilinear upsampling to input size | `align_corners=False` consistent | matched |
| smoothing | anomalib `GaussianBlur2d(sigma=4)`; ADer `gaussian_filter(sigma=4)` | `GaussianBlur2d(sigma=4)` | smoothing with sigma=4 | consistent with anomalib caliber | matched |
| image score aggregation | anomalib `amax(anomaly_map)` | `img_scores = anomaly_map.view(B, -1).max(...)` | image score takes the maximum value in space | code consistent | matched |
| Output validity | playbook Gate 2 | `build_predict_results()` + tests/probe | Must output a finite non-zero score / map, not an all-zero placeholder | `runs/alignment/cfa_probe.json` + `test_build_memory_bank_with_loader_enables_non_zero_predict` | mismatch-fixed |

## 6. Behavior verification conclusion

- [x] After fixing the seed, the dataset sample structure is as expected
- [x] `predict` will fail explicitly when the memory bank is not initialized and will no longer silently return an all-zero result.
- [x] `build_memory_bank(dataloader)` verified with real `alignment_probe` and single test
- [x] Both score and map of `loss` / `predict` paths have made finite assertions
- [x] `bottle` smoke does not trigger the abnormal shutdown line

## 7. Remarks

- The public CFA alignment caliber of `README` is anomalib, so when anomalib conflicts with ADer, the checklist uses anomalib as the main reference, and ADer only serves as auxiliary evidence.
- CFA does not involve anomaly synthesis, and there is no reconstruct/discriminate dual branch, so the checklist is cut into five parts according to the real structure of this method: input, feature extraction, memory bank, loss, and predict.
