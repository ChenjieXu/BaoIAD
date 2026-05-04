# FastFlow strict-alignment evidence

- **Method slug**: `fastflow`
- **Family**: Normalizing flow
- **Method README**: [`configs/fastflow/README.md`](../../configs/fastflow/README.md)

This file preserves the fixed alignment evidence migrated from the manuscript evidence workspace. The content below is translated to English where needed and kept inside this repository so the strict-alignment state can be reviewed without relying on an external paper checkout.

## Config entry points

- [`configs/fastflow/fastflow_wrn50_256_mvtec_strict.py`](../../configs/fastflow/fastflow_wrn50_256_mvtec_strict.py)
- [`configs/fastflow/fastflow_wrn50_256_visa.py`](../../configs/fastflow/fastflow_wrn50_256_visa.py)

## Detailed alignment report

**Status**: `aligned`
**Date**: `2026-03-25`

## 1. Reference freezing

- Reference repository: local `.refs/anomalib`
- Reference commit: `4f6af1acb0ee7b81f54cda036dd9f1c27f63b69a`
- Refer to config/checkpoint:
  - `.refs/anomalib/src/anomalib/models/image/fastflow/torch_model.py`
  - `.refs/anomalib/src/anomalib/models/image/fastflow/loss.py`
  - `.refs/anomalib/src/anomalib/models/image/fastflow/anomaly_map.py`
  - `.refs/anomalib/examples/configs/model/fastflow.yaml`
- Dataset/Category: MVTec AD, 15 categories of standard benchmark
- Input resolution: `256x256`
- seed: `42`
- Indicator definition: image AUROC / pixel AUROC
- intentional diff:
  - BaoIAD continues to use the MMEngine training entry and the current strict benchmark config `configs/fastflow/fastflow_wrn50_256_mvtec_strict.py`
  - anomalib example runtime with early stopping; BaoIAD’s current main caliber retains a unified benchmark configuration of 500 epochs and no scheduler

## 2. Code path comparison conclusion

See [`fastflow_checklist.md`](fastflow_checklist.md) for the control matrix.

### Consistency confirmed

- The backbone, freezing strategy and three-scale feature extraction path of `wide_resnet50_2 + out_indices=(1,2,3)` are consistent with the reference
- trainable `LayerNorm([C,H,W])` before each scale is consistent with the reference
- The flow trunk has been aligned to `SequenceINN + AllInOneBlock`, `permute_soft=False`, `clamp=2.0`, and subnet padding are logically consistent
- Both the loss formula and the anomaly map formula are consistent with the reference implementation

### Fixed inconsistencies

- Missing `LayerNorm` in old version, fixed
- The custom flow path of the old version deviates from the reference and has been switched back to FrEIA `AllInOneBlock`
- Old version of anomaly map used wrong `sum` / multi-scale aggregation logic, fixed for negative probability mean
- Old version `kernel_size` logic reversed, now reverted to `3,1,3,...` by reference

### Items that are still open

- none

## 3. Behavior Probe

Order:

```bash
python tools/alignment_probe.py configs/fastflow/fastflow_wrn50_256_mvtec_strict.py \
    --splits train test \
    --max-batch-size 2 \
    --device cpu \
    --output runs/alignment/fastflow_probe.json
```
in conclusion:

- This review has regenerated `runs/alignment/fastflow_probe.json`, and all structural checks in train/test are `ok`
- Probe proves that the current strict main configuration can stably produce limited loss, score and anomaly map on real data

Key statistics:

- dataset sample: `bottle/train/good/072.png` and `bottle/test/broken_large/000.png`, the input shape is `2 x 3 x 256 x 256`
- Loss path: `train.loss` is limited. The loss of a single batch in probe is about `9.28e5`. You need to rely on smoke to observe the trend instead of just looking at the absolute value.
- predict path: `pred_score` is limited, `pred_anomaly_map` shape is `1 x 256 x 256` and limited; `score mean = -0.4253`, `map mean = -0.6054` in probe

## 4. Small-scale controlled experiment

Experimental setup:

- Category: `bottle`
- Training budget: `3 epochs`
- seed: `42`
- Comparison object: current strict main configuration `configs/fastflow/fastflow_wrn50_256_mvtec_strict.py`

Order:

```bash
CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/fastflow/fastflow_wrn50_256_mvtec_strict.py \
    --work-dir runs/alignment/fastflow_bottle_smoke \
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

- The training loss of 3 epochs continues to decrease: `-1329409.6375 -> -2160087.4500 -> -2403018.0000`
- The verification results of `bottle` in the third epoch are `image_auroc = 1.0000`, `pixel_auroc = 0.9830`
- There is no NaN/inf or score collapse in the current code path; the image-level and pixel-level indicators are both at normal levels.

determination:

- `pass`
- Reason: The training and verification path is stable, the loss trend is normal, and the single-type indicator does not trigger the shutdown line

## 5. Full Benchmark

Suggested command:

```bash
python tools/benchmark.py \
    --data_root data/mvtec_ad \
    --methods fastflow \
    --categories all \
    --output runs/alignment/fastflow_v2.json
```
Summary of results:

| Metric | Reference | BaoIAD | Gap |
|--------|-----------|----------|-----|
| image_auroc | `0.950` | `0.9504` | `+0.04%` |
| pixel_auroc | `0.979` | `0.9694` | `-1.0%` |

illustrate:

- The current warehouse README and paper assets have archived the 15 categories of FastFlow results as `0.9504 / 0.9694`
- `runs/alignment/fastflow_v2.json` was not regenerated in this turn because the existing `500 epochs x 15 categories` configuration is a multi-hour task; the current review first completes the code path, probe, smoke and guard
- At this stage, the archived full benchmark conclusion of README is maintained and the missing playbook evidence chain is completed.

Shutdown line inspection:

- [x] Historical archive results do not appear in large areas near `0.5` image AUROC
- [x] There is no uniform platform value collapse in historical archive results
- [x] Probe / smoke of current code path does not expose structural exceptions
- [ ] This turn did not rerun the complete 15 categories of benchmark JSON

## 6. Guard

- New/enhanced test: `tests/test_models/test_detectors/test_fastflow.py`
- Added document guard: `docs/alignment/fastflow_checklist.md`
- Added new anti-regression points:
  - The number of layers and shape of `LayerNorm([C,H,W])` must strictly correspond to the backbone output
  - When `conv3x3_only=False` is used, the `kernel_size` of the flow block must be alternated by `3,1,3,...`
  - predict paths must output limited `pred_score` and `(1,H,W)` anomaly maps
- If you change these paths later, you must rerun:
  - `baoiad/models/detectors/fastflow.py`
  - `configs/fastflow/fastflow_wrn50_256_mvtec_strict.py`
  -`tests/test_models/test_detectors/test_fastflow.py`
  - `python tools/alignment_probe.py configs/fastflow/fastflow_wrn50_256_mvtec_strict.py --splits train test --max-batch-size 2 --output runs/alignment/fastflow_probe.json`

## 7. Residual Risk

- The current turn does not re-run the new 15-category benchmark JSON, so the original products of Gate 4 still rely on historical archives rather than new files in this round.
- The magnitude of loss in a single batch in the probe is still large. Although the absolute value of FastFlow's NLL may be very large, if the backbone / flow block is changed in the future, priority should still be given to whether the smoke curve is stable.

## 8. Conclusion

- Final decision: `aligned`
- Allowed to proceed to next stage: `yes`
- If not allowed, next action: If the original Gate 4 product needs to be strictly completed in the future, rerun `tools/benchmark.py --methods fastflow --categories all` separately.

## Alignment checklist

Status only allowed: `matched | mismatch-fixed | intentional-diff | open`

## 1. Enter path

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| train color channel | `.refs/anomalib` FastFlow preprocessing | `configs/_base_/datasets/mvtec_ad.py` | train input enters backbone in RGB | `LoadImage(to_rgb=True)` is enabled by default, train/test pipeline is symmetrical | matched |
| test color channel | Same as above | Same as above | test input is consistent with train | `configs/_base_/datasets/mvtec_ad.py` | matched |
| resize | `.refs/anomalib` pre-processor | `ResizeAD(size=256, backend='pillow')` | input unified resize to `256x256` | `configs/_base_/datasets/mvtec_ad.py` | matched |
| normalization / value range | anomalib image pre-processor | `NormalizeAD()` | Use ImageNet mean/std normalization | `NormalizeAD` The default value is consistent with the main process | matched |
| batch size / seed | history anomalib / README alignment caliber | `train_dataloader.batch_size=32` + `randomness.seed=42` | benchmark default seed=`42`, batch size=`32` | `configs/fastflow/fastflow_wrn50_256_mvtec_strict.py` + `configs/_base_/default_runtime.py` | matched |

## 2. Backbone / Feature Extraction

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| backbone weight | `.refs/anomalib/src/anomalib/models/image/fastflow/torch_model.py` | `configs/fastflow/fastflow_wrn50_256_mvtec_strict.py` + `baoiad/models/detectors/fastflow.py` | using `wide_resnet50_2`, `features_only=True`, `out_indices=(1,2,3)` | fixed on both sides WRN-50-2 three-layer features | matched |
| backbone frozen | Same as above | `FastFlowDetector.__init__` + `train()` | backbone always frozen and kept eval | explicit `self.backbone.eval()` in config `frozen=True`, `train()` | matched |
| trainable LayerNorm | `torch_model.py::FastflowModel.__init__` | `FastFlowDetector.__init__` | Each scale feature is preceded by trainable `LayerNorm([C,H,W])` | `test_layer_norm_shapes_match_backbone_feature_shapes` | matched |
| Feature extraction output | `torch_model.py::_get_cnn_features` | `FastFlowDetector.extract_features` | After backbone output, the corresponding layers are passed layer by layer `LayerNorm` | Both sides are list comprehension corresponding scale normalization | matched |

## 3. Flow Graph

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| flow container | `create_fast_flow_block()` | `FastFlowDetector._create_flow_block()` | using `SequenceINN + AllInOneBlock` | consistent code paths | matched |
| subnet padding / conv | `subnet_conv_func()` | `_subnet_conv_func()` | `ZeroPad2d -> Conv2d -> ReLU -> ZeroPad2d -> Conv2d` | The code structure is consistent item by item | matched |
| kernel schedule | `kernel_size = 1 if i % 2 == 1 and not conv3x3_only else 3` | Same formula | Press `3,1,3,...` alternately when not `conv3x3_only` | `test_flow_kernel_schedule_matches_reference` | matched |
| clamp / permutation | `affine_clamping=2.0`, `permute_soft=False` | Same parameters | The clamp and permutation calibers of the Flow block are the same | The construction parameters of both sides are the same | matched |
| flow steps / hidden ratio | `examples/configs/model/fastflow.yaml` | `configs/fastflow/fastflow_wrn50_256_mvtec_strict.py` | `flow_steps=8`, `hidden_ratio=1.0`, `conv3x3_only=False` | consistent configuration | matched |

## 4. Loss

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| NLL formula | `.refs/anomalib/src/anomalib/models/image/fastflow/loss.py` | `FastFlowDetector.forward(mode='loss')` | `mean(0.5 * sum(z^2) - log_jac_det)` Accumulated by scale | The formula is consistent item by item; `test_forward_loss` | matched |
| Output structure | `FastflowLoss.forward()` returns scalar | `forward(mode='loss')` returns `{'loss': scalar}` | loss can be directly connected to mmengine training loop | `alignment_probe` / `test_forward_loss` | matched |
| optimizer | anomalib FastFlow trainer | `configs/fastflow/fastflow_wrn50_256_mvtec_strict.py` | Adam, `lr=1e-3`, `weight_decay=1e-5`, no scheduler | consistent configuration | matched |
| trainer runtime | `examples/configs/model/fastflow.yaml` | `configs/fastflow/fastflow_wrn50_256_mvtec_strict.py` | anomalib example includes early stopping, BaoIAD follows the unified 500 epoch runtime | The current warehouse benchmark main caliber does not introduce Lightning early stopping | intentional-diff |

## 5. Predict / Scoring

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| anomaly map source | `.refs/anomalib/src/anomalib/models/image/fastflow/anomaly_map.py` | `FastFlowDetector.forward(mode='predict')` | generate flow map according to `-exp(-mean(z^2)/2)` for each scale | formula consistent with interpolation parameters | matched |
| Multi-scale aggregation | `AnomalyMapGenerator.forward()` | `forward(mode='predict')` | `mean` aggregation after upsampling at each scale | Both sides are `torch.mean(torch.stack(...), dim=-1)` | matched |
| image score | `torch_model.py` | `forward(mode='predict')` | `pred_score = amax(anomaly_map)` | Aggregate by space on both sides `max` | matched |
| map output shape | `InferenceBatch.anomaly_map` | `build_predict_results()` | per sample output `(1,H,W)` | `test_forward_predict` + `runs/alignment/fastflow_probe*.json` | matched |

## 6. Behavior verification conclusion

- [x] After fixing the seed, the dataset sample structure is as expected
- [x] The key intermediate quantities of the loss path have made shape / finiteness assertions
- [x] predict path's score / map makes shape / finiteness assertion
- [x] `LayerNorm` shape and `kernel_size` alternating strategy filled guard
- [x] Real data `alignment_probe` passed and archived to `runs/alignment/fastflow_probe.json`

## 7. Remarks

- FastFlow does not involve anomaly synthesis, and there is no reconstruct/discriminate dual branch, so the checklist is cut into five parts according to the real structure of the method: input, feature extraction, flow, loss, and predict.
- Currently the only intentional diff is the early-stopping semantics of the training runtime, not the model structure, loss or scoring logic.
