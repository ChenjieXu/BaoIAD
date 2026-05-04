# AnoVL strict-alignment evidence

- **Method slug**: `anovl`
- **Family**: Vision-language / foundation
- **Method README**: [`configs/anovl/README.md`](../../configs/anovl/README.md)

This file preserves the fixed alignment evidence migrated from the manuscript evidence workspace. The content below is translated to English where needed and kept inside this repository so the strict-alignment state can be reviewed without relying on an external paper checkout.

## Config entry points

- [`configs/anovl/anovl_vitb16plus_240_mvtec_strict.py`](../../configs/anovl/anovl_vitb16plus_240_mvtec_strict.py)
- [`configs/anovl/anovl_vitb16plus_240_visa.py`](../../configs/anovl/anovl_vitb16plus_240_visa.py)

## Detailed alignment report

**Status**: `playbook-complete`
**Date**: `2026-04-06`

## 1. Reference freezing

- Official reference repository: `.refs/AnoVL`
- commit: `3a70bfdaea6baf1eeb140c5de8155b535bd94833`
-Official operating entrance: `test_zero_shot.sh -> vl_test.py`
- backbone / weight:
  - `ViT-B-16-plus-240`
  - `laion400m_e32`
- Input resolution: `240`
- `features_list`: `[3, 6, 9, 12]`
-dataloader:
  - Official eval `batch_size=1`
  - No conventional train dataloader semantics, BaoIAD’s train split is only used for smoke/probe link occupancy
- optimizer/scheduler:
  - Regular training: `none`
  - TTA: `AdamW(lr=1e-3, weight_decay=0.0)`
  - scheduler: `none`
- rounds / early stopping:
  - Regular training: `none`
  - TTA epoch: `5`
  - early stopping: `none`
- loss:
  - `soft_loss = -(pred[0] * log(pred[0])).sum(-1).mean()`
  - `hard_loss = (-mask * log(pred[1:]) - (1-mask) * log(pred[0])).sum(-1).mean()`
  - Total loss: `soft_loss + 0.5 * hard_loss`
- predict path:
  - Image score: Construct official `22-view` augmentation for a single image, encode image/global feature in one batch, do softmax with normal/abnormal text feature, take abnormal prob and average `22` views
  - anomaly map: Take the `V-V` patch token of the last requested layer, do softmax with the text feature after `ln_post + proj`, take the abnormal channel, and do `replicate pad + 3x3 avg_pool + bilinear upsample(align_corners=True)`
- Special agreement:
  - zero-shot, no regular training
  - per-image TTA
  - Multi-category prompt cache
  - The official preprocessing is CLIP normalize; BaoIAD strict configuration remains `ResizeAD + NormalizeAD(ImageNet)`, and the detector is internally mapped to CLIP normalize, which is equivalent to the official value

## 2. Code path comparison conclusion

See [`anovl_checklist.md`](anovl_checklist.md) for the control matrix.

The strict mainline after this round of repairs:

- Added formal strict configuration: `configs/anovl/anovl_vitb16plus_240_mvtec_strict.py`
- `tools/benchmark.py` The default entry point has been switched to the strict file
- `AnoVLDetector` has been aligned to the official `22-view` image score
- `features_list` now actually participates in layer extraction, and takes the last requested `V-V` layer for anomaly map according to official semantics
- image score now takes prompts according to sample categories, and no longer mistakenly reuses the first sample category in the batch
- TTA is compatible with probe / eval calls under `torch.inference_mode()`
- The default prompt build is changed to runtime-device lazy cache to prevent the constructor from running the complete prompt ensemble on the CPU first

## 3. Behavior Probe

Order:

```bash
python tools/alignment_probe.py configs/anovl/anovl_vitb16plus_240_mvtec_strict.py \
    --splits train test \
    --max-batch-size 2 \
    --device cuda \
    --output runs/alignment/anovl_probe.json
```
Output file: `runs/alignment/anovl_probe.json`

result:

- `passed = true`
-train:
  - `loss` unique key is `loss`
  - `loss = 0.0`
  - All finite values
- test:
  - `pred_score = 0.5004306436`
  - `pred_anomaly_map` shape=`[1, 240, 240]`
  - map stats:
    - `min = 0.4706613`
    - `max = 0.4894524`
    - `mean = 0.4768254`
    -`std = 0.0038669`

## 4. Smoke

Execute `1 epoch` bottle smoke according to the `1-5 epoch` range required by the user:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/anovl/anovl_vitb16plus_240_mvtec_strict.py \
    --work-dir runs/alignment/anovl_bottle_smoke \
    --cfg-options \
        train_cfg.max_epochs=1 \
        train_cfg.val_interval=1 \
        train_dataloader.dataset.cls_names="['bottle']" \
        test_dataloader.dataset.cls_names="['bottle']" \
        val_dataloader.dataset.cls_names="['bottle']" \
        train_dataloader.dataset.multi_class=False \
        test_dataloader.dataset.multi_class=False \
        val_dataloader.dataset.multi_class=False
```
Result Documentation/Evidence:

- `runs/alignment/anovl_bottle_smoke/20260402_204902/20260402_204902.log`
- `runs/alignment/anovl_bottle_smoke/epoch_1.pth`

in conclusion:

- Training loss `0.0000` throughout the process, no divergence, no NaN
- val indicator:
  - `ad/image_auroc = 0.9643`
  - `ad/pixel_auroc = 0.9193`
  - `ad/aupro = 0.8042`
- anomaly map sanity:
  - probe sample map is not all zeros or all brights
  - bottle val `pixel_auroc = 0.9193`, indicating that map is not a degenerate constant map

## 5. Full Benchmark

Order:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/benchmark.py \
    --data_root data/mvtec_ad \
    --categories all \
    --methods anovl \
    --output runs/alignment/anovl_e32_strict_benchmark.json
```
Result Documentation/Evidence:

- `runs/alignment/anovl_e32_strict_benchmark.json`
- `runs/benchmark/anovl/all/20260406_091427/20260406_091427.log`

result:

- `image_auroc = 0.9216`
- `pixel_auroc = 0.8971`
- `image_f1max = 0.9303`
- `image_ap = 0.9668`
- `aupro = 0.7746`
- `time = 1327.4s`

Difference relative to paper reference `0.925 / 0.906 / 0.778`:

- image `-0.0034`
- pixel `-0.0089`
- aupro `-0.0034`

in conclusion:

- official `e32` strict has completed the three-step closed loop of probe, bottle smoke, and full benchmark.
- The new authoritative result is based on `runs/alignment/anovl_e32_strict_benchmark.json`
- Historical `e31` / `strict240` results are only retained as archived comparisons and no longer serve as current strict authority

## 6. Guard

- strict configuration:
  - `configs/anovl/anovl_vitb16plus_240_mvtec_strict.py`
- Passed:
  - `pytest tests/test_configs.py -k anovl -q`
  - `python tools/alignment_probe.py ...`
  - `CUDA_VISIBLE_DEVICES=0 python tools/benchmark.py ... --methods anovl`
- Updated and performed directional regression:
  - `pytest tests/test_models/test_detectors/test_anovl.py -k uses_per_sample -q`
- It is still recommended to do a complete catch-up on an idle GPU:
  - `pytest tests/test_models/test_detectors/test_anovl.py -q`

## Alignment checklist

| Checklist | Official reference | BaoIAD strict | Evidence | Status | Processing |
|--------|----------|------------------|------|------|------|
| reference repository | `.refs/AnoVL` `3a70bfda...` | frozen to method report | `.refs/AnoVL/test_zero_shot.sh`, `.refs/AnoVL/vl_test.py` | matched | write `anovl.md` |
| backbone / weight | `ViT-B-16-plus-240` + `laion400m_e32` | `anovl_vitb16plus_240_mvtec_strict.py` fixed to local `e32` weight | `configs/anovl/anovl_vitb16plus_240_mvtec_strict.py` | matched | strict main line cut to `e32` |
| input resolution | `240` | `img_size=240` | `test_zero_shot.sh`, strict config | matched | none |
| batch size | official eval `batch_size=1` | train/val/test fully explicit `batch_size=1` | `vl_test.py`, strict config | matched | none |
| Layer selection / `features_list` | `[3,6,9,12]`, anomaly map actually takes the last requested `V-V` layer | Now explicitly extracts the requested layers, and takes the last `V-V` layer | `vl_test.py`, `baoiad/models/detectors/anovl.py` | fixed | Previously `features_list` basically did not take effect |
| Input preprocessing | `Resize(240) + CenterCrop(240) + CLIP normalize` | `ResizeAD(240) + NormalizeAD(ImageNet)`, then mapped to CLIP normalize by detector | `vl_test.py:get_data_transforms`, detector `_normalize_for_clip` | matched-by-equivalence | Recorded as numerical equivalence implementation |
| loss formula | `soft_loss + 0.5 * hard_loss` | `_tta_loss()` consistent | `vl_test.py:loss_func`, detector `_tta_loss` | matched | none |
| image score | Official `22-view` augmentation, average abnormal prob after one-time batch encoding | Now aligned to official `22-view` batched scoring | `utils.py:aug`, `vl_test.py`, detector `_augment_views/_compute_image_score` | fixed | Previously `6-view` and not in line with official |
| image score multi-category semantics | Each image is calculated according to its own category prompt | Now sample text prompt sample by sample | detector `forward()` | fixed | The first sample category of the batch was mistakenly reused before |
| anomaly map generation | last requested `V-V` layer patch token -> `ln_post + proj` -> softmax abnormal -> replicate pad -> avg_pool(3) -> bilinear upsample(`align_corners=True`) | aligned | `vl_test.py`, detector `_run_tta/_compute_anomaly_map/forward` | fixed | previous upsample `align_corners=False` |
| TTA protocol | per-image `AdamW(lr=1e-3)`, `epoch=5` | aligned | `vl_test.py`, strict config | matched | none |
| Inference-mode compatible | The official script does not do TTA in inference mode | BaoIAD has explicitly exited `torch.inference_mode()` and cloned the ordinary tensor and then reversed | detector `_run_tta` | fixed | Repair probe failure |
| prompt when to build the device | Officially build `model.to(device)` first, then build text prompts | Now changed to runtime-device lazy cache | `vl_test.py`, detector `_get_text_features` | fixed | Previously, the constructor would build prompt on the CPU first |
| Regular training protocol | No regular training | strict config uses `max_epochs=1` only placeholder smoke/train loop | `test_zero_shot.sh`, strict config | matched-by-adaptation | clear in config comments |
| Probe | strict config must be runnable `train + test` | `runs/alignment/anovl_probe.json` `passed=true` | `runs/alignment/anovl_probe.json` | matched | passed |
| bottle smoke | single class gate verification | `img=0.9643`, `pxl=0.9193`, `aupro=0.8042` | `runs/alignment/anovl_bottle_smoke/...log` | matched | passed |
| full benchmark | strict Main line requires official `e32` full results | `img=0.9216`, `pxl=0.8971`, `aupro=0.7746` | `runs/alignment/anovl_e32_strict_benchmark.json` | matched | Completed |
| benchmark default entry | should fall to the strict mainline | `_METHOD_CONFIG_PRIORITY['anovl']` has put strict first | `tools/benchmark.py` | fixed | previously it would fall to the `224` configuration by default |
| Anomalib deviation | Local `anomalib` There is no official AnoVL implementation in the reference tree for comparison | `N/A` | `.refs/anomalib` | n/a | Checklist only refers to the official repository |
