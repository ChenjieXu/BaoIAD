# RD strict-alignment evidence

- **Method slug**: `rd`
- **Family**: Knowledge distillation
- **Method README**: [`configs/rd/README.md`](../../configs/rd/README.md)

This file preserves the fixed alignment evidence migrated from the manuscript evidence workspace. The content below is translated to English where needed and kept inside this repository so the strict-alignment state can be reviewed without relying on an external paper checkout.

## Config entry points

- [`configs/rd/rd_wrn50_256_mvtec_strict.py`](../../configs/rd/rd_wrn50_256_mvtec_strict.py)
- [`configs/rd/rd_wrn50_256_visa.py`](../../configs/rd/rd_wrn50_256_visa.py)

## Detailed alignment report

**Status**: `playbook-complete`
**Update**: 2026-03-28

## 1. Spec

- **Main Reference**: Official `RD4AD` warehouse, local freeze path `.refs/RD4AD_official`
- **Reference commit**: `6554076872c65f8784f6ece8cfb39ce77e1aee12`
- **Paper**: Deng & Li, "Anomaly Detection via Reverse Distillation from One-Class Embedding", CVPR 2022
- **strict main configuration**: `configs/rd/rd_wrn50_256_mvtec_strict.py`
- **backbone / scoring freeze**:
  - `wide_resnet50_2` torchvision encoder, `layer1/2/3`
  - OCBE + reversed WRN-50-2 decoder
  - flattened cosine loss
  - additive anomaly map
  - `scipy.ndimage.gaussian_filter(sigma=4)`
  - image score = `max(anomaly_map)`
- **runtime freeze**:
  - `seed=111`
  - train batch size = `16`
  - test batch size = `1`
  - optimizer = `Adam(lr=0.005, betas=(0.5, 0.999), weight_decay=0)`
  - max epochs = `200`
- **runtime-only intentional diff**:
  - In order to avoid `num_workers=0` on AFS causing strict full benchmark to degrade to `4h+ / bottle`, the current strict benchmark mainline retains `num_workers=4` and `persistent_workers=True`
  - This difference only affects the dataloader runtime and does not change the model structure, loss, predict or scoring semantics

See [`rd_checklist.md`](rd_checklist.md) for the control matrix.

## 2. Evidence

- **Gate 0/1**
  - The official master reference has been switched back from the local `.refs/anomalib` / `.refs/ader` proxy caliber to `.refs/RD4AD_official`
  - `docs/alignment/rd_checklist.md` has been completed, there is currently no `open` item
- **Gate 2**
  - `runs/alignment/rd_probe.json` Archived
  - probe results show:
    - train batch sample meta information, `gt_mask` structure is normal
    - loss path limited: `loss=1.9882`
    - test predict path limited: `pred_score=2.0833`, `pred_anomaly_map` shape=`(1,256,256)`
- **Gate 3**
  - strict `bottle` smoke Completed: `runs/benchmark/rd/bottle/20260327_145724`
  - `1 epoch` smoke results: `img=0.9048`, `pxl=0.8954`, `image_f1max=0.9153`, `aupro=0.6727`
- **Gate 4**
  - fresh strict `15/15` full benchmark Completed: `runs/alignment/rd_strict_full_benchmark_merged.json`
  - strict `15/15` Average results:
    - `image_auroc = 0.9851`
    - `pixel_auroc = 0.9781`
    - `image_f1max = 0.9759`
    -`image_ap = 0.9939`
    - `aupro = 0.9329`
- **Guard**
  - detector tests: `tests/test_models/test_detectors/test_reverse_distillation.py`
  - benchmark priority / single-class guard: `tests/test_utils/test_benchmark_config_detection.py`

## 3. Repair this round

- Added strict main configuration `configs/rd/rd_wrn50_256_mvtec_strict.py`
  - Detach the mainline from the historical anomalib-aligned config and freeze the official RD4AD caliber separately
  - strict config changed back to `FeatureExtractor` / torchvision `wide_resnet50_2`, no longer relying on `TIMMBackbone(features_only=True)` as the main acceptance path
- `ReverseDistillation` detector explicitly aligns with official `test.py`
  - Added `_flatten_cosine_loss()` to fix the training loss to the official flatten cosine formula
  - Added `_compute_anomaly_map()`, explicitly fixed to layer-by-layer `1 - cosine_similarity` and then aggregated by `add`
  - Added `_smooth_anomaly_map()`, strict main line defaults to Gaussian smoothing of `scipy`
- benchmark smoke path correction
  - `tools/benchmark.py` additionally forces `train_cfg.val_begin=1` under `--epochs` override
  - Otherwise, `val_begin=10` strict config like RD will not produce any verification indicators when doing `1 epoch` smoke
- strict runtime blocker fix
  - On the AFS data path, the official `num_workers=0` will pull the `bottle` full run to about `4h44m` ETA
  - Added `benchmark_keep_dataloader_workers=True` and fixed strict config workers to `4`
  - Comparative evidence:
    - `runs/benchmark/rd/bottle/20260327_162553`: `0 workers`, `Epoch(train)[2] eta ≈ 4:44:25`
    - `runs/benchmark/rd_worker_probe/20260327_163155`: `4 workers`, training restored to approximately `1.0-1.9s/iter`

## 4. Current conclusion

- Currently `RD` Completed Gate 0-5:
  - Reference freeze
  - checklist
  - Behavior probe
  - `bottle` smoke
  - strict `15/15` full benchmark
  - guard/documentation archive
- This round of strict `15/15` is finally merged into `runs/alignment/rd_strict_full_benchmark_merged.json` by the sharding results
- Main category results:
  - `bottle = 1.0000 / 0.9869`
  - `cable = 0.9666 / 0.9729`
  - `grid = 0.9900 / 0.9932`
  - `hazelnut = 1.0000 / 0.9895`
  - `toothbrush = 0.9778 / 0.9911`
  -`transistor = 0.9708 / 0.9250`
- Comparison based on the average `img=0.985 / pxl=0.969` of anomalib proxy retained for a long time in the warehouse:
  - image gap = `+0.01%`
  - pixel gap = `+0.91%`
- The current result does not trigger stop-line and can be formally recorded as `playbook-complete`

## 5. Keep historical results

- Historical anomalib proxy main configuration remains as `configs/rd/rd_wrn50_256_mvtec.py`
- Historical `15/15` proxy results continue to remain as:
  - `image_auroc = 0.9784`
  - `pixel_auroc = 0.9740`
- These numbers are no longer used as the acceptance conclusion of the current strict official main line, but are only used as historical benchmark records.

## Alignment checklist

Status only allowed: `matched | mismatch-fixed | intentional-diff | open`

## 1. Enter path

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| train color channel | `.refs/RD4AD_official/dataset.py::MVTecDataset.__getitem__` | `LoadImage(to_rgb=True)` | train image by RGB input encoder | `docs/alignment/rd.md` + `runs/alignment/rd_probe.json` | matched |
| test color channel | Same as above | `configs/_base_/datasets/mvtec_ad.py` | test and train share RGB path | `LoadImage` default `to_rgb=True` | matched |
| resize / crop | `.refs/RD4AD_official/dataset.py::get_data_transforms()` | `ResizeAD(size=256, backend='pillow')` | Officially `Resize(256) -> CenterCrop(256)`; actually equivalent to `256x256` The input shape in resize | `runs/alignment/rd_probe.json` is fixed to `256x256` | matched |
| normalization / value range | Same as above | `NormalizeAD()` | ImageNet mean/std normalization | The input range in `rd_probe.json` is consistent with ImageNet normalize | matched |
| batch size / seed | `.refs/RD4AD_official/main.py::setup_seed/train()` | `configs/rd/rd_wrn50_256_mvtec_strict.py` | strict mainline frozen as `seed=111`, train `batch=16`, test `batch=1` | strict config + detector config test | matched |
| dataloader workers | Official `DataLoader(...)` Default `num_workers=0` | `benchmark_keep_dataloader_workers=True`, `num_workers=4` | Only relax the runtime, do not change the algorithm semantics | `runs/benchmark/rd/bottle/20260327_162553` vs `runs/benchmark/rd_worker_probe/20260327_163155` shows that `0 workers` ETA under AFS is about `4h44m/bottle`, `4 workers` is restored to about `1.0-1.9s/iter` | intentional-diff |

## 2. Anomaly Synthesis

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| Perlin mask / texture blending | — | — | RD no anomaly synthesis | The method itself does not contain anomaly synthesis branches | intentional-diff |
| clean/anomaly sampling probability | — | — | Same as above | Not applicable | intentional-diff |

## 3. Reconstruct branch

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| teacher encoder | `.refs/RD4AD_official/main.py::wide_resnet50_2(pretrained=True)` | `FeatureExtractor(backbone_name='wide_resnet50_2', out_indices=(1,2,3), frozen=True)` | Using the `layer1/2/3` third-level features of torchvision WRN-50-2, and teacher frozen | `configs/rd/rd_wrn50_256_mvtec_strict.py` | matched |
| OCBE / bottleneck | `.refs/RD4AD_official/de_resnet.py::MFF_OCE` | `baoiad/models/detectors/reverse_distillation.py::OCBE` | The channel, stride, and residual block structures are consistent | This round of strict comparison review, there is currently no `open` item | matched |
| student decoder | `.refs/RD4AD_official/de_resnet.py::de_wide_resnet50_2` | `StudentDecoder` | Reverse WRN-50-2 decoder, return order `[f1, f2, f3]` | Code path item by item comparison | matched |
| loss input | `.refs/RD4AD_official/main.py::loss_fucntion()` | `ReverseDistillation._flatten_cosine_loss()` | Flatten layer by layer into `(B, -1)` and then do cosine loss | `tests/test_models/test_detectors/test_reverse_distillation.py` | matched |

## 4. Discriminate branch

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| discriminate head | — | — | RD no discriminate branch | The method itself is not applicable | intentional-diff |
| skip / logits / number of categories | — | — | Same as above | Not applicable | intentional-diff |

## 5. Loss

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| cosine loss formula | `.refs/RD4AD_official/main.py::loss_fucntion()` | `_flatten_cosine_loss()` | loss is the sum of layer-by-layer flatten cosine dissimilarity | targeted unit test has been completed | matched |
| optimizer | `.refs/RD4AD_official/main.py::Adam(...)` | `optim_wrapper` | `Adam(lr=0.005, betas=(0.5, 0.999), weight_decay=0)` | strict config | matched |
| scheduler | No official scheduler | `param_scheduler = []` | No additional scheduler | strict config | matched |

## 6. Predict / Scoring

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| anomaly map source | `.refs/RD4AD_official/test.py::cal_anomaly_map()` | `_compute_anomaly_map()` | Each layer `1 - cosine_similarity`, upsampled to the input size and then aggregated by `add` | `tests/test_models/test_detectors/test_reverse_distillation.py` | matched |
| Interpolation parameters | Same as above | `F.interpolate(..., align_corners=True)` | Bilinear interpolation and `align_corners=True` | detector code | matched |
| image score aggregation | `.refs/RD4AD_official/test.py::evaluation()` | `pred_score = anomaly_map.max()` | The image score takes the maximum value of the anomaly map space | smoke/single test path is consistent | matched |
| Post-processing / smoothing | `.refs/RD4AD_official/test.py::gaussian_filter(sigma=4)` | `_smooth_anomaly_map(..., smoothing_backend='scipy')` | strict main line `scipy.ndimage.gaussian_filter` | strict config + detector code | matched |
| predict output packaging | official return array directly | `build_predict_results()` | only retain BaoIAD unified interface difference in the output packaging layer | BaoIAD unified `ADDataSample` interface | intentional-diff |

## 7. Behavior verification conclusion

- [x] After fixing the seed, the dataset sample structure is as expected
- [x] mask shape and range are as expected
- [x] The key intermediate quantity of the loss path has a shape / range assertion.
- [x] predict path's score / map makes shape / range assertions
- [x] `bottle` 1-epoch smoke passed and the abnormal shutdown line was not triggered
- [x] strict `15/15` full benchmark completed, stop-line not triggered

## 8. Remarks

- The real data probe has been archived to `runs/alignment/rd_probe.json`.
- strict `bottle` smoke has been archived to `runs/benchmark/rd/bottle/20260327_145724` and currently results in `img=0.9048`, `pxl=0.8954`, `aupro=0.6727`.
- strict `15/15` full benchmark The final merged results can be found in `runs/alignment/rd_strict_full_benchmark_merged.json`.
- The current final means are `img=0.9851`, `pxl=0.9781`, `image_f1max=0.9759`, `aupro=0.9329`.
- `img=0.9784 / pxl=0.9740` in the current README is only retained as historical anomalib proxy results.
