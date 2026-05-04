# DFM strict-alignment evidence

- **Method slug**: `dfm`
- **Family**: Feature-memory / density
- **Method README**: [`configs/dfm/README.md`](../../configs/dfm/README.md)

This file preserves the fixed alignment evidence migrated from the manuscript evidence workspace. The content below is translated to English where needed and kept inside this repository so the strict-alignment state can be reviewed without relying on an external paper checkout.

## Config entry points

- [`configs/dfm/dfm_256_mvtec_strict.py`](../../configs/dfm/dfm_256_mvtec_strict.py)
- [`configs/dfm/dfm_256_visa.py`](../../configs/dfm/dfm_256_visa.py)

## Detailed alignment report

**Status**: `playbook-complete`
**Date**: `2026-04-06`

## 1. Reference freezing

- Reference repository: local `.refs/anomalib` snapshot
- Reference commit: `4f6af1acb0ee7b81f54cda036dd9f1c27f63b69a`
- Refer to config/checkpoint:
  - `.refs/anomalib/examples/configs/model/dfm.yaml`
  - `.refs/anomalib/src/anomalib/models/image/dfm/README.md`
  - `Wide ResNet-50`
  - `layer3`
  - `pooling_kernel_size=4`
  - `pca_level=0.97`
  - `score_type=fre`
- Dataset/Category: `MVTec AD` standard `15` class, single class train/test
- Input resolution: `256x256`
- seed: `42`
- Indicator definition:
  - Primary alignment index: `image AUROC`
  - Main selection index: `mean absolute image gap` relative to published 15 class table
- Current benchmark mainline:
  - Configuration: `configs/dfm/dfm_256_mvtec_strict.py`
  - backbone: `wide_resnet50_2.tv_in1k`
- Current best-repro sidecar:
  - Configuration: `configs/dfm/dfm_256_mvtec.py`
  - backbone: `wide_resnet50_2.tv2_in1k`
- Other archive candidates:
  - `configs/dfm/dfm_256_mvtec_racm.py`
- intentional diff:
  - The current conclusion only covers `score_type='fre'`
  - When `score_type='nll'`, BaoIAD still outputs all zeros `pred_anomaly_map` to adapt to the unified `ADDataSample` interface

## 2. Code path comparison conclusion

See [`dfm_checklist.md`](dfm_checklist.md) for the control matrix.

### Consistency confirmed

- The DFM main implementation is numerically equivalent to the reference path manually copied from the anomalib source code:
  - The maximum absolute error of preprocessing is about `4.8e-7`
  - `layer3` The maximum absolute error of the feature is about `1.1e-4`, and the average is about `1.1e-6`
  - The final `pred_score` and `pred_anomaly_map` on the synthetic batch are exactly the same element by element
- Input path remains `RGB -> Resize(256, pillow/antialias) -> ImageNet Normalize -> PackADInputs`
- PCA/Gaussian fitting, `fre/nll` scoring, `fre` anomaly map and upsampling paths are consistent with the anomalib DFM main implementation
- `MemoryBankHook` will call `build_memory_bank()` after training ends

### Fixed inconsistencies

- The historical feature layer index bug has been fixed from `feats[-1]` to `feats[self._feat_idx]`
- `ResizeAD`'s pillow path has had extra `uint8` quantization removed
- PCA zero-variance degradation with single-sample memory-bank warmup fixed
- Candidate for reprinting the old version `timm`, `timm<0.9` has been added to be compatible with import fallback and WRN-50 local checkpoint fallback

### Current configuration identity conclusion

- There is no strict / non-strict fork in the code implementation itself; the difference mainly comes from the WRN-50 weight tag freezing method
- The default benchmark mainline of the warehouse is now explicitly switched to the `tv_in1k` strict configuration
- `tv2_in1k` remains as a standalone sidecar as it is closer to the published proxy, but no longer represents the default entry

## 3. Behavior Probe

Current mainline probe:

```bash
python tools/alignment_probe.py configs/dfm/dfm_256_mvtec_strict.py \
    --splits train test \
    --max-batch-size 1 \
    --output runs/alignment/dfm_strict_probe.json
```
in conclusion:

- `runs/alignment/dfm_strict_probe.json` passed
- `train/test` The batch, loss, memory-bank warmup, and predict structures are all normal.
- The first call to `predict` will trigger fitted guard first, and then the probe will automatically warmup `1` train batches and execute `build_memory_bank()`

Additional probe:

- `runs/alignment/dfm_reference_probe.json`: `tv2_in1k` best-repro sidecar
- `runs/alignment/dfm_probe_racm_pinned.json`: `racm_in1k` historical archive

## 4. Small-scale controlled experiment

Current mainline smoke:

```bash
python tools/benchmark.py \
    --data_root data/mvtec_ad \
    --methods dfm \
    --config configs/dfm/dfm_256_mvtec_strict.py \
    --categories bottle \
    --output runs/alignment/dfm_strict_bottle_smoke.json \
    --timeout 3600
```
result:

- `runs/alignment/dfm_strict_bottle_smoke.json`
- `image_auroc = 0.9992`
- `pixel_auroc = 0.9444`
- `image_f1max = 0.9921`

determination:

- `pass`
- Reason: The smoke link of strict `tv_in1k` is normal, and the image/pixel result has no collapse.

sidecar archive:

- `runs/alignment/dfm_reference_bottle_smoke.json`: `tv2_in1k`, `image_auroc = 0.9968`, `pixel_auroc = 0.9442`

## 5. Full Benchmark

Current mainline strict archive:

- `runs/alignment/dfm_strict_v1.json`

Summary of results:

| Metric | Published Reference | BaoIAD Strict Mainline | Gap |
|--------|----------------------|--------------------------|-----|
| image_auroc | `0.9430` | `0.9362` | `-0.0068` |
| image_f1 | `0.9500` | `0.9463` | `-0.0037` |
| mean abs image gap | `—` | `0.0152` | `—` |

Class 15 image AUROC:

| Category | Reference | BaoIAD Strict Mainline | Gap |
|----------|-----------|--------------------------|-----|
| bottle | `0.999` | `0.9992` | `+0.0002` |
| cable | `0.969` | `0.9629` | `-0.0061` |
| capsule | `0.924` | `0.9465` | `+0.0225` |
| carpet | `0.855` | `0.8118` | `-0.0432` |
| grid | `0.784` | `0.7678` | `-0.0162` |
| hazelnut | `0.978` | `0.9779` | `-0.0001` |
| leather | `0.997` | `0.9949` | `-0.0021` |
| metal_nut | `0.939` | `0.9379` | `-0.0011` |
| pill | `0.962` | `0.9741` | `+0.0121` |
| screw | `0.873` | `0.7977` | `-0.0753` |
| tile | `0.995` | `0.9820` | `-0.0130` |
| toothbrush | `0.969` | `0.9639` | `-0.0051` |
| transistor | `0.971` | `0.9771` | `+0.0061` |
| wood | `0.975` | `0.9693` | `-0.0057` |
| zipper | `0.961` | `0.9800` | `+0.0190` |
| **Average** | **`0.943`** | **`0.9362`** | **`-0.0068`** |

Shutdown line inspection:

- [x] No large area image AUROC near `0.5` appears
- [x] Multiple categories did not collapse to similar platform values.
- [x] score / pixel There is no obvious abnormal collapse in the result
- [x] strict mainline `15/15` can be archived stably and used as the default entrance to the warehouse

## 5.1 Candidate screening records

| Candidate | Config / Env | Scope | image avg | mean abs image gap | Conclusion |
|-----------|-------------|-------|-----------|--------------------|------|
| `tv_in1k` | `configs/dfm/dfm_256_mvtec_strict.py` | 15/15 | `0.9362` | `0.0152` | Current benchmark strict mainline |
| `tv2_in1k` | `configs/dfm/dfm_256_mvtec.py` | 15/15 | `0.9487` | `0.0083` | best-repro sidecar, retained but no longer the default entry |
| `racm_in1k` | `configs/dfm/dfm_256_mvtec_racm.py` | 15/15 | `0.9458` | `0.0119` | Second best, archived |
| `timm 0.6.13 + wide_resnet50_2 default` | `PYTHONPATH=.cache/dfm_timm063_overlay` + `model_name=wide_resnet50_2` | Sensitive Category 6 | `0.8788` | `0.0303` | Significantly worse, eliminated |

Screening conclusion:

- strict `tv_in1k` is still closer to the published table on only a few classes, and will significantly break `screw / carpet / tile / zipper`
- `timm 0.6.13` The default sensitive 6 category results of `wide_resnet50_2` are consistent with the `tv_in1k` path trend, so it does not enter the full benchmark
- `tv2_in1k`'s `mean abs gap = 0.016867` on sensitive 6 classes is better than current `racm`'s `0.023433`, and continues to lead on the 15 class full benchmark
- However, according to the current warehouse configuration identity rules, the default benchmark mainline is now fixed as strict; `tv2_in1k` is only retained as best-repro auxiliary certificate

## 5.2 Official History Commit Reprint

Key official candidates:

- `92f08e52` `Update DFM results (#674)`
- `0ef8ab1e` `Patch Timm Feature Extractor (#714)`
- Environment caliber: `timm 0.6.13` overlay + `wide_resnet50_2` default name
- Tool entrance: `tools/dfm_official_repro.py`

Completed official reproduction evidence:

- `runs/alignment/dfm_official_92f08e52_grid_summary.json`
- `runs/alignment/dfm_official_92f08e52_carpet_summary.json`
- `runs/alignment/dfm_official_92f08e52_capsule_summary.json`
- `runs/alignment/dfm_official_92f08e52_zipper_summary.json`
- Combined summary: `runs/alignment/dfm_official_92f08e52_bad4_summary.json`
- `runs/alignment/dfm_official_0ef8ab1e_carpet_summary.json`
- `runs/alignment/dfm_official_0ef8ab1e_capsule_summary.json`
- `runs/alignment/dfm_official_0ef8ab1e_grid_summary.json`
- `runs/alignment/dfm_official_0ef8ab1e_zipper_summary.json`
- Combined summary: `runs/alignment/dfm_official_0ef8ab1e_bad4_summary.json`

Official `92f08e52` results on the 4 worst classes:

| Category | Published Reference | Official `92f08e52` | Sidecar `tv2_in1k` |
|----------|----------------------|----------------------|--------------------|
| carpet | `0.855` | `0.8122` | `0.8876` |
| capsule | `0.924` | `0.9485` | `0.9549` |
| grid | `0.784` | `0.7669` | `0.8012` |
| zipper | `0.961` | `0.9800` | `0.9719` |

bad-class gap comparison:

- Official `92f08e52` Category 4 `mean abs gap = 0.02585`
- Official `0ef8ab1e` Category 4 `mean abs gap = 0.02585`
- sidecar `tv2_in1k` Class 4 `mean abs gap = 0.0229`

determination:

- `92f08e52`, which directly corresponds to `Update DFM results (#674)`, and `0ef8ab1e` which follows it, are no closer to published bad classes than sidecar `tv2_in1k`
- This shows that the current remaining deviation is not just a lack of BaoIAD replication. The published table itself is likely to also rely on unfrozen historical environment details.
- Therefore, DFM needs to retain two conclusions at the same time: strict configuration is the default mainline of the warehouse, `tv2_in1k` is the current optimal published-proxy approximate sidecar

## 6. Guard

- New/enhanced tests:
  - `tests/test_models/test_detectors/test_dfm.py`
  - `tests/test_utils/test_dfm_reference_repro.py`
  - `tests/test_utils/test_benchmark_config_detection.py`
- New/enhanced tools:
  - `tools/dfm_reference_repro.py`
- Added new anti-regression points:
  - DFM default entry for `tools/benchmark.py` must return `configs/dfm/dfm_256_mvtec_strict.py`
  - strict master configuration must be explicitly frozen as `wide_resnet50_2.tv_in1k`
  - `tv2_in1k` and `racm_in1k` must be left as separate sidecar/archive configs
  - `timm 0.6.13` overlay must be able to complete `import baoiad`
  - Single-sample memory-bank warmup must not be triggered by PCA zero-variance degradation `IndexError`
  - `tools/dfm_reference_repro.py` must be able to output a summary of failed candidates instead of crashing directly
- Key products:
  -`runs/alignment/dfm_strict_probe.json`
  - `runs/alignment/dfm_strict_bottle_smoke.json`
  - `runs/alignment/dfm_strict_v1.json`
  - `runs/alignment/dfm_reference_probe.json`
  - `runs/alignment/dfm_reference_bottle_smoke.json`
  - `runs/alignment/dfm_reference_v1.json`
  - `runs/alignment/dfm_reference_v1_summary.json`
  - `runs/alignment/dfm_repro_tv2_sensitive.json`
  - `runs/alignment/dfm_repro_tv2_sensitive_summary.json`
- If you change these paths later, you must rerun:
  - `configs/dfm/dfm_256_mvtec_strict.py`
  - `configs/dfm/dfm_256_mvtec.py`
  -`configs/dfm/dfm_256_mvtec_racm.py`
  - `tools/benchmark.py`
  - `baoiad/models/detectors/dfm.py`
  - `baoiad/models/backbones/timm_backbone.py`
  - `tools/dfm_reference_repro.py`
  - `tests/test_models/test_detectors/test_dfm.py`

## 7. Residual Risk

- The published table does not explicitly freeze the WRN-50 weight tag; the current conclusion still partially relies on candidate matrix backcasting
- strict mainline `tv_in1k` is weaker than `tv2_in1k` sidecar; default entry is not the same as best published-proxy approximation
- Official `92f08e52` and `0ef8ab1e` historical commits have been forged into bad-class, but the results are still not better than `tv2_in1k` sidecar; the remaining deviation is likely to come from deeper historical environment/dependency caliber
- `capsule / carpet / grid / zipper` remains the main source of residual bias
- Currently fully aligned conclusions only cover `score_type='fre'`

## 8. Conclusion

- Current benchmark mainline:
  - `configs/dfm/dfm_256_mvtec_strict.py`
  - `wide_resnet50_2.tv_in1k`
- Current best-repro sidecar:
  - `configs/dfm/dfm_256_mvtec.py`
  - `wide_resnet50_2.tv2_in1k`
- Description:
  - The detector logic is numerically equivalent to the reference
  - The default entrance to the warehouse has been switched back to strict, and non-strict `tv2_in1k` is no longer allowed to represent the current mainline.
  - `tv2_in1k` is still the closest single winner of all verified candidates to the published table, and therefore continues to be retained as sidecar evidence

## 9. Closing decision

- **Playbook Status**: `playbook-complete`
- **Closing date**: `2026-04-06`
- **Closing basis**:
  - Code numerically equivalent to the anomalib reference implementation (preprocessing error ~4.8e-7, characterization error ~1.1e-4)
  - Checklist all closed (no `open` items)
  - Probe / bottle smoke / 15/15 full benchmark have been completed
  - Official commit `92f08e52` / `0ef8ab1e` The fork has been completed, and it is confirmed that the remaining deviations come from the weight tag rather than the code path
  - All 8 tests passed
- **Known caveat**:
  - strict `tv_in1k` mainline is weaker than `tv2_in1k` sidecar (mean image gap: strict 0.9362 vs sidecar 0.9487)
  - The published table does not explicitly freeze the WRN-50 weight tag
  - `capsule / carpet / grid / zipper` remains the main source of residual bias
  - Fully aligned conclusion only covers `score_type='fre'`

## Alignment checklist

Status only allowed: `matched | mismatch-fixed | intentional-diff | open`

## 1. Enter path

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| train/test color channel | `.refs/anomalib` Default pre-processor + MVTec AD datamodule | `configs/_base_/datasets/mvtec_ad.py` + `baoiad/datasets/transforms/loading.py` | Both training and testing use RGB input | `LoadImage(to_rgb=True)`, train/test pipeline consistent | `matched` |
| resize / interpolation | anomalib default `Resize(256)`, old reports frozen to torchvision/PIL antialias paths | `configs/_base_/datasets/mvtec_ad.py` + `baoiad/datasets/transforms/augmentation.py` | `256x256`, PIL/torchvision resize, avoid extra quantization | `ResizeAD(backend='pillow', antialias=True)` | `matched` |
| normalization / value range | anomalib default ImageNet normalization | `configs/_base_/datasets/mvtec_ad.py` + `baoiad/datasets/transforms/augmentation.py` | ImageNet mean/std normalization, input remains float32 | `NormalizeAD` uses `(123.675,116.28,103.53)/(58.395,57.12,57.375)` | `matched` |

## 2. Backbone / Feature Extraction

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| backbone type | `.refs/anomalib/src/anomalib/models/image/dfm/torch_model.py` | `configs/dfm/dfm_256_mvtec_strict.py` | `wide_resnet50_2` pre-training backbone | strict mainline explicitly frozen to `TIMMBackbone(model_name='wide_resnet50_2.tv_in1k', pretrained=True)`; `tv2_in1k` and `racm_in1k` continue to remain independent sidecar/archive config | `matched` |
| layer selection | `.refs/anomalib/src/anomalib/models/image/dfm/torch_model.py` | `configs/dfm/dfm_256_mvtec_strict.py` + `baoiad/models/detectors/dfm.py` | extract only `layer3` | strict config freeze `layer='layer3'`; TIMM path inject `out_indices=(3,)` | `matched` |
| Feature layer index | `.refs/anomalib/src/anomalib/models/image/dfm/torch_model.py:get_features()` | `baoiad/models/detectors/dfm.py:_extract_features()` | Features are taken according to the request layer, and return to the "last layer" is not allowed | The historical bug has been revised from `feats[-1]` to `feats[self._feat_idx]`, and single test coverage has been added | `mismatch-fixed` |
| memory-bank life cycle | `.refs/anomalib/src/anomalib/models/image/dfm/lightning_model.py` + `fit()` | `baoiad/models/detectors/dfm.py` + `baoiad/engine/hooks/memory_bank_hook.py` | train collects features and explicitly fits PCA/Gaussian after training | `build_memory_bank()` has been connected to `MemoryBankHook`, and added fitted guard before predict | `mismatch-fixed` |

## 3. Pooling / Flatten / PCA

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| pooling | `.refs/anomalib/examples/configs/model/dfm.yaml` + `torch_model.py` | `configs/dfm/dfm_256_mvtec_strict.py` + `baoiad/models/detectors/dfm.py` | WRN-50 caliber uses `pooling_kernel_size=4` | strict config and detector are the same by default | `matched` |
| flatten method | `.refs/anomalib/src/anomalib/models/image/dfm/torch_model.py:get_features()` | `baoiad/models/detectors/dfm.py:_extract_features()` | After pooling, batch flatten to 2D feature matrix | `out.view(B, -1)` consistent with reference | `matched` |
| PCA component rules | `.refs/anomalib/examples/configs/model/dfm.yaml` + PCA component | `configs/dfm/dfm_256_mvtec_strict.py` + `baoiad/models/detectors/dfm.py:_PCA` | `pca_level=0.97`, select principal components based on cumulative variance | strict main configuration freeze `pca_level=0.97`; fit/transform/inverse_transform path exists | `matched` |
| Gaussian fitting | `.refs/anomalib/src/anomalib/models/image/dfm/torch_model.py:fit()` | `baoiad/models/detectors/dfm.py:fit()` | Fitting a single class of Gaussians after PCA when `score_type='nll'` | The code paths are consistent; the current alignment benchmark does not use `nll` | `matched` |

## 4. Predict / Scoring

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| image score | `.refs/anomalib/src/anomalib/models/image/dfm/torch_model.py:score()` | `baoiad/models/detectors/dfm.py:_score()` | `fre` uses the sum of reconstruction errors, `nll` uses Gaussian NLL | both score branches are isomorphic to the reference | `matched` |
| anomaly map source | `.refs/anomalib/src/anomalib/models/image/dfm/torch_model.py:score()` | `baoiad/models/detectors/dfm.py:_score()` | `fre` generated from the feature reconstruction error map after channel summation | `torch.sum(fre, dim=1)` after `unsqueeze(1)` | `matched` |
| Upsample to input resolution | `.refs/anomalib/src/anomalib/models/image/dfm/torch_model.py:forward()` | `baoiad/models/detectors/dfm.py:forward()` | Bilinear upsample to original image size | `F.interpolate(..., mode='bilinear', align_corners=False)` | `matched` |
| `nll` map output | anomalib does not return anomaly map under `nll` | `baoiad/models/detectors/dfm.py:forward()` | BaoIAD prediction result structure needs to be uniformly exposed `pred_anomaly_map` | `nll` is filled with all-zero map to facilitate unified evaluation/visualization interface; the current DFM alignment caliber is frozen as `score_type='fre'` | `intentional-diff` |

## 5. Behavior verification conclusion

- [x] After fixing the seed, the dataset sample structure is as expected
- [x] The input, label, mask shape and value range of train/test batch have been recorded through probe
- [x] `loss` path only collects memory-bank and returns limited dummy loss
- [x] The score / map of `predict` after `build_memory_bank()` is a finite value
- [x] There is no `open` item in the current checklist
- [x] The main configuration has explicitly pinned the WRN-50 weight tag used by the current benchmark.
