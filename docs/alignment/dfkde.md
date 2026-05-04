# DFKDE strict-alignment evidence

- **Method slug**: `dfkde`
- **Family**: Feature-memory / density
- **Method README**: [`configs/dfkde/README.md`](../../configs/dfkde/README.md)

This file preserves the fixed alignment evidence migrated from the manuscript evidence workspace. The content below is translated to English where needed and kept inside this repository so the strict-alignment state can be reviewed without relying on an external paper checkout.

## Config entry points

- [`configs/dfkde/dfkde_256_mvtec_strict.py`](../../configs/dfkde/dfkde_256_mvtec_strict.py)
- [`configs/dfkde/dfkde_256_visa.py`](../../configs/dfkde/dfkde_256_visa.py)

## Detailed alignment report

**Status**: `playbook-complete`
**Date**: `2026-03-27`

## 1. Reference freezing

- Reference warehouse:
  - Primary reference: local `.refs/anomalib`
- Reference commit:
  - anomalib: `4f6af1acb0ee7b81f54cda036dd9f1c27f63b69a`
- Refer to config/checkpoint:
  - `.refs/anomalib/src/anomalib/models/image/dfkde/torch_model.py`
  - `.refs/anomalib/src/anomalib/models/image/dfkde/lightning_model.py`
  - `.refs/anomalib/src/anomalib/models/components/classification/kde_classifier.py`
  - `.refs/anomalib/src/anomalib/models/components/stats/kde.py`
  - `.refs/anomalib/examples/configs/model/dfkde.yaml`
- Data set/category: MVTec AD, the main target is the 15-category standard benchmark; this round of smoke uses `bottle`
- Input resolution: `256x256`
- seed: `42`
- Indicator definition: image AUROC / pixel AUROC
- intentional diff:
  - BaoIAD reserves `forward(mode='loss') -> {'loss': 0}` for compatibility with MMEngine training loop
  - BaoIAD additionally returns uniform `pred_anomaly_map` in `predict`, which is only used for unified evaluation interface; the official DFKDE ontology only has image-level score

## 2. Code path comparison conclusion

See [`dfkde_checklist.md`](dfkde_checklist.md) for the control matrix.

### Consistency confirmed

- `configs/dfkde/dfkde_256_mvtec.py` is aligned with the anomalib mainline example configurations: `resnet18`, `layer4`, `n_pca_components=16`, `feature_scaling_method='scale'`, `max_training_points=40000`
- `DFKDEDetector` is now executed according to the official path `TIMMBackbone -> GAP -> flatten -> PCA -> scale -> Gaussian KDE -> log density sigmoid`
- Scott bandwidth, `scale` normalization, and `0.05 / 12` image score sigmoid parameters are all consistent with the main reference

### Fixed inconsistencies

- `_KDEClassifier.fit()` used to secretly reduce the PCA dimension to the number of samples when the number of samples was insufficient; now it has been changed to "direct failure without automatic dimensionality reduction" consistent with the anomalib main line.
- `_GaussianKDE.fit()` used to do `1e-6` regular fallback for non-positive definite covariance matrices; it has been removed and restored to the direct `cholesky` path of the official mainline
- `extract_features()` only took `feats[-1]` before; now it is consistent with the official `get_features()`, and all layers returned by backbone are in order `GAP + flatten + concat`
- Empty memory bank was returned silently before; now it is changed to report an error explicitly
- The `tools/alignment_probe.py` entry script has been supplemented, and additional warmup logic in the probe phase has been added to DFKDE to avoid direct interruption of the probe under strict low sample semantics

### Items that are still open

- No algorithm level `open` items

## 3. Behavior Probe

Order:

```bash
python tools/alignment_probe.py configs/dfkde/dfkde_256_mvtec.py \
    --splits train test \
    --max-batch-size 2 \
    --output runs/alignment/dfkde_probe_main.json
```
in conclusion:

- `runs/alignment/dfkde_probe_main.json` has been generated and the structure check passed
- `test` split hits the path of "classifier not yet fit" for the first time; the probe then automatically collects `15` train batches, upgrades the staged memory bank to `32` samples, and then completes KDE fit.
- After warmup, `pred_score` and uniform `pred_anomaly_map` are both limited values, indicating that the main path after strict alignment can forward normally.

Key statistics:

- dataset sample:
  - train preview: `data/mvtec_ad/zipper/train/good/065.png`
  - test preview: `data/mvtec_ad/bottle/test/broken_large/000.png`
  - Input shape: `2 x 3 x 256 x 256`
- loss path:
  - `loss=0.0`
  - finite=`true`
- predict path:
  - `memory_bank_warmup.used = true`
  - `memory_bank_warmup.loss_batches = 15`
  - `memory_bank_warmup.staged_samples = 32`
  - `pred_score mean = 0.4683`
  - `pred_anomaly_map shape = 1 x 256 x 256`

## 4. Small-scale controlled experiment

Experimental setup:

- Category: `bottle`
- Training budget: `1 epoch`
- seed: `42`
- Comparison objects:
  - BaoIAD current main configuration `configs/dfkde/dfkde_256_mvtec.py`
  - Official mainline reference runner `tools/dfkde_official_reference.py`

Order:

```bash
python tools/train.py configs/dfkde/dfkde_256_mvtec.py \
    --work-dir runs/alignment/dfkde_bottle_smoke \
    --cfg-options \
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

- Only embedding collection is done during the training phase, so `loss=0.0000` is expected behavior and does not mean training collapse.
- `MemoryBankHook` successfully calls `build_memory_bank()` after the epoch ends
- `bottle` verification results are `image_auroc=0.9603`, `pixel_auroc=0.7456`
- The official mainline `bottle` single-category reference run has been supplemented:
  - Command: `python tools/dfkde_official_reference.py --categories bottle --batch-size 128 --output runs/alignment/dfkde_official_bottle_reference.json`
  - Final result: `image_auroc=0.9603`
  - The gap with the current BaoIAD `bottle` is `≈0.00%`
  - Process notes: The earliest version of the reference runner misused the local `torchvision resnet18-f37072fd.pth`, resulting in `bottle=0.9294` and class-by-class gaps being artificially high; after switching to `resnet18_a1_0-d63eafa0.pth` corresponding to timm `a1_in1k`, the official single-class results were aligned with BaoIAD

determination:

- `pass`
- Reason: The current implementation has completed the single-class reference-vs-BaoIAD comparison with the official mainline. The build/validate path has been completely run through, and there is no unfit predict, NaN score or abnormal shutdown line.

## 5. Full Benchmark

Suggested command:

```bash
python tools/benchmark.py \
    --data_root data/mvtec_ad \
    --methods dfkde \
    --categories all \
    --output runs/alignment/dfkde_main_full.json
```
Summary of results:

| Metric | Reference | BaoIAD | Gap |
|--------|-----------|----------|-----|
| image_auroc | `0.7462` | `0.7461` | `-0.01%` |
| image_auroc (legacy 92f08e52) | `0.762` | `0.7463` | `-1.6%` |

illustrate:

- `0.7463 vs 0.762` is the old version of anomalib (`92f08e52`) historical benchmark, and the current main reference is only retained as legacy comparison.
- The current main reference uses the official `DfkdeModel` of `.refs/anomalib` main `4f6af1ac...`, and explicitly loads the corresponding `resnet18_a1_0-d63eafa0.pth` of timm `a1_in1k`
- fresh full benchmark products:
  -Official mainline reference: `runs/alignment/dfkde_official_main_reference.json`
  - BaoIAD full benchmark: `runs/alignment/dfkde_main_full.json`
- The maximum absolute difference of class-wise image AUROC is only `0.0011` (`hazelnut`), and the mean difference is `-0.0001`

Shutdown line inspection:

- [x] `probe` and `bottle` smoke do not appear score collapse or NaN
- [x] Official mainline `15/15` fresh reference completed
- [x] BaoIAD `15/15` fresh full benchmark completed
- [x] The current code path level differences have been closed to the checklist
- [x] The full compare with the official mainline did not trigger the shutdown line

## 6. Guard

- New/enhanced tests:
  - `tests/test_models/test_detectors/test_dfkde.py`
  - `tests/test_utils/test_alignment_probe.py`
- Added/restored tool entrance:
  - `tools/alignment_probe.py`
  - `tools/dfkde_official_reference.py`
- Added document guard:
  - `docs/alignment/dfkde_checklist.md`
- If you change these paths later, you must rerun:
  - `baoiad/models/detectors/dfkde.py`
  - `baoiad/utils/alignment_probe.py`
  - `tools/dfkde_official_reference.py`
  - `python tools/alignment_probe.py configs/dfkde/dfkde_256_mvtec.py --splits train test --max-batch-size 2 --output runs/alignment/dfkde_probe_main.json`
  - `python tools/dfkde_official_reference.py --categories bottle --batch-size 128 --output runs/alignment/dfkde_official_bottle_reference.json`
  -`python tools/dfkde_official_reference.py --categories all --batch-size 128 --output runs/alignment/dfkde_official_main_reference.json`
  - `python tools/benchmark.py --data_root data/mvtec_ad --methods dfkde --categories all --output runs/alignment/dfkde_main_full.json --batch_size 128`
  -`pytest tests/test_models/test_detectors/test_dfkde.py tests/test_utils/test_alignment_probe.py -q`

## 7. Residual Risk

- Currently, `predict` still uses the underlying exception path when the classifier is not fit. The probe is responsible for warmup and digging. It is not an explicit human-readable error report.
- DFKDE is still sensitive to sampling and random seeds; if the timm `resnet18` pre-training tag is subsequently replaced, the reference weight caliber must be re-frozen
- uniform `pred_anomaly_map` is just a placeholder for the BaoIAD evaluation interface and should not be misinterpreted as the official DFKDE pixel-level capability

## 8. Conclusion

- Final decision: `playbook-complete`
- Allowed to proceed to next stage: `yes`
- If not allowed, next action: None

## Alignment checklist

Status only allowed: `matched | mismatch-fixed | intentional-diff | open`

## 1. Enter path

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| train color channel | `.refs/anomalib` default pre-processor | `configs/_base_/datasets/mvtec_ad.py` | train input enters backbone in RGB | `LoadImage` default `to_rgb=True`; `runs/alignment/dfkde_probe_main.json` | matched |
| test color channel | Same as above | Same as above | test input maintains the same channel order as train | `runs/alignment/dfkde_probe_main.json` of `inputs.shape=[2,3,256,256]` | matched |
| resize | `.refs/anomalib` MVTec example config | `ResizeAD(size=256)` | input unified resize to `256x256` | `configs/_base_/datasets/mvtec_ad.py` | matched |
| normalization / value range | anomalib image pre-processor | `NormalizeAD()` | Use ImageNet mean/std normalization | probe input statistics are limited | matched |
| seed / batch size | anomalib main caliber `seed=42`, `batch=32` | `randomness.seed=42` + base dataloader `batch=32` | The public main caliber follows the anomalib mainline | `configs/_base_/default_runtime.py` + `configs/_base_/datasets/mvtec_ad.py` | matched |

## 2. Backbone / Feature Extraction

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| backbone weights | `.refs/anomalib/.../dfkde/torch_model.py` | `configs/dfkde/dfkde_256_mvtec.py` | use `resnet18` timm pre-trained weights | `TIMMBackbone(model_name='resnet18', pretrained=True)` | matched |
| Layer selection | `.refs/anomalib/examples/configs/model/dfkde.yaml` | `configs/dfkde/dfkde_256_mvtec.py` | Only take `layer4` | `out_indices=(4,)` | matched |
| GAP + flatten | `.refs/anomalib/.../dfkde/torch_model.py:get_features()` | `DFKDEDetector.extract_features()` | Do each return layer `adaptive_avg_pool2d -> flatten` | `tests/test_models/test_detectors/test_dfkde.py::test_extract_features_concatenates_all_requested_layers` | mismatch-fixed |
| Multi-layer splicing sequence | Same as above | Same as above | Press backbone to return to the order of splicing features | Same as above | matched |

## 3. PCA / KDE

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| PCA dimensions | `.refs/anomalib/examples/configs/model/dfkde.yaml` | `configs/dfkde/dfkde_256_mvtec.py` | `n_pca_components=16` | config consistent | matched |
| Low sample semantics | `.refs/anomalib/.../components/classification/kde_classifier.py` | `_KDEClassifier.fit()` | When the number of samples is less than the PCA dimension, it will fail directly and will not be automatically reduced | `tests/test_models/test_detectors/test_dfkde.py::test_build_memory_bank_does_not_shrink_pca_for_small_sample_count` | mismatch-fixed |
| Feature scaling | Same as above | `_KDEClassifier._pre_process()` | Scaling by maximum L2 norm when using `scale` | Code paths consistent | matched |
| Scott bandwidth | `.refs/anomalib/.../components/stats/kde.py` | `_GaussianKDE.fit()` | Calculate bandwidth according to Scott rule | Code paths are consistent | matched |
| Covariance singularity cover | Official no additional regular fallback | `_GaussianKDE.fit()` | No longer silently add `1e-6` regularity and try again | Remove local fallback this round | mismatch-fixed |
| max training points | `.refs/anomalib/examples/configs/model/dfkde.yaml` | `configs/dfkde/dfkde_256_mvtec.py` | `max_training_points=40000` random subsampling | config consistent with logic | matched |

## 4. Fit / Predict / Scoring

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| empty memory bank | `.refs/anomalib/.../dfkde/torch_model.py:fit()` | `DFKDEDetector.fit()` | Explicit failure on empty memory bank | `tests/test_models/test_detectors/test_dfkde.py::test_build_memory_bank_raises_on_empty_bank` | mismatch-fixed |
| image score | `.refs/anomalib/.../components/classification/kde_classifier.py` | `_KDEClassifier.predict()` | `log density -> sigmoid(0.05, 12)` | The formula is consistent | matched |
| memory bank hook entry | anomalib `MemoryBankMixin.fit()` | `build_memory_bank()` | explicitly trigger KDE fit after training | `MemoryBankHook` log + `runs/alignment/dfkde_bottle_smoke` | matched |
| anomaly map | The official output is only image-level `pred_score` | `forward(mode='predict')` | BaoIAD additionally returns uniform `pred_anomaly_map` to be compatible with the evaluation interface | `build_predict_results()` + probe output | intentional-diff |
| dummy loss | Official training only collects embeddings and does not optimize | `forward(mode='loss') -> {'loss': 0}` | Keep MMEngine loop operational | `runs/alignment/dfkde_probe_main.json` / smoke log | intentional-diff |

## 5. Behavior verification conclusion

- [x] After fixing the seed, the dataset sample structure is as expected
- [x] `train loss` path output finite scalar
- [x] `test predict` path's `pred_score / pred_anomaly_map` all limited
- [x] `alignment_probe` has automatically filled in enough memory bank warmup for DFKDE
- [x] `bottle` smoke does not trigger the abnormal shutdown line

## 6. Remarks

- DFKDE does not have anomaly synthesis, reconstruct branches or discriminate branches, so the checklist only retains the four key paths of input, feature extraction, PCA/KDE, and predict.
- The current playbook main reference has been switched to the `.refs/anomalib` mainline; `0.762` in the README is only retained as the old version benchmark history.
