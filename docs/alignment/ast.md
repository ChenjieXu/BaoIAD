# AST strict-alignment evidence

- **Method slug**: `ast`
- **Family**: Knowledge distillation
- **Method README**: [`configs/ast/README.md`](../../configs/ast/README.md)

This file preserves the fixed alignment evidence migrated from the manuscript evidence workspace. The content below is translated to English where needed and kept inside this repository so the strict-alignment state can be reviewed without relying on an external paper checkout.

## Config entry points

- [`configs/ast/ast_effnet_b5_768_mvtec_strict.py`](../../configs/ast/ast_effnet_b5_768_mvtec_strict.py)
- [`configs/ast/ast_effnet_b5_768_visa.py`](../../configs/ast/ast_effnet_b5_768_visa.py)

## Detailed alignment report

**Status**: `aligned`
**Date**: `2026-03-28`

## 1. Reference freezing

- Reference warehouse: `https://gh-proxy.com/https://github.com/marco-rudolph/AST`
- Reference commit: `8c243ad9adac68e874f87edc6618aa5ea2827228`
- Refer to config/checkpoint:
  - Official `config.py`
  - Official `model.py`
  - Official `train_teacher.py`
  - Official `train_student.py`
  - Official `eval.py`
- Data set/category: `MVTec AD`, RGB, single-class training, 15-class standard benchmark
- Input resolution: `768x768`
- seed: `42`
- Indicator definition:
  - image AUROC main caliber = `mean over maps` (official RGB-only caliber)
  - Auxiliary image AUROC = `max over maps`
  - pixel AUROC = official `depth_len = img_len // 4 = 192` caliber
- intentional diff:
  - At this stage, `timm tf_efficientnet_b5` is still used to extract `block-35` features; it is confirmed that the output shape is consistent with the official configuration as `304x24x24`
  - Still using mmengine training/validation loop with `build_predict_results()` encapsulating prediction output

## 2. Code path comparison conclusion

See [`ast_checklist.md`](ast_checklist.md) for the control matrix.

### Consistency confirmed

- strict main configuration has been fixed to `configs/ast/ast_effnet_b5_768_mvtec_strict.py`
- The main line of AST has been switched to the official two-stage paradigm: `teacher NF -> student CNN`
- strict lines have been filled `pos_enc=True`, `pos_enc_dim=32`
- teacher loss has been changed to the official Jacobian caliber by spatial position, and it no longer incorrectly sums in the spatial dimension first.
- predict has output both `pred_score_mean` and `pred_score_max`, and the strict RGB main line is currently corrected to `pred_score=mean`
- strict evaluator fixed `resize_mask=192`

### Fixed inconsistencies

- The old AST entry is only repaired at `img_size=768`, but the train/test pipeline is still locked at `256` by base config
- The old AST only does joint loss, there is no official `teacher -> student` two-stage
- Old AST teacher Jacobian is a per-sample scalar, not an official `HxW` graph
- The `tools/alignment_probe.py` entry has been added, no longer only the README command without scripts
- `val_evaluator / test_evaluator` of strict config has been changed to mmengine compatible metric list form, teacher/student smoke can enter the verification loop normally
- strict config has explicitly declared `benchmark_multi_class = False` to prevent benchmark runner from mistaking AST as multi-class method
- The benchmark default entry has explicitly prioritized the strict AST configuration

### Items that are still open

- No algorithm level `open` items
- The current remaining gaps are concentrated at the image-level of `toothbrush`, but the full mean has fallen into the acceptable alignment range

## 3. Current implementation entry

- strict config: `configs/ast/ast_effnet_b5_768_mvtec_strict.py`
- two-stage train script: `tools/train_ast.py`
- probe script: `tools/alignment_probe.py`

Completed Gate 2 / Gate 3 products:

```bash
CUDA_VISIBLE_DEVICES=1 python tools/alignment_probe.py configs/ast/ast_effnet_b5_768_mvtec_strict.py \
    --splits train test \
    --max-batch-size 2 \
    --device cuda \
    --output runs/alignment/ast_probe_strict.json \
    --cfg-options \
        "train_dataloader.dataset.cls_names=['bottle']" \
        train_dataloader.dataset.multi_class=False \
        "test_dataloader.dataset.cls_names=['bottle']" \
        test_dataloader.dataset.multi_class=False \
        "val_dataloader.dataset.cls_names=['bottle']" \
        val_dataloader.dataset.multi_class=False
```

```bash
CUDA_VISIBLE_DEVICES=2 python tools/train_ast.py configs/ast/ast_effnet_b5_768_mvtec_strict.py \
    --work-dir runs/alignment/ast_bottle_smoke_v2 \
    --cfg-options \
        train_cfg.max_epochs=1 \
        train_cfg.val_interval=1 \
        train_dataloader.batch_size=1 \
        test_dataloader.batch_size=1 \
        val_dataloader.batch_size=1 \
        "train_dataloader.dataset.cls_names=['bottle']" \
        train_dataloader.dataset.multi_class=False \
        "test_dataloader.dataset.cls_names=['bottle']" \
        test_dataloader.dataset.multi_class=False \
        "val_dataloader.dataset.cls_names=['bottle']" \
        val_dataloader.dataset.multi_class=False
```

```bash
CUDA_VISIBLE_DEVICES=3 python tools/benchmark.py \
    --data_root data/mvtec_ad \
    --methods ast \
    --config configs/ast/ast_effnet_b5_768_mvtec_strict.py \
    --categories all \
    --timeout 10800 \
    --work-dir-root runs/alignment/ast_strict_full_workdirs_v2 \
    --output runs/alignment/ast_strict_full_benchmark_v2.json
```

## 4. Behavior Probe
in conclusion:

- `pass`
- Archive: `runs/alignment/ast_probe_strict.json`
- The current probe runs with the `bottle` single-class strict caliber, and both `train/test` paths pass

Key statistics:

- train input: `2 x 3 x 768 x 768`
- train `loss_student=53.7622`, limited
- test `pred_score` limited, `mean=321.5356`
- test `pred_anomaly_map` shape=`[1, 192, 192]`

## 5. Small-scale controlled experiment

Experimental setup:

- Category: `bottle`
- Training budget: `teacher 1 epoch + student 1 epoch`
- seed: `42`
- Path:
  - teacher: `runs/alignment/ast_bottle_smoke_v2/teacher`
  - student: `runs/alignment/ast_bottle_smoke_v2/student`

observe:

- teacher stage train loss continues to decrease from `709.1138 -> 623.8500`
- teacher `bottle` Verification:
  - `image_auroc=0.8952`
  - `image_auroc_mean=0.9810`
  - `pixel_auroc=0.9408`
- student stage train loss continues to decrease from `0.4200 -> 0.1969`
- student `bottle` verification:
  - `image_auroc(mean)=0.9952`
  - `image_auroc(max)=0.9786`
  - `image_auroc_mean=0.9952`
  -`pixel_auroc=0.9708`
- Compared with teachers, students have significantly improved in `image_auroc(mean/max) / pixel_auroc`, and there is no image score collapse, pure noise map or unified platform value.

determination:

- `pass`
- Reason: The loss curves of both stages decline normally, student verification is better than teacher, and the current strict main line can enter the full benchmark.

## 6. Full Benchmark

Final archive:

- corrected full JSON: `runs/alignment/ast_strict_corrected_full_v4.json`
- Intermediate full JSON (old `max` main caliber, retains only process evidence): `runs/alignment/ast_strict_full_benchmark_v2.json`
- `zipper` Single category make-up run: `runs/alignment/ast_strict_zipper_v3.json`
- work dir root:`runs/alignment/ast_strict_full_workdirs_v2`

illustrate:

- Benchmark has previously corrected AST strict config's `benchmark_multi_class=False` to ensure that it is executed according to the single-class `15/15` path
- `2026-03-28` further confirms that the official RGB-only main image score should be `mean over maps`, and strict config has been synchronously corrected to `image_score_mode='mean'`
- In the completed `14` class, you can directly use `image_auroc_mean` in the old full benchmark JSON as corrected image AUROC; `wood` reads `image_auroc_mean=1.0000` from its strict student log; finally merge it into fresh `zipper` strict run to get corrected `15/15`
- The `image_auroc` column of the completed category in the old `runs/alignment/ast_strict_full_benchmark_v2.json` is still the `max` caliber before splitting. It can only be used as an intermediate product and cannot be directly used as the final strict archive.

Summary of results:

| Metric | Reference | BaoIAD strict corrected | Gap |
|--------|-----------|---------------------------|-----|
| image_auroc | `0.9920` | `0.9858` | `-0.62%` |
| pixel_auroc | — | `0.9549` | — |

Current weakest image classes:

- `toothbrush = 0.9250`
- `pill = 0.9727`
- `grid = 0.9791`
- `transistor = 0.9808`
- `cable = 0.9835`

Shutdown line inspection:

- [x] No large area image AUROC near `0.5` appears
- [x] Multiple categories did not collapse to similar platform values.
- [x] score histogram and smoke observation showed no obvious abnormal contraction.
- [x] The difference from the official reference mean is within the acceptable range

## 7. Guard

- `tests/test_models/test_detectors/test_ast.py`
- `tests/test_structures/test_ad_data_sample.py`
- `tests/test_evaluation/test_ad_metric.py`
- `tests/test_utils/test_benchmark_config_detection.py`

## 8. Conclusion

- Final decision: `playbook-complete`
- Allowed to proceed to next stage: `yes`
- Current conclusions:
  - AST strict official alignment completed
  - The current main archive results are subject to `runs/alignment/ast_strict_corrected_full_v4.json`
  - Historical `256` / `max-score` paths are retained as legacy process evidence only

## Alignment checklist

Status only allowed: `matched | mismatch-fixed | intentional-diff | open`

## 1. Enter path

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| train color channel | official `utils.py` / `Image.open(...).convert('RGB')` | `LoadImage` | RGB input | strict config + dataset pipeline | `matched` |
| test color channel | Same as above | Same as above | RGB input | strict config + dataset pipeline | `matched` |
| resize / crop | official `transforms.Resize((768,768))` | `ResizeAD(size=768)` | `768x768` | `configs/ast/ast_effnet_b5_768_mvtec_strict.py` | `matched` |
| normalization / value range | official `Normalize(mean,std)` | `NormalizeAD` | ImageNet normalize | strict config | `matched` |

## 2. Feature path

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| backbone | official `efficientnet-b5` | `tf_efficientnet_b5` | block-35 feature shape = `304x24x24` | local shape check | `intentional-diff` |
| extract layer | Official `extract_layer=35` | `extract_layer=35` | Get the 35th MBConv block output | strict config | `matched` |
| map len | official `map_len=24` | `map_len=24` | feature map = `24x24` | strict config | `matched` |

## 3. Teacher / Student

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| teacher NF | official `get_nf()` | `TeacherNF` | permutation + conditional coupling | `baoiad/models/detectors/ast.py` | `matched` |
| positional encoding | official `pos_enc=True`, `pos_enc_dim=32` | `positional_encoding_2d()` | as teacher cond and student input concat | detector implementation | `matched` |
| student CNN | official `Student` | `StudentCNN` | `conv1 -> residual blocks -> conv2` | detector implementation | `matched` |
| two-stage training | official `train_teacher.py` / `train_student.py` | `tools/train_ast.py` | first teacher then student | train script | `matched` |

## 4. Loss

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| teacher NLL | official `get_nf_loss()` | `_teacher_loss_map()` | `0.5*sum(z^2)-jac`, `jac` is `HxW` | detector implementation | `mismatch-fixed` |
| student loss | official `get_st_loss()` | `_student_loss_map()` | channel mean MSE map | detector implementation | `matched` |
| stage freeze | Official teacher/student separate training | `training_phase` | Teacher stage does not train students; student stage freezes teacher | detector tests | `matched` |

## 5. Predict / Scoring

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| teacher image score | official `mean_st / max_st` | `pred_score_mean / pred_score_max` | simultaneously output mean/max | detector implementation + tests | `matched` |
| image score main caliber | official RGB-only `mean over maps` | `pred_score` | strict benchmark uses `mean` | strict config / metric | `mismatch-fixed` |
| pixel map size | official `depth_len=img_len//4=192` | `score_map_size=192` + evaluator `resize_mask=192` | pixel indicator is calculated as `192x192` | strict config | `mismatch-fixed` |
| post-processing | official bicubic upsample | `_build_results()` | `bicubic` upsampling | detector implementation | `matched` |

## 6. Behavior verification conclusion

- [x] `alignment_probe` passed
- [x] `bottle` Two-stage smoke passed
- [x] `15/15` strict benchmark archived
- [x] AST strict config loadable
- [x] AST single test has covered `teacher/joint/student` three main paths
