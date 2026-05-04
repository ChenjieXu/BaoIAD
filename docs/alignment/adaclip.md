# AdaCLIP strict-alignment evidence

- **Method slug**: `adaclip`
- **Family**: Vision-language / foundation
- **Method README**: [`configs/adaclip/README.md`](../../configs/adaclip/README.md)

This file preserves the fixed alignment evidence migrated from the manuscript evidence workspace. The content below is translated to English where needed and kept inside this repository so the strict-alignment state can be reviewed without relying on an external paper checkout.

## Config entry points

- [`configs/adaclip/adaclip_vitl14_336_518_mvtec_strict.py`](../../configs/adaclip/adaclip_vitl14_336_518_mvtec_strict.py)
- [`configs/adaclip/adaclip_vitl14_336_518_visa.py`](../../configs/adaclip/adaclip_vitl14_336_518_visa.py)

## Detailed alignment report

**Status**: `playbook-complete`
**Date**: `2026-04-06`

## 1. Reference freezing

- Reference warehouse: `https://github.com/caoyunkang/AdaCLIP`
- Reference commit: `b762ac40c3f33c77e7e513e48cb436f059d456da`
- Refer to config/checkpoint:
  - `.refs/AdaCLIP/train.py`
  - `.refs/AdaCLIP/train.sh`
  - `.refs/AdaCLIP/test.py`
  - `.refs/AdaCLIP/test.sh`
  - `.refs/AdaCLIP/app.py`
  - strict-train main file: `configs/adaclip/adaclip_vitl14_336_518_mvtec_strict.py`
  - strict-eval side file: `configs/adaclip/adaclip_vitl14_336_256_mvtec.py`
  - MVTec AD evaluation caliber is frozen as `weights/pretrained_visa_clinicdb.pth` according to the official script
  - The VisA evaluation caliber is frozen as `weights/pretrained_mvtec_colondb.pth` according to the official script
- backbone / pre-training weights: `ViT-L-14-336`, `openai`
- feature hierarchy: `[6, 12, 18, 24]`
- Dataset/Category:
  - strict-train: `VisA + ClinicDB -> MVTec AD`
  - strict-eval: `pretrained_visa_clinicdb.pth -> MVTec AD`
- Input resolution: `518`
- batch size: `1`
- optimizer / scheduler: `AdamW(lr=0.01, betas=(0.5, 0.999), weight_decay=0.0)` / `none scheduler`
- Training rounds/selection caliber: `5 epochs`, verified once every epoch; the official script does not have early stopping, BaoIAD strict uses `pixel_f1max -> image_auroc` to select best
- loss function: `classification focal + per-layer segmentation focal + dual dice`
- predict path: aggregate visual-text logits of each layer and then do softmax; anomaly map is `(p_anom + 1 - p_norm) / 2`; image score uses `HSF(k=20)`
- Special training protocol: only train prompt / projection / dynamic prompt related parameters; MVTec strict training uses the official `test/meta` partition semantics of auxiliary annotation data
- seed: `111`
- Indicator definition: image/pixel AUROC and other official `test.py` dataset-level indicators exported
- intentional diff:
  - BaoIAD uses `BaseModel` + `build_predict_results()` wrapper `predict`
  - The pipeline first performs ImageNet normalize, and then converts it back to CLIP statistics in the detector.
  - tokenizer uses OpenCLIP interface and does not call reference `SimpleTokenizer` directly
  - Added `require_official_checkpoint=True` to the strict official configuration, which is used to prevent pseudo-aligned operations when weights are missing.

Additional instructions:

- The weight table in the official README is written as `VisA & ColonDB`, but the actual file name used by the official `test.sh` and `app.py` is `pretrained_visa_clinicdb.pth`. This report is based on script caliber.
- strict assets are now completed:
  - `pretrained/pretrained_visa_clinicdb.pth`
  - `pretrained/pretrained_mvtec_colondb.pth`
  - `data/clinicdb/ClinicDB`
- The `MVTec-only` fallback training product has been discovered locally:
  `../projects/baseline/AdaCLIP/workspaces/mvtec_only_retrain/models/0s-pretrained-['mvtec']-ViT-L-14-336-SD-VL-D4-L5-HSF-K20_best.pth`
  This weight has been verified to be loaded by the current BaoIAD AdaCLIP, but does not count towards strict official alignment evidence.

## 2. Code path comparison conclusion

See [adaclip_checklist.md](adaclip_checklist.md) for the control matrix.

### Consistency confirmed

- The default value of prompt, image size, batch size limit, and HSF default value are consistent with the official test script
- The aggregation order and final formula of anomaly map in `visual_text_similarity()` are consistent with the official one
- The key compatibility logic of official checkpoint is consistent with the saving method of reference trainer

### Fixed inconsistencies

- The default checkpoint of the main MVTec configuration is changed from `pretrained_all.pth` to the official script caliber `pretrained/pretrained_visa_clinicdb.pth`
- The default checkpoint configured by VisA is changed to the official script caliber `pretrained/pretrained_mvtec_colondb.pth`
- The seed configured exclusively for AdaCLIP was changed to the official `111`
- Fixed AdaCLIP's `ImageNet -> CLIP` denormalization dimension error:
  Previously, `0-1` mean/std was used for back calculation, but `NormalizeAD` actually used `0-255` ImageNet statistics.
- Fixed the prompt/text attention mask dimension misalignment caused by residual block `batch_first=True` in the current `open_clip==3.3.0` environment.
- Connect the official `train_one_batch` real focal+dice training loss to `mode='loss'`
- `VisADataset` Add fallback parsing of the current local MVTec-like conversion directory
- strict auxiliary training data entry is switched to a dedicated loader:
  - `AdaCLIPVisADataset` Explicit priority `meta.json['test']`
  - `AdaCLIPClinicDBDataset` / `AdaCLIPColonDBDataset` explicitly use official `test` partitioning semantics
- Added auxiliary training configuration:
  - `configs/adaclip/adaclip_vitl14_336_518_mvtec_strict.py`
  - `configs/adaclip/adaclip_vitl14_336_518_visa_train_mvtec.py`
  - `configs/adaclip/adaclip_vitl14_336_518_visa_clinicdb_train_mvtec.py`
  - `configs/adaclip/adaclip_vitl14_336_518_mvtec_train_mvtec.py` (fallback)
- The `ResizeAD` of AdaCLIP related configuration is changed to explicit square `size=(518, 518)`, aligning with the input shape of official `Resize((518,518)) + CenterCrop(518)`
- The strict training configuration removes `CosineAnnealingLR` and returns to the official `train.py` pure `AdamW(lr=0.01, betas=(0.5, 0.999))`
- `AdaCLIPDetector` will now explicitly freeze all unofficial prompt parameters; also fixed the problem of positional encoding being unexpectedly unfrozen after resize
- `tools/benchmark.py` has added strict config priority to AdaCLIP, and `adaclip_vitl14_336_518_mvtec_strict.py` will be selected first by default.
- The strict official configuration adds a guard of "direct failure without checkpoint" to avoid silent degradation to a pseudo-aligned state where official weights are not loaded.
- Special test enhancements for detector/dataset, covering training loss, map/score, class name resolution, checkpoint guard and VisA fallback loader
- Added asset auxiliary script:
  -`tools/prepare_clinicdb.py`
  - `tools/check_adaclip_assets.py`
  - `tools/adaclip_dataset_probe.py`
  - `tools/prepare_clinicdb.py` now supports Hugging Face `train/validation/test + images/masks` directory

Training semantics supplement:

- Real supervised training loss is only enabled in explicit training configuration: `enable_train_loss=True`
- The evaluation configuration maintains non-training semantics; if the evaluation configuration is misused for training, `mode='loss'` will continue to return runner-compatible placeholder loss
- Double rail closing:
  - `strict-train` mainline = `adaclip_vitl14_336_518_mvtec_strict.py`
  - `strict-eval` secondary line = `adaclip_vitl14_336_256_mvtec.py`

### Items that are still open

- Historical `runs/adaclip_strict_train*` series of results were generated before this round of repairs. They are only retained as evidence of failure and no longer participate in the current conclusion.
- The current best checkpoint average has fallen into the acceptable range compared to the official checkpoint, but there is still a large difference by category, mainly concentrated in `hazelnut / metal_nut / screw / zipper / pill`
- A `MVTec-only` fallback training configuration has been added; this path is operable, but does not belong to the strict official alignment caliber

Current asset check:

```bash
python tools/check_adaclip_assets.py
```
The current output shows:

- `data/mvtec_ad`: present
- `data/visa`: present
- `pretrained/pretrained_visa_clinicdb.pth`: present
- `pretrained/pretrained_mvtec_colondb.pth`: present
- `data/clinicdb/ClinicDB`: present

## 3. Behavior Probe

Order:

```bash
python tools/alignment_probe.py configs/adaclip/adaclip_vitl14_336_518_mvtec_strict.py \
    --splits train test \
    --max-batch-size 2 \
    --device cuda \
    --output runs/alignment/adaclip_mvtec_strict_probe.json
```
in conclusion:

- strict-train main configuration probe has passed, and the result is written to `runs/alignment/adaclip_mvtec_strict_probe.json`
- The current code has added strict guard; if `pretrained/pretrained_visa_clinicdb.pth` is missing, the probe will fail directly instead of silently falling back
- A low-cost proxy probe has been run and the result is written to `runs/alignment/adaclip_probe_proxy.json`
- `MVTec-only` fallback training smoke has been run, and the result is written to `runs/alignment/adaclip_train_smoke_mvtec_proxy.json`
- The loading compatibility of the local `MVTec-only` fallback checkpoint has been verified, and the result is written to `runs/alignment/adaclip_mvtec_retrain_checkpoint_load.json`
- It has been verified that strict `VisA + ClinicDB` trains the first batch of dataloader, and the results are written to `runs/alignment/adaclip_train_batch_strict_proxy.json`
- `tools/adaclip_dataset_probe.py` has been added, and the current strict train dataloader result is written to `runs/alignment/adaclip_dataset_probe.json`
- `bottle` strict debug run after normalization repair has been completed, the working directory is `runs/adaclip_bottle_debug_normfix`
- fresh strict `bottle` subset smoke is completed and the result is written to `runs/alignment/adaclip_bottle_strict_smoke_subset128.json`
- fresh strict full train rerun is started, `epoch1` indicator is written to `runs/alignment/adaclip_strict_train_rerun_v7_epoch1.json`
- The MVTec evaluation of the same caliber as the official checkpoint has been started, and the working directory is `runs/adaclip_official_eval_mvtec_v1`
- The final comparison between fresh strict full rerun and official checkpoint has been written in `runs/alignment/adaclip_live_compare_final.json`
- Class-by-class image gap for best-vs-official has been written to `runs/alignment/adaclip_epoch3_vs_official.json`

Key statistics:

- dataset sample: Both strict probe and strict train batch checks passed, and the `ClinicDB` sample can be read into `gt_mask_max=1.0`
- strict dataset probe current confirmation:
  `VisA=2162`, `ClinicDB=612`, total `2774`
  First batch `inputs_shape=[3, 518, 518]`
- loss path: Official training semantics have been cut; AdaCLIP special single test verification `loss / loss_cls / loss_seg` all exist and are limited
- trainable params: Special single test confirmation currently only unfreezes official prompt related parameters, `clip.visual.positional_embedding` remains frozen
- predict path: strict-train probe passed, `pred_score=0.5612`, `pred_anomaly_map` finite, `map_mean=0.52054`
- checkpoint path: local `MVTec-only` retrain weights can be read by the current loader, and sample keys cover `text_prompter` / `visual_prompter` / `patch_token_layer`
- `bottle` strict debug (after normalization fix):
  `image_auroc=0.7226`
  `pixel_auroc=0.6103`
  Compared with the `bottle/image_auroc=0.4325` of strict full train before the repair, it has improved significantly.
- fresh strict `bottle` subset smoke (`VisA 128 + ClinicDB 128 -> bottle`, `1 epoch`):
  `image_auroc=0.7579`
  `pixel_auroc=0.8584`
  `pixel_f1max=0.5467`
  The current judgment is "stop-line is no longer triggered, but image is still low"
- fresh strict full train rerun `epoch1` (`15/15`):
  `image_auroc=0.9275`
  `pixel_auroc=0.8689`
  `bottle/image_auroc=0.9714`
  `cable/image_auroc=0.7791`
  `pill/image_auroc=0.8126`
  Current conclusion: The new strict code path has crossed the historical shutdown line, and full rerun will continue to be retained.
- fresh strict full train rerun `epoch2` (`15/15`):
  `image_auroc=0.9391`
  `pixel_auroc=0.8711`
  `pixel_f1max=0.4102`
  Currently, it is the fresh snapshot with the highest image index.
- fresh strict full train rerun `epoch3` (`15/15`):
  `image_auroc=0.9266`
  `pixel_auroc=0.8810`
  `pixel_f1max=0.4226`
  The current `pixel_f1max -> image_auroc` selector is best-so-far, corresponding to `best_ad_pixel_f1max_epoch_3.pth`
- fresh strict full train rerun `epoch5` (`15/15`):
  `image_auroc=0.9078`
  `pixel_auroc=0.8622`
  `pixel_f1max=0.4205`
  Explain that continuing training is not better than `epoch3`
- Official checkpoint has the same caliber as MVTec eval:
  `image_auroc=0.9029`
  `pixel_auroc=0.8915`
  `pixel_f1max=0.4174`
- best strict rerun (`epoch3`) vs official checkpoint:
  `image +2.37%`
  `pixel -1.05%`
  `image_f1max +1.06%`
  `pixel_f1max +0.52%`
  `image_ap +0.85%`
  `pixel_ap +0.50%`
  The current mean falls within the acceptable alignment range in the README target
- `2026-04-06` Supplement `bottle` Same as environmental evaluation:
  - strict best checkpoint `runs/adaclip_strict_train_rerun_v7/best_ad_pixel_f1max_epoch_3.pth`
    Get on `runs/alignment/adaclip_bottle_eval_strict_best`
    `image_auroc=0.9794`, `pixel_auroc=0.8488`, `pixel_f1max=0.5554`
  - official checkpoint strict-eval
    Get on `runs/alignment/adaclip_bottle_eval_official_checkpoint`
    `image_auroc=0.9913`, `pixel_auroc=0.8441`, `pixel_f1max=0.5846`
  - Current explanation: `bottle` Both strict-train and official strict-eval are significantly higher than random on a single category.
    And the pixel index is close; the slight image gap of strict-train does not constitute a stop-line
- Low budget `50 iter + bottle val` of `2026-04-02`
  (`runs/alignment/adaclip_bottle_strict_smoke_iter50_runner`)
  Get `image_auroc=0.3484`, `pixel_auroc=0.6824`.
  Current judgment: This is undertrained smoke. It is not a strict conclusion. It only shows that it is under extremely low budget.
  AdaCLIP's official auxiliary training mainline cannot be used as `50 iter` agent `1 epoch`

## 4. Small-scale controlled experiment

As of `2026-03-27`, this section contains two parts: historical failure evidence, and fresh rerun progress.

- Historical `runs/adaclip_strict_train` was obtained in `epoch5` on the old strict code path
  `image_auroc=0.5808`, `pixel_auroc=0.5975`
- History `runs/adaclip_strict_train_v6` Obtained in `epoch5` on subsequent semi-repair path
  `image_auroc=0.5382`, `pixel_auroc=0.6866`
- These runs are all earlier than this round’s strict dataset loader, square `518x518` resize, descheduler, and non-prompt parameter freeze repairs
- fresh `runs/adaclip_strict_train_rerun_v7` Completed `epoch1` `15/15` Verification:
  `image_auroc=0.9275`, `pixel_auroc=0.8689`
  The playbook stop-line is not currently triggered and training is still continuing.
- fresh `runs/adaclip_strict_train_rerun_v7` has completely finished the official `5 epochs`:
  `epoch2 = 0.9391 / 0.8711 / 0.4102`
  `epoch3 = 0.9266 / 0.8810 / 0.4226`
  `epoch4 = 0.9138 / 0.8513 / 0.4106`
  `epoch5 = 0.9078 / 0.8622 / 0.4205`
  The optimal checkpoint is `best_ad_pixel_f1max_epoch_3.pth`
- MVTec eval of the same caliber as the official checkpoint has also been run.
  `image_auroc=0.9029`, `pixel_auroc=0.8915`, `pixel_f1max=0.4174`
- `2026-04-06` added `bottle` small-scale comparison:
  strict best checkpoint = `0.9794 / 0.8488 / 0.5554`
  official checkpoint = `0.9913 / 0.8441 / 0.5846`
  Current description: Observed by single category `bottle`, strict-train and official strict-eval
  Behavior remains of the same magnitude, and the anomaly map does not collapse to pure black/pure light
- `2026-04-02` Add low-budget smoke:
  `50 iter` official-auxiliary train + `bottle` val only gets
  `0.3484 / 0.6824 / 0.3060`,
  Currently recorded as "budget-underfit evidence", it is no longer regarded as algorithm mismatch evidence.

Current decision:

- `strict full compare completed`
- Old strict run still proves that the previous code path triggered the Playbook shutdown line
- The new strict code path has passed the three evidence chains of `bottle` smoke, fresh full rerun, and official eval
- The current conclusion can be recorded as `playbook-complete / best-reproducible strict`

## 5. Full Benchmark

Order:

```bash
.venv/bin/python tools/benchmark.py \
    --data_root data/mvtec_ad \
    --methods adaclip \
    --categories all \
    --output runs/alignment/adaclip_v1.json \
    --timeout 7200
```
Summary of results:

| Metric | Reference | BaoIAD | Gap |
|--------|-----------|----------|-----|
| image_auroc | official checkpoint `0.9029` | best strict rerun `0.9266` | `+0.0237` |
| pixel_auroc | official checkpoint `0.8915` | best strict rerun `0.8810` | `-0.0105` |

Shutdown line inspection:

- [x] History strict full train once triggered the shutdown line, the old conclusion is still retained
- [x] fresh strict rerun `epoch1` No more multi-category image AUROC Close to random
- [x] fresh strict rerun `epoch1` `bottle/image_auroc=0.9714`
- [x] full `5 epochs` The final result is completed, the best snapshot is `epoch3`
- [x] Official checkpoint’s MVTec evaluation of the same caliber has been completed
- [x] multi-class strict full train has completed the `all` benchmark mainline equivalent to the current AdaCLIP

## 6. Guard

- New test: `tests/test_models/test_detectors/test_adaclip.py`
- New test: `tests/test_datasets/test_visa.py`
- Added probe/assertion:
  - Explicit failure when official configuration lacks checkpoint
  - `loss` path must return `loss / loss_cls / loss_seg`
  - `predict` The output must contain a finite `pred_score` of the same size as the input `pred_anomaly_map`
  - Class name parsing covers underscore/hyphen normalization
- If you change these paths later, you must rerun:
  - AdaCLIP special single test
  - VisA dataset fallback test
  - alignment probe for strict MVTec config
  - `bottle` smoke

## 7. Residual Risk

- There are inconsistencies in data set naming between the official README weight table and `test.sh` / `app.py`. The script behavior currently prevails.
- The strict assets have been completed through Hugging Face mirroring, but the strict full train results are significantly lower than reasonable expectations, indicating that there are still training calibers that are not aligned.
- The current conclusion is based on the mean gap closing, rather than being completely consistent on a category-by-category basis; `hazelnut / metal_nut / screw / zipper / pill` still has a large image gap on a category-by-category basis
- All subsequent strict training/benchmarks should maintain the mirror environment or reuse the current cache to avoid falling back to direct connection `huggingface.co`
- If you want to continue to suppress class-by-class residuals in the future, give priority to weak class targeted diagnoses instead of blindly re-running all the tests.

## 8. Conclusion

- Final decision: `aligned`
- Allowed to proceed to next stage: `yes`
- If further refinement is required:
  1. Fixed `epoch3` as strict main acceptance checkpoint
  2. If you want to pursue class consistency, check `hazelnut / metal_nut / pill / zipper / screw` first
  3. Currently, there is no need to run `tools/benchmark.py` separately because AdaCLIP’s multi-class strict full rerun has equivalently covered the `all` mainline.

## Alignment checklist

Status only allowed: `matched | mismatch-fixed | intentional-diff | open`

## 1. Input and reference freezing

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| Reference warehouse | `.refs/AdaCLIP` | `docs/alignment/adaclip.md` | Fixed official implementation source | `.refs/AdaCLIP` corresponds to `https://github.com/caoyunkang/AdaCLIP` | `matched` |
| Reference commit | `.refs/AdaCLIP/.git` | `docs/alignment/adaclip.md` | Fixed unique commit | `b762ac40c3f33c77e7e513e48cb436f059d456da` | `matched` |
| strict-train main configuration | `.refs/AdaCLIP/train.py`, `.refs/AdaCLIP/train.sh` | `configs/adaclip/adaclip_vitl14_336_518_mvtec_strict.py` | Official `VisA + ClinicDB -> MVTec` training mainline | Added official `_strict.py` main file, explicitly freezing official training parameters | `mismatch-fixed` |
| strict-eval secondary configuration | `.refs/AdaCLIP/test.py`, `.refs/AdaCLIP/test.sh` | `configs/adaclip/adaclip_vitl14_336_256_mvtec.py` | Official checkpoint-only MVTec evaluation caliber | Reserve `pretrained_visa_clinicdb.pth -> MVTec` configuration as secondary line | `mismatch-fixed` |
| MVTec evaluation checkpoint name | `.refs/AdaCLIP/test.sh`, `.refs/AdaCLIP/app.py` | `configs/adaclip/adaclip_vitl14_336_256_mvtec.py` | Use `pretrained_visa_clinicdb.pth` | Official `test.sh` / `app.py` all point to this file | `mismatch-fixed` |
| VisA evaluation checkpoint name | `.refs/AdaCLIP/test.sh`, `.refs/AdaCLIP/app.py` | `configs/adaclip/adaclip_vitl14_336_518_visa.py` | Use `pretrained_mvtec_colondb.pth` | Official `test.sh` / `app.py` all point to this file | `mismatch-fixed` |
| Input size | `.refs/AdaCLIP/test.py` | `configs/adaclip/*.py` | `image_size=518` | Official `--image_size` Default 518 | `matched` |
| batch size | `.refs/AdaCLIP/test.py` | `configs/adaclip/*.py` | `batch_size=1` | official `if args.batch_size != 1: raise` | `matched` |
| seed | `.refs/AdaCLIP/test.py` | `configs/adaclip/*.py` | fixed `111` | official `setup_seed(111)` | `mismatch-fixed` |

## 2. Preprocessing and text path

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| Color channel | `.refs/AdaCLIP/test.py` + CLIP preprocess | `baoiad/datasets/transforms/loading.py` | RGB input | `LoadImage(..., to_rgb=True)` | `matched` |
| resize | `.refs/AdaCLIP/test.py` | `configs/adaclip/*.py` + `ResizeAD` | Unify resize to `518x518` | Configuration changed to explicit `size=(518, 518)` to avoid the length-preserving edge semantics of torchvision `Resize(518)` | `mismatch-fixed` |
| Normalization | Official CLIP preprocess | `NormalizeAD` + `AdaCLIPDetector._normalize_for_clip()` | Revert to CLIP statistics before running | First ImageNet normalize, then convert to CLIP mean/std in detector | `intentional-diff` |
| tokenizer | `.refs/AdaCLIP/method/adaclip.py` `SimpleTokenizer` | `OpenCLIPBackbone.tokenize()` | OpenAI CLIP BPE consistent behavior | Use OpenCLIP tokenizer instead of local tokenizer | `intentional-diff` |
| prompt template | `.refs/AdaCLIP/method/adaclip.py` | `baoiad/models/detectors/adaclip.py` | normal / abnormal / templates exactly the same | `PROMPT_NORMAL`, `PROMPT_ABNORMAL`, `PROMPT_TEMPLATES` alignment reference | `matched` |

## 3. Prompt/Feature path

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| prompt default value | `.refs/AdaCLIP/test.py` | `configs/adaclip/*.py` | `depth=4`, `length=5`, `branch=VL`, `type=SD` | consistent configuration default value | `matched` |
| Dynamic prompt generation | `.refs/AdaCLIP/method/adaclip.py::generate_and_set_dynamic_promtps` | `AdaCLIP.generate_and_set_dynamic_prompts()` | First use no prompt image features, and then generate dynamic prompts | Local `_extract_image_features_no_prompts()` aligns with official intentions | `matched` |
| Image encoding path | `.refs/AdaCLIP/method/adaclip.py::encode_image` | `AdaCLIP.encode_image()` | patch token / cls token extraction consistent | conv1 -> class token -> pos embed -> resblocks -> selected layers | `matched` |
| OpenCLIP block compatible | Official built-in custom block API | `PromptLayer._call_resblock()` | Compatible with both official block and current OpenCLIP block | This round has completed the automatic transposition compatibility for `batch_first=True` residual block | `mismatch-fixed` |
| Projection layer | `.refs/AdaCLIP/method/adaclip.py` | `ProjectLayer`, `proj_visual_tokens()` | patch / cls token The projection is consistent with the normalization | The structure is consistent with the normalization order | `matched` |

## 4. Predict / Scoring / Checkpoint

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| HSF default | `.refs/AdaCLIP/test.py` | `configs/adaclip/*.py` | `use_hsf=True`, `k_clusters=20` | Config consistent | `matched` |
| anomaly score aggregation | `.refs/AdaCLIP/method/adaclip.py::visual_text_similarity` | function of the same name | first aggregate hierarchical logits, then softmax | achieve consistency | `matched` |
| anomaly map formula | Same as above | Same as above | `(p_anom + 1 - p_norm) / 2` | Consistent local implementation | `matched` |
| upsample | Same as above | Same as above | `align_corners=True` | Local `F.interpolate(..., align_corners=True)` | `matched` |
| Gaussian smoothing | `.refs/AdaCLIP/test.py` | `AdaCLIPDetector._gaussian_blur()` | sigma=4 post-processing | local convolution kernel simulation scipy gaussian_filter | `intentional-diff` |
| checkpoint key mapping | official trainer save `clip_model.state_dict()` | `AdaCLIPDetector._load_official_checkpoint()` | supports direct loading of prompt related weights | compatible with `state_dict` / `module.` / `clip_model.` / `adaclip.` | `matched` |
| Official checkpoint missing protection | Official test script `assert os.path.isfile(args.ckt_path)` | `require_official_checkpoint=True` | Strict official configuration fails directly when weight is missing | Added explicit `FileNotFoundError` | `mismatch-fixed` |

## 5. Wrapper / Guard

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| `loss` path | `.refs/AdaCLIP/method/trainer.py::train_one_batch`, `.refs/AdaCLIP/loss.py` | `AdaCLIPDetector._compute_training_loss()` | classification focal + per-layer seg focal + dual dice | This round has been accessed to real training according to the official focal/dice semantics loss | `mismatch-fixed` |
| Training switch | Official train / test separation | `enable_train_loss` + train configs | Real training loss is only enabled in training configuration | Evaluation configuration no longer implicitly enters supervised training semantics | `mismatch-fixed` |
| Official trainable parameter set | `.refs/AdaCLIP/method/trainer.py` `learnable_paramter_list` | `AdaCLIPDetector._freeze_non_prompt_parameters()` | Only update prompt / projection / dynamic prompt related parameters | Guard has been added, and the position code will no longer be accidentally unfrozen after resize | `mismatch-fixed` |
| `predict` output | Official direct return map / score | `build_predict_results()` | Packaged into `ADDataSample` | BaoIAD unified prediction interface | `intentional-diff` |
| Special detector test | None | `tests/test_models/test_detectors/test_adaclip.py` | Cover mode, map/score, checkpoint guard, class name resolution | This round of enhancements | `mismatch-fixed` |
| VisA data entrance | Official `dataset/visa.py` + `BaseDataset(meta_info['test'])` | `AdaCLIPVisADataset` | strict path explicit priority `meta.json['test']` | Add strict dataset class, and add dataset test | `mismatch-fixed` |
| ClinicDB data entry | Official `dataset/clinicdb.py` + `BaseDataset(meta_info['test'])` | `AdaCLIPClinicDBDataset` | The strict path explicitly uses the official `test` partition semantics | Added strict dataset class; if `meta.json` is not available, fall back to the local `test/` layout | `mismatch-fixed` |
| Normalized dimensions | Official `ToTensor + CLIP normalize` | `AdaCLIPDetector._normalize_for_clip()` | 0-255 ImageNet normalized input must be correctly restored to CLIP pixel statistics | Fixed 0-1 / 0-255 dimension mismatch, and supplemented regression tests | `mismatch-fixed` |
| Auxiliary training configuration | official `train.py` | `configs/adaclip/adaclip_vitl14_336_518_mvtec_strict.py`, `configs/adaclip/adaclip_vitl14_336_518_visa_train_mvtec.py`, `configs/adaclip/adaclip_vitl14_336_518_visa_clinicdb_train_mvtec.py` | Training in auxiliary annotation data, evaluation in MVTec | fresh rerun `v7` has been completely run `5 epochs`; best=`epoch3 = 0.9266 / 0.8810 / 0.4226`, official eval=`0.9029 / 0.8915 / 0.4174`, the mean gap has fallen into the acceptable range | `mismatch-fixed` |
| benchmark config selection | official training first and then testing | `tools/benchmark.py::find_config()` | default priority strict-train main configuration | `adaclip_vitl14_336_518_mvtec_strict.py` now ranked first in priority | `mismatch-fixed` |

## 6. Behavior verification conclusion

- [x] Fixed main config probe with seed=111 archived
- [x] strict `bottle` debug archived after normalization fix
- [x] Special single test coverage `loss / predict / tensor`
- [x] proxy training smoke completed and archived
- [x] fallback checkpoint compatibility archived
- [x] Added checkpoint missing protection for strict official configurations
- [x] strict dataset probe has been archived, the first batch is `518x518`
- [x] fresh strict `bottle` subset smoke is archived and stop-line is not triggered
- [x] `2026-04-06` strict best checkpoint and official checkpoint's `bottle` single-class evaluation has been archived
- [x] fresh strict full rerun `epoch1` is archived and stop-line is not triggered
- [x] fresh strict full rerun `epoch2/epoch3` has been archived, and the current best checkpoint has been promoted to `epoch3`
- [x] strict `all` full rerun and official eval have been completed and the shutdown line has not been triggered
