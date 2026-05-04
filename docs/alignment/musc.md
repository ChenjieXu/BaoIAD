# MuSc strict-alignment evidence

- **Method slug**: `musc`
- **Family**: Vision-language / foundation
- **Method README**: [`configs/musc/README.md`](../../configs/musc/README.md)

This file preserves the fixed alignment evidence migrated from the manuscript evidence workspace. The content below is translated to English where needed and kept inside this repository so the strict-alignment state can be reviewed without relying on an external paper checkout.

## Config entry points

- [`configs/musc/musc_vitl14_336_518_mvtec_strict.py`](../../configs/musc/musc_vitl14_336_518_mvtec_strict.py)
- [`configs/musc/musc_vitl14_336_518_visa.py`](../../configs/musc/musc_vitl14_336_518_visa.py)

## Detailed alignment report

## Status

- **Playbook status**: `playbook-complete`
- **Reference repo**: local `.refs/MuSc`
- **Reference commit**: `72d58ad56c0cafa2b056bd0aa7676f9c21fccbc4`
- **Reference entrypoint**: `.refs/MuSc/examples/musc_main.py`
- **Reference config**: `.refs/MuSc/configs/musc.yaml`
- **Strict config**: `configs/musc/musc_vitl14_336_518_mvtec_strict.py`
- **Checklist**: `docs/alignment/musc_checklist.md`

MuSc is a zero-shot method. This round of alignment focuses on the official CLIP backbone, preprocessing, whole-test mutual scoring and per-category benchmark protocols, rather than the training budget.

## Reference Freeze

### Run mainline

- Official warehouse: `https://gh-proxy.com/https://github.com/xrli-U/MuSc`
- The only running entrance: `.refs/MuSc/examples/musc_main.py`
- Only main configuration: `.refs/MuSc/configs/musc.yaml`
- Core implementation: `.refs/MuSc/models/musc.py`
- Core modules: `.refs/MuSc/models/modules/_LNAMD.py`, `_MSM.py`, `_RsCIN.py`
- MVTec data path and transformation: `.refs/MuSc/datasets/mvtec.py` + reference `open_clip`

### Freeze parameters

- backbone:`ViT-L-14-336`
- pretrained:`openai`
- input resolution:`518`
- batch size: `4`
- feature layers: `[5, 11, 17, 23]`
- LNAMD `r_list`: `[1, 3, 5]`
- MSM `topmin` Range: `[0.0, 0.3]`
- RsCIN `k_list`: MVTec path fixed to `[1, 2, 3]`
- seed:`42`

### Training / loss caliber

- Officially no optimizer, scheduler, epoch budget or early stopping
- There is no official parameter update training; `examples/musc_main.py` directly executes mutual scoring on the test set
- strict caliber:
  - optimizer:`N/A`
  - scheduler: `N/A`
  - epochs: `N/A`
  - loss:`N/A`
- BaoIAD reserves `train_cfg + optim_wrapper + loss=0` only for MMEngine compatibility and does not participate in official numerical alignment

### Predict path

The official MVTec main path is frozen as:

`CLIP encode_image -> LNAMD(r in [1,3,5]) -> MSM(topmin 0~0.3) -> mean over layers -> mean over r_list -> bilinear upsample to 518 -> image score = map max -> RsCIN(k=[1,2,3])`

### Special Agreement

- MuSc is **zero-shot whole-test mutual scoring**
- benchmarks must be run independently by category**
- The CLIP strict path must use the modified version `open_clip` that comes with the MuSc repository.
- MuSc YAML's `feature_layers=[5,11,17,23]` needs to be parsed into the extraction layer `[6,12,18,24]` at runtime

## Close this round

- Create a new formal strict entry: `configs/musc/musc_vitl14_336_518_mvtec_strict.py`
- `tools/benchmark.py` default `musc` config has been switched to strict file
- `CONFIG_MATRIX` updated from `nonestrict` to the official strict mainline
- The core detector/backbone algorithm path will continue to use the existing implementation after review, without new formula-level fixes

## Code path comparison conclusion

See `docs/alignment/musc_checklist.md`。 Current conclusion:

- `LNAMD / MSM / RsCIN / score_all()` is still consistent with the official implementation
- strict CLIP preprocessing continues to be fixed to `OpenCLIPPreprocessAD`
- strict backbone continues to be fixed `require_ref_open_clip=True`
- benchmark continues to fix per-category eval-only and no longer mixes 15 categories into one scoring pool

## evidence

- **Probe**: `runs/alignment/musc_probe.json`
- **Bottle smoke**: `runs/alignment/musc_bottle_smoke/`
- **Strict full benchmark**: `runs/alignment/musc_strict_full_20260406.json`
- **Published MVTec AD headline**: image AUROC `97.8`, pixel AUROC `97.3`, PRO `93.8`

Current fresh strict full main results:

- BaoIAD strict `15/15`: `image_auroc=0.9777`, `pixel_auroc=0.9711`, `aupro=0.9402`
- Difference from published headline: image `-0.03%`, pixel `-0.19%`, AUPRO `+0.22%`

Gate 3 Description:

- `bottle` smoke still uses `train.py + max_epochs=5` for link verification
- Since MuSc has no official training, runner-compatible `loss` will remain constant `0.0`
- The smoke condition has been met at the `1`th val in this round, so subsequent repetitions of epochs will be stopped after getting the valid indicator.
- The real acceptance signal for smoke is:
  - No NaN / No divergence
  - image AUROC is significantly higher than `0.5`
  - anomaly map does not collapse to all zeros or all brights

New evidence for this round:

- `runs/alignment/musc_probe.json`: `13/13` checks passed
- `runs/alignment/musc_bottle_smoke/20260402_203305/20260402_203305.log`: The `1`th val reaches `img=0.9992 / pxl=0.9848 / aupro=0.9614`
- `benchmark.find_config('musc')` now resolves to `configs/musc/musc_vitl14_336_518_mvtec_strict.py`
- `runs/alignment/musc_strict_full_20260406.json`: fresh Gate 4 `15/15` rerun completed, mean `img=0.9777 / pxl=0.9711 / aupro=0.9402`
- fresh Gate 4 is completely consistent with the historical archive `runs/alignment/musc_strict_full.json` category by category, `max_abs(image_delta)=0.0`, `max_abs(pixel_delta)=0.0`

## Residual Notes

- `screw` is still the weakest image-side category, more like RSCIN / image aggregation sensitivity issue than segmentation path corruption
- `metal_nut` and `transistor` are still weak pixel-side categories
- These residual weak classes do not affect the strict mean to enter the acceptable range of published headline

## Alignment checklist

| Module | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| backbone layer selection | `.refs/MuSc/configs/musc.yaml` + `.refs/MuSc/models/musc.py` | `baoiad/models/backbones/musc_clip_backbone.py` | YAML `feature_layers=[5,11,17,23]` is parsed to CLIP extraction layer `[6,12,18,24]` | `tests/test_models/test_detectors/test_musc.py::test_musc_clip_backbone_uses_reference_layer_offset` | `matched` |
| CLIP strict import | `.refs/MuSc/models/musc.py` + `.refs/MuSc/models/backbone/open_clip/*` | `baoiad/models/backbones/musc_clip_backbone.py` | strict CLIP mainline must use MuSc modified version `open_clip`, and cannot be silently returned generic path | `tests/test_models/test_detectors/test_musc.py::test_musc_clip_backbone_can_require_reference_open_clip` | `matched` |
| Input preprocessing | `.refs/MuSc/datasets/mvtec.py` + `.refs/MuSc/models/backbone/open_clip/transform.py` | `configs/musc/musc_vitl14_336_518_mvtec_strict.py` + `OpenCLIPPreprocessAD` | `RGB -> Resize((518,518), bicubic) -> CenterCrop(518) -> ToTensor() -> OpenAI normalize` | `tests/test_datasets/test_transforms.py::TestOpenCLIPPreprocessAD` | `matched` |
| loss path | `.refs/MuSc/examples/musc_main.py` | `baoiad/models/detectors/musc.py` | Official no training loss; BaoIAD only retains the runner compatible shell of `loss=0` | code review + zero-shot reference freeze | `intentional-diff` |
| LNAMD / MSM / RsCIN formula and sequence | `.refs/MuSc/models/modules/_LNAMD.py` + `_MSM.py` + `_RsCIN.py` + `.refs/MuSc/models/musc.py` | `baoiad/models/detectors/musc.py::score_all` | `LNAMD -> MSM -> mean(layer) -> mean(r_list) -> upsample -> image max -> RsCIN` | code review | `matched` |
| anomaly map generation | `.refs/MuSc/models/musc.py` | `baoiad/models/detectors/musc.py::score_all` | bilinear upsampling to `518`, no additional smoothing, no color/order change | code review + probe/smoke outputs | `matched` |
| image score aggregation | `.refs/MuSc/models/musc.py` | `baoiad/models/detectors/musc.py::score_all` | First take `map max` to get `ac_score`, then do `RsCIN(k=[1,2,3])` | code review | `matched` |
| whole-test deferred scoring | Official zero-shot test process | `ADTestLoop` + `MuScDetector.score_all()` | First cache the placeholders of all test samples, and then perform mutual scoring uniformly | `tests/test_models/test_detectors/test_musc.py::test_musc_score_all_updates_placeholder_predictions` | `matched` |
| special protocol: per-category eval | `.refs/MuSc/models/musc.py` category loop | `configs/musc/musc_vitl14_336_518_mvtec_strict.py` + `tools/benchmark.py` | benchmark must be run separately by category, and 15 categories cannot be mixed into a mutual-scoring pool | `benchmark_multi_class=False`, `benchmark_eval_only=True`, `tests/test_utils/test_benchmark_config_detection.py::test_musc_benchmark_prefers_strict_config` | `mismatch-fixed` |
| strict configuration identity | playbook strict naming | `configs/musc/musc_vitl14_336_518_mvtec_strict.py` | MuSc mainline requires formal `_strict.py` entry instead of just relying on document declaration strict | `tests/test_utils/test_benchmark_config_detection.py::test_musc_strict_config_freezes_official_zero_shot_hparams` | `mismatch-fixed` |
| Gate 2 probe | playbook Gate 2 | strict config + `tools/alignment_probe.py` | train/test structure access passed | `runs/alignment/musc_probe.json` (`13/13` checks passed) | `matched` |
| Gate 3 smoke | playbook Gate 3 | strict config + `tools/train.py` | `bottle` smoke no NaN, image score can be separated, map does not collapse | `runs/alignment/musc_bottle_smoke/20260402_203305/20260402_203305.log` (`epoch1 val: img=0.9992, pxl=0.9848, aupro=0.9614`) | `matched` |
| Gate 4 full benchmark | playbook Gate 4 | `tools/benchmark.py --methods musc --categories all` | strict `15/15` is close to published headline and does not trigger stop-line | `runs/alignment/musc_strict_full_20260406.json` (`15/15`, avg `img=0.9777`, `pxl=0.9711`, `aupro=0.9402`) + fresh/old archive exact-match compare | `matched` |

## Notes

- Anomalib is not used as the main reference for MuSc in this round; the only authority is the official MuSc repository
- `loss=0` is the framework compatibility layer and should not be misread as "MuSc official training loss"
