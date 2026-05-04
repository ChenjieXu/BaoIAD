# AnomalyCLIP strict-alignment evidence

- **Method slug**: `anomalyclip`
- **Family**: Vision-language / foundation
- **Method README**: [`configs/anomalyclip/README.md`](../../configs/anomalyclip/README.md)

This file preserves the fixed alignment evidence migrated from the manuscript evidence workspace. The content below is translated to English where needed and kept inside this repository so the strict-alignment state can be reviewed without relying on an external paper checkout.

## Config entry points

- [`configs/anomalyclip/anomalyclip_vitl14_336_518_mvtec_strict.py`](../../configs/anomalyclip/anomalyclip_vitl14_336_518_mvtec_strict.py)
- [`configs/anomalyclip/anomalyclip_vitl14_336_518_visa.py`](../../configs/anomalyclip/anomalyclip_vitl14_336_518_visa.py)

## Detailed alignment report

**Status**: `playbook-complete`
**Date**: `2026-04-06`

## 1. Reference freezing

- **Reference**: Official repository `zqhang/AnomalyCLIP`, commit `3911738c0867544f545a076ad78f3f11d9ecbfdf`
- **Authoritative Code Path**:
  - `.refs/AnomalyCLIP/train.py`
  - `.refs/AnomalyCLIP/test.py`
  - `.refs/AnomalyCLIP/train.sh`
  - `.refs/AnomalyCLIP/test.sh`
  - `.refs/AnomalyCLIP/loss.py`
  - `.refs/AnomalyCLIP/utils.py`
  - `.refs/AnomalyCLIP/results/9_12_4_multiscale/zero_shot/log.txt`
- **Reference results**: MVTec mean `image AUROC = 91.6`, `pixel AUROC = 91.1`
- **Official Agreement Freeze**:
  - backbone / weights: `AnomalyCLIP_lib.load("ViT-L/14@336px")`
  - input size: `518`
  - train batch size: `8`
  - test batch size: `1`
  - dataloader workers: `0`
  - optimizer: `Adam(lr=1e-3, betas=(0.5, 0.999), weight_decay=0.0)`
  - scheduler: none
  - train budget: `15` epochs, `save_freq=1`, no early stopping
  - loss: `cross_entropy(image logits) + 4 * (focal + dice_anom + dice_norm)`
  - predict path:
    - image score: anomaly probability of `softmax(image_features @ text_features / 0.07)`
    - anomaly map: per-layer similarity map is converted to `(p_anom + 1 - p_norm) / 2` and then summed layer by layer
    - smoothing: `gaussian_filter(sigma=4)`
  - Special training protocols:
    - CLIP backbone full course `eval` + `requires_grad=False`
    - Training only `AnomalyCLIP_PromptLearner`
    - DPAM fixed `20`
    - The official `Dataset(...)` of `train.py` does not explicitly pass `mode`, so the auxiliary train actually reads the VisA `test` split
    - The official transform is `Resize((518, 518))`; not "short side to 518"
    - The official training script does not run MVTec verification within epoch, only saves the checkpoint, and separate it after training `test.py`
    - MVTec strict mainline is `VisA -> MVTec`

## 2. Current main line

- **strict main configuration**: `configs/anomalyclip/anomalyclip_vitl14_336_518_mvtec_strict.py`
- **legacy archive**: `configs/anomalyclip/anomalyclip_vitl14_336_518_visa_train_mvtec.py`
- **Checklist**: [anomalyclip_checklist.md](anomalyclip_checklist.md)

This round refreezes `anomalyclip` from the old eval-only checkpoint path to the trainable strict mainline. Currently the strict mainline has been completed:

- Official `train.py/test.py` caliber trainable `VisA -> MVTec` config
- Official `Resize((518, 518))` and `num_workers=0` paths
- `tools/benchmark.py` default entry switched to new `_strict.py`
- Checkpoint reload path repair after train
- probe / smoke / checkpoint reload test all cleared

The old eval-only files remain, but only as historical archives and no longer represent the current strict identity.

## 3. Code path comparison conclusion

See [anomalyclip_checklist.md](anomalyclip_checklist.md) for the control matrix.

### Consistency confirmed

- strict runtime authority is still official `AnomalyCLIP_lib` + official `AnomalyCLIP_PromptLearner`
- `ViT-L/14@336px`, `features_list=[24]`, `prompt depth=9 / n_ctx=12 / t_n_ctx=4`, `temperature=0.07`, `DPAM=20` are all consistent with the official script
- The train loss formula and predict score/map aggregation path are consistent with the official `train.py/test.py`

### Repair this round

- Added trainable strict config `configs/anomalyclip/anomalyclip_vitl14_336_518_mvtec_strict.py`
- strict train dataloader changed to official semantic `VisA split='test'`
- strict preprocess changed to official `518x518` square resize, removing the old "short side=518" approximate path
- strict dataloader workers changed to official `0`, the default training no longer runs MVTec val in each round
- The default entry for strict benchmark is changed to the new `_strict.py`
- `AnomalyCLIPOfficialDetector` adds visual positional embedding resize in the initialization phase to fix the problem that checkpoint cannot be reloaded after train.

### Currently completed

- fresh `15/15` full strict benchmark has been rerun according to the new trainable strict main line.

## 4. Behavior Probe

Order:

```bash
python tools/alignment_probe.py configs/anomalyclip/anomalyclip_vitl14_336_518_mvtec_strict.py \
    --splits train test \
    --max-batch-size 2 \
    --device cuda \
    --output runs/alignment/anomalyclip_probe_v2.json
```
in conclusion:

- `runs/alignment/anomalyclip_probe_v2.json` Archived, `passed=true`
- both train/test paths pass.
- The train batch preview hits the official auxiliary semantics: `data/visa/pcb2/test/bad/032.JPG`
- train inputs have been restored to their official consistent square shape `518x518`

Key statistics:

- train loss:
  - `loss=6.6076`
  - `loss_cls=0.6898`
  - `loss_seg=5.9178`
- test predict:
  - `pred_score=0.6042`
  - map stats: `min=0.1452`, `max=0.7521`, `mean=0.5237`, `std=0.0663`

## 5. Smoke

The user's original smoke template also changed the train set to `MVTec bottle`, which conflicts with the official `VisA -> MVTec` mainline. What is actually executed in this round is the official protocol compatible version of smoke:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    configs/anomalyclip/anomalyclip_vitl14_336_518_mvtec_strict.py \
    --work-dir runs/alignment/anomalyclip_bottle_smoke \
    --cfg-options \
        train_cfg.max_epochs=5 \
        train_cfg.val_interval=1 \
        train_cfg.val_begin=1 \
        train_dataloader.dataset.cls_names="['candle']" \
        train_dataloader.dataset.multi_class=False \
        test_dataloader.dataset.cls_names="['bottle']" \
        test_dataloader.dataset.multi_class=False \
        val_dataloader.dataset.cls_names="['bottle']" \
        val_dataloader.dataset.multi_class=False
```
result:

- epoch 1:
  - train loss `4.5505`
  - val `image_auroc=0.6802`, `pixel_auroc=0.8841`
- epoch 2:
  - train loss `4.3571`
  - val `image_auroc=0.7452`, `pixel_auroc=0.8801`
- epoch 3:
  - train loss `4.2859`
  - val `image_auroc=0.7952`, `pixel_auroc=0.8850`
- epoch 4:
  - train loss `4.0989`
  - val `image_auroc=0.7984`, `pixel_auroc=0.8878`
- epoch 5:
  - train loss `4.2698`
  - val `image_auroc=0.8183`, `pixel_auroc=0.8881`

determination:

- `pass`
- loss is limited throughout, no divergence, no NaN
- `image_auroc` is significantly higher than `0.5`
- After training, checkpoint can be reloaded by fresh process

checkpoint reload verification:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    configs/anomalyclip/anomalyclip_vitl14_336_518_mvtec_strict.py \
    runs/alignment/anomalyclip_bottle_smoke/epoch_5.pth \
    --work-dir runs/alignment/anomalyclip_bottle_smoke_eval \
    --cfg-options \
        test_dataloader.dataset.cls_names="['bottle']" \
        test_dataloader.dataset.multi_class=False
```
- `tools/test.py` successfully reloaded `epoch_5.pth`
- The test indicator is consistent with the smoke val: `image_auroc=0.8183`, `pixel_auroc=0.8881`

Single sample prediction statistics after training:

- `pred_score=0.9346`
- map stats:
  - `min=0.00068`
  - `max=0.9174`
  - `mean=0.0919`
  - `std=0.1908`

This shows that the anomaly map after smoke is not all zeros, nor is the entire map bright.

## 6. Conclusion

- Current conclusion: `playbook-complete`
- Completed gate:
  - [x] Reference frozen rewrite
  - [x] Code path checklist rewritten
  - [x] strict config build
  - [x] benchmark default entry switch
  - [x] strict probe passed
  - [x] bottle smoke passed
  - [x] checkpoint reload path repair and verification
- Completed results:
  - [x] fresh `15/15` full strict benchmark

Historical note:

- The archive results `image_auroc=0.9188`, `pixel_auroc=0.9105` of the old eval-only mainline can still be used as checkpoint caliber circumstantial evidence
- but it no longer represents the current strict mainline, since the current strict has been redefined as a trainable `VisA -> MVTec` path

## 7. fresh full strict rerun

`2026-04-06` Completed fresh trainable strict benchmark:

```bash
CUDA_VISIBLE_DEVICES=3 python tools/benchmark.py \
    --data_root data/mvtec_ad \
    --categories all \
    --methods anomalyclip \
    --device cuda \
    --timeout 20000 \
    --output runs/alignment/anomalyclip_trainable_strict_full.json \
    --work-dir-root runs/alignment/benchmark_trainable_strict
```
Results archive:

- merged json: `runs/alignment/anomalyclip_trainable_strict_full.json`
- training work dir: `runs/alignment/benchmark_trainable_strict/anomalyclip/all`
- Training log: `runs/alignment/benchmark_trainable_strict/anomalyclip/all/20260406_091739/20260406_091739.log`
- Duration: `14332.8s` (about `3.98h`)

In order to confirm the direction as soon as possible, a full MVTec test has been done on `epoch_1.pth`:

```bash
CUDA_VISIBLE_DEVICES=1 python tools/test.py \
    configs/anomalyclip/anomalyclip_vitl14_336_518_mvtec_strict.py \
    runs/alignment/benchmark_trainable_strict/anomalyclip/all/epoch_1.pth \
    --work-dir runs/alignment/anomalyclip_epoch1_eval
```
Epoch-1 full results:

- mean `image_auroc=0.8992`
- mean `pixel_auroc=0.8963`

Final full strict result:

- mean `image_auroc=0.9162`
- mean `pixel_auroc=0.9089`
- Gap relative to official reference `0.916 / 0.911`:
  - image `+0.0002`
  - pixel `-0.0021`

This shows that the main line of trainable strict has basically coincided with the official reference, and can be officially closed according to the playbook.

## Alignment checklist

Status only allowed: `matched | mismatch-fixed | intentional-diff | open`

## 1. Input and reference freezing

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| Reference repository | `.refs/AnomalyCLIP` | `docs/alignment/anomalyclip.md` | Fixed unique reference source | commit frozen as `3911738c0867544f545a076ad78f3f11d9ecbfdf` | `matched` |
| MVTec main protocol | `README.md`, `train.sh`, `test.sh`, `results/9_12_4_multiscale/zero_shot/log.txt` | `configs/anomalyclip/anomalyclip_vitl14_336_518_mvtec_strict.py` | strict The main line is `VisA -> MVTec` | New `_strict.py` has been switched to this protocol | `mismatch-fixed` |
| auxiliary train split | `train.py` default `Dataset(..., mode='test')` | strict config `train_dataloader.dataset.split='test'` | training side uses VisA `test` split | probe train preview=`data/visa/pcb2/test/bad/032.JPG` | `mismatch-fixed` |
| input size | `train.py`, `test.py`, `train.sh`, `test.sh` | strict config + detector | `image_size=518` | strict config fixed `518` | `matched` |
| resize semantics | `utils.py::get_transform` | strict pipeline `ResizeAD(size=(518,518))` | Officially square `Resize((518,518))`, not short side `518` | `probe_v2` train inputs have become `518x518` | `mismatch-fixed` |
| batch size | `train.py`, `test.py`, `train.sh`, `test.sh` | strict config | train `8`, test `1` | strict config explicit freeze | `matched` |
| dataloader workers | `train.py`, `test.py` Default DataLoader | strict config | `num_workers=0` | strict config has been changed to `0` / `persistent_workers=False` | `mismatch-fixed` |
| seed / cudnn deterministic | `train.py::setup_seed`, `test.py::setup_seed` | strict config | `seed=111`, `deterministic=True`, `benchmark=False` | `randomness=dict(seed=111, deterministic=True)` | `mismatch-fixed` |

## 2. CLIP / Prompt / DPAM

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| CLIP build entry | `.refs/AnomalyCLIP/AnomalyCLIP_lib.load` | `AnomalyCLIPOfficialDetector` | Use the official `load("ViT-L/14@336px")` | strict detector directly calls the official load | `matched` |
| Prompt learner | `.refs/AnomalyCLIP/prompt_ensemble.py` | `AnomalyCLIPOfficialDetector.prompt_learner` | Use the official `AnomalyCLIP_PromptLearner` | strict detector directly import the official prompt learner | `matched` |
| prompt hyperparameters | `train.py`, `test.py`, `train.sh`, `test.sh` | strict config | `depth=9`, `n_ctx=12`, `t_n_ctx=4` | strict config explicit freeze | `matched` |
| DPAM surgery | `train.py`, `test.py` | strict detector | `DAPM_replace(20)` | detector fixed during initialization `20` | `matched` |
| CLIP freeze + prompt-only train | `train.py` | strict detector + optimizer | CLIP `eval` + `requires_grad=False`, only train prompt learner | smoke train path passed | `matched` |
| visual positional embedding resize | `AnomalyCLIP.py::forward` will update positional embedding according to the input grid | `AnomalyCLIPOfficialDetector._resize_visual_positional_embedding()` | The fresh model must first be expanded to the `518` grid, and checkpoint can be reloaded after training | `epoch_5.pth` -> `tools/test.py` Successfully reloaded | `mismatch-fixed` |

## 3. Preprocessing / Loss / Predict

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| image reading | `dataset.py` + PIL transforms | strict pipeline `LoadImage(backend='pil')` | PIL / RGB path | strict config explicit freeze | `mismatch-fixed` |
| mask read | `dataset.py` + PIL transforms | strict pipeline `LoadMask(backend='pil')` | PIL / nearest mask path | strict config explicit freeze | `mismatch-fixed` |
| resize | `utils.py::get_transform` | strict pipeline `ResizeAD(size=(518,518), ... official_pil=True)` | PIL bicubic square resize to `518x518` | `probe_v2` train/test shapes aligned | `mismatch-fixed` |
| Normalization | `utils.py::get_transform` Use CLIP stats | `NormalizeAD` + detector `_normalize_for_clip()` | repo first goes to ImageNet norm, detector then restores CLIP norm | Current repo unified interface difference | `intentional-diff` |
| loss formula | `train.py` | strict detector `mode='loss'` | `CE + 4 * (focal + dice_anom + dice_norm)` | probe train loss / smoke train loss normal | `matched` |
| image score | `test.py` | strict detector `predict` | anomaly softmax prob | probe/test path passed | `matched` |
| anomaly map | `test.py` | strict detector `predict` | per-layer similarity -> anomaly map -> sum | probe/test path passed | `matched` |
| Gaussian smoothing | `test.py` of `gaussian_filter(sigma=4)` | strict detector `_gaussian_blur_bchw()` | `sigma=4` smoothing | Torch convolution approximation SciPy | `intentional-diff` |

## 4. Wrapper / Benchmark

| Project | Reference Path | BaoIAD Path | Desired Behavior | Evidence | Status |
|------|----------|---------------|----------|------|------|
| strict config naming | playbook requirements | `configs/anomalyclip/anomalyclip_vitl14_336_518_mvtec_strict.py` | `_strict.py` main entry | new file created | `mismatch-fixed` |
| benchmark default entry | `tools/benchmark.py` | `_METHOD_CONFIG_PRIORITY['anomalyclip']` | strict file ranks first | metadata test passed | `mismatch-fixed` |
| train-root protection | `VisA -> MVTec` protocol | `benchmark_keep_train_data_root=True` | benchmark Do not change train root back to MVTec | metadata test passed | `matched` |
| Automatic test after train | strict trainable mainline | `benchmark_test_after_train=True` | benchmark checkpoint must be loaded after training to run `tools/test.py` | metadata test + `tools/test.py` reload passed | `mismatch-fixed` |
| MVTec is not done by default during the training period val | official `train.py` | strict config `train_cfg.val_begin=16`, `val_interval=16` | 15 epoch official train does not run within the epoch val | strict config has been changed to skip the training period val | `mismatch-fixed` |
| legacy eval-only path | historical archive | `anomalyclip_vitl14_336_518_visa_train_mvtec.py` | retained but no longer represents the strict mainline | Documentation downgraded to archive | `mismatch-fixed` |
| repo-local approximation | Unofficial strict | `AnomalyCLIPDetector` + `anomalyclip_vitl14_336_256_mvtec.py` | Keep fallback, no longer pretend to be strict | benchmark has been removed by default | `mismatch-fixed` |

## 5. Behavior verification

- [x] strict `alignment_probe` passed
- [x] `bottle` smoke (`VisA candle -> MVTec bottle`) passed
- [x] `epoch_5.pth` can be successfully reloaded by fresh `tools/test.py`
- [x] anomaly map non-degenerate after smoke
- [x] fresh `15/15` full strict benchmark completed
