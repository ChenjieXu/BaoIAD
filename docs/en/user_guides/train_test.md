# Training and Testing

## Training

Train an anomaly detector using `tools/train.py`:

```bash
python tools/train.py <config> --work-dir runs/<experiment_name>
```

### Arguments

| Argument | Required | Description |
|---|---|---|
| `config` | Yes | Path to a training config file (e.g. `configs/patchcore/patchcore_wrn50_256_mvtec_strict.py`) |
| `--work-dir` | No | Directory for checkpoints and logs. Defaults to the config's `work_dir` field. |
| `--resume` | No | Resume training from the latest checkpoint in `work_dir` (reads `<work_dir>/last_checkpoint`). |
| `--cpu` | No | Force CPU-only execution. Disables CUDA and MPS backends. |
| `--cfg-options` | No | Override config values in `key=value` format. See [Config Guide](config.md) for details. |

### Examples

Train PatchCore on a single MVTec AD category:

```bash
python tools/train.py configs/patchcore/patchcore_wrn50_256_mvtec_strict.py \
    --work-dir runs/patchcore_bottle \
    --cfg-options \
    train_dataloader.dataset.data_root=data/mvtec_ad \
    "train_dataloader.dataset.cls_names=['bottle']" \
    train_dataloader.dataset.multi_class=False \
    test_dataloader.dataset.data_root=data/mvtec_ad \
    "test_dataloader.dataset.cls_names=['bottle']" \
    test_dataloader.dataset.multi_class=False
```

Override training epochs:

```bash
python tools/train.py configs/rd/rd_wrn50_256_mvtec_strict.py \
    --work-dir runs/rd_short \
    --cfg-options train_cfg.max_epochs=50
```

Resume a interrupted training run:

```bash
python tools/train.py configs/draem/draem_256_mvtec_strict.py \
    --work-dir runs/draem_bottle \
    --resume
```

### Multi-GPU Training

BaoIAD uses the MMEngine launcher for distributed training. Launch with `torchrun` or `torch.distributed.launch`:

```bash
torchrun --nproc_per_node=2 tools/train.py <config> --work-dir runs/...
```

The config must set `launcher='pytorch'` (some configs do this via `_base_`). Refer to the [MMEngine documentation](https://mmengine.readthedocs.io/) for multi-node setup.

## Testing

Evaluate a trained model using `tools/test.py`:

```bash
python tools/test.py <config> <checkpoint> --work-dir runs/<experiment_name>
```

### Arguments

| Argument | Required | Description |
|---|---|---|
| `config` | Yes | Path to the test config file (usually the same as training). |
| `checkpoint` | No | Path to a checkpoint `.pth` file. If provided, overrides `load_from` in the config. |
| `--work-dir` | No | Directory for test outputs (logs, metrics, visualizations). |
| `--cpu` | No | Force CPU-only execution. |
| `--cfg-options` | No | Override config values in `key=value` format. |

### Examples

Test with a specific checkpoint:

```bash
python tools/test.py configs/patchcore/patchcore_wrn50_256_mvtec_strict.py \
    runs/patchcore_bottle/epoch_100.pth \
    --work-dir runs/patchcore_bottle_test
```

Test on CPU:

```bash
python tools/test.py configs/rd/rd_wrn50_256_mvtec_strict.py \
    runs/rd_bottle/best.pth \
    --cpu
```

## Output Directory Structure

After training or testing, the work directory contains:

```
runs/<experiment_name>/
├── <timestamp>.log           # Training/testing log
├── <timestamp>.log.json      # Structured log (JSON lines)
├── last_checkpoint           # Path to the latest checkpoint (for resume)
├── epoch_10.pth              # Periodic checkpoints
├── epoch_20.pth
├── ...
├── vis/                      # Visualization outputs (if enabled)
│   ├── 000000.png
│   ├── 000001.png
│   └── ...
└── latest_metrics.json       # RegAD-specific: latest epoch metrics
```

Checkpoints are managed by the `CheckpointHook` (configured in `default_hooks.checkpoint`). The default settings (from [`configs/_base_/default_runtime.py`](../../../configs/_base_/default_runtime.py)) save every 10 epochs and keep at most 3 checkpoints.

## Special Training Scripts

Some methods require training procedures that differ from the standard MMEngine `Runner.train()` loop. BaoIAD provides dedicated scripts for these.

### tools/train_ast.py — Two-Stage AST Training

AST (Asymmetric Student-Teacher) trains in two sequential stages:

1. **Teacher stage**: Trains the teacher model with `model.training_phase=teacher`.
2. **Student stage**: Trains the student model with `model.training_phase=student`, loading the teacher checkpoint from stage 1.

The script handles checkpoint passing between stages automatically.

```bash
python tools/train_ast.py configs/ast/ast_effnet_b5_768_mvtec_strict.py \
    --work-dir runs/ast_bottle
```

**Arguments**: Same as `tools/train.py` (`config`, `--work-dir`, `--resume`, `--cpu`, `--cfg-options`).

**When to use**: Always use this script instead of `tools/train.py` for AST configs.

### tools/train_regad_strict.py — RegAD Few-Shot Protocol

RegAD uses a few-shot protocol with support-set sampling and cross-category training. This script implements the official RegAD training loop with:

- Deterministic support-set sampling (or loading official support sets)
- Cosine-annealing learning rate schedule
- Per-epoch evaluation with multi-round support set averaging
- Best-checkpoint tracking by balanced image+pixel AUROC

```bash
python tools/train_regad_strict.py configs/regad/regad_wrn50_256_mvtec_strict.py \
    --work-dir runs/regad_bottle
```

**Arguments**: Same as `tools/train.py` (`config`, `--work-dir`, `--resume`, `--cpu`, `--cfg-options`).

**When to use**: Always use this script for RegAD configs. It does not use the MMEngine `Runner`; instead it runs a custom training loop to match the official RegAD protocol exactly.

### tools/train_vitad_exact_order.py — ViTAD Verified Sample Order Replay

ViTAD's official training uses a specific per-epoch sample ordering. BaoIAD does not generate or distribute that artifact. Given a user-supplied order JSON whose origin has been verified, this script:

1. Refuses to start when the order file is missing.
2. Overrides the sampler to `PerEpochOrderSampler` with the supplied file.
3. Replays the recorded per-epoch ordering during training.

```bash
python tools/train_vitad_exact_order.py configs/vitad/vitad_256_mvtec_strict.py \
    --order-file /path/to/verified_vitad_order.json \
    --work-dir runs/vitad_bottle
```

**Arguments**: Same as `tools/train.py`, plus `--order-file`. Standard MMEngine resume checkpoints may contain Python objects; use `--trusted-checkpoint` only after verifying their origin and integrity.

**When to use**: Use this script only when a verified order artifact is available. Supplying an arbitrary JSON does not establish official equivalence.

## Resume Training

To resume from a previously interrupted run:

```bash
python tools/train.py <config> --work-dir <same_work_dir> --resume
```

The `--resume` flag reads `<work_dir>/last_checkpoint` to find the most recent checkpoint and restores the model state, optimizer state, and epoch counter. The checkpoint file must exist and be valid.

For the special training scripts, `--resume` works the same way but respects each script's checkpoint conventions:
- **AST**: Resumes each stage independently.
- **RegAD**: Restores model, optimizer, and best-metrics tracking.
- **ViTAD**: Resumes from the MMEngine checkpoint.
