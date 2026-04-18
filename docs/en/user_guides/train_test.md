# Train and Test

## Training

### Basic Training

Train a method on all categories of a dataset:

```bash
python tools/train.py configs/patchcore/patchcore_wrn50_256_mvtec_strict.py \
    --work-dir runs/patchcore
```

### Single-Category Training

Most anomaly detection methods train one model per product category. To train on a specific category:

```bash
python tools/train.py configs/patchcore/patchcore_wrn50_256_mvtec_strict.py \
    --work-dir runs/patchcore_bottle \
    --cfg-options \
    train_dataloader.dataset.cls_names="['bottle']" \
    train_dataloader.dataset.multi_class=False
```

### Multi-Category Training

Some methods (UniAD, ViTAD, InvAD) support training a single model on all categories:

```bash
python tools/train.py configs/uniad/uniad_effnet_b4_256_mvtec_strict.py \
    --work-dir runs/uniad \
    --cfg-options train_dataloader.dataset.multi_class=True
```

### Override Training Parameters

```bash
python tools/train.py <config> --work-dir runs/test \
    --cfg-options \
    train_dataloader.batch_size=16 \
    optim_wrapper.optimizer.lr=0.001 \
    train_cfg.max_epochs=50
```

### Force CPU Training

```bash
python tools/train.py <config> --work-dir runs/test --cpu
```

:::{warning}
CPU training is extremely slow and not recommended for most methods.
:::

### Resume Training

Resume from the latest checkpoint:

```bash
python tools/train.py <config> --work-dir runs/patchcore --resume
```

Load from a specific checkpoint (without resuming the optimizer state):

```bash
python tools/train.py <config> --work-dir runs/patchcore \
    --cfg-options load_from='runs/patchcore/epoch_10.pth'
```

## Testing

### Basic Testing

Test with a trained checkpoint:

```bash
python tools/test.py configs/patchcore/patchcore_wrn50_256_mvtec_strict.py \
    runs/patchcore/best.pth
```

### Test with Override

```bash
python tools/test.py <config> runs/patchcore/best.pth \
    --cfg-options test_dataloader.dataset.cls_names="['bottle']"
```

## Memory Bank Methods

Methods like PatchCore, SPADE, PaDiM, DFM, DFKDE, and RegAD use a two-phase workflow:

1. **Phase 1**: Train (for memory bank methods, this is often just feature extraction)
2. **Phase 2**: Build memory bank via `MemoryBankHook`

The `MemoryBankHook` is automatically registered in `configs/_base_/default_runtime.py` and calls `model.build_memory_bank()` or `model.fit()` after training completes and before validation.

No special action is needed -- the hook handles the lifecycle automatically.

## Training Output

During training, BaoIAD logs metrics with the `ad/` prefix:

```
ad/image_auroc: 0.9523
ad/image_f1max: 0.8912
ad/pixel_auroc: 0.9781
ad/bottle/image_auroc: 0.9834
ad/cable/image_auroc: 0.9210
...
```

- **Averaged metrics**: `ad/<metric>: <value>` (averaged across categories)
- **Per-category metrics**: `ad/<category>/<metric>: <value>`

## Work Directory Structure

```
runs/patchcore/
├── <timestamp>.log         # Training log
├── <timestamp>.log.json    # JSON-format log
├── vis_data/               # Visualization data
├── best.pth                # Best checkpoint (by val metric)
├── epoch_10.pth            # Periodic checkpoint
└── last_checkpoint         # Path to latest checkpoint
```
