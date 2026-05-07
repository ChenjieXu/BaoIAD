# Troubleshooting

## Missing `torch` or `mmcv` installation errors

**Symptom:** `ModuleNotFoundError: No module named 'mmcv'` or `No module named 'torch'`.

**Solution:** Install the core dependencies:

```bash
pip install torch torchvision
pip install mmcv>=2.0
```

BaoIAD requires PyTorch >= 2.0 and mmcv >= 2.0. If you have a CUDA GPU, make sure the torch version matches your CUDA toolkit. See the [PyTorch install page](https://pytorch.org/get-started/locally/) for guidance.

## Missing `open_clip` errors (vision-language methods)

**Symptom:** `ModuleNotFoundError: No module named 'open_clip'` when running WinCLIP, AnomalyCLIP, MuSc, AACLIP, AnoVL, AdaCLIP, or SAA+.

**Solution:** Install the vision-language optional dependency:

```bash
pip install -e ".[vl]"
```

This installs `open_clip_torch`. Alternatively: `pip install open_clip_torch`.

## Missing `FrEIA` errors (normalizing flow methods)

**Symptom:** `ModuleNotFoundError: No module named 'FrEIA'` when running FastFlow, CFlow, DifferNet, U-Flow, or PyramidFlow.

**Solution:** Install the normalizing flow optional dependency:

```bash
pip install -e ".[flow]"
```

This installs `FrEIA>=0.2`. Alternatively: `pip install "FrEIA>=0.2"`.

## Missing `faiss` errors (DFKDE)

**Symptom:** `ModuleNotFoundError: No module named 'faiss'` when running DFKDE.

**Solution:** Install faiss:

```bash
pip install -e ".[faiss-cpu]"   # CPU version
# or
pip install -e ".[faiss-gpu]"   # GPU version (requires compatible CUDA)
```

Note: the GPU variant pins `numpy<2` and `faiss-gpu==1.7.2`.

## Missing `geomloss`, `imgaug`, or `mmpretrain` errors

**Symptom:** `ModuleNotFoundError` for one of these packages.

**Solution:** Each is an optional dependency group:

```bash
pip install -e ".[geomloss]"    # for methods using geomloss
pip install -e ".[imgaug]"      # for DRAEM and other synthesis methods
pip install -e ".[mmpretrain]"  # for methods using mmpretrain backbones
```

Or install everything at once:

```bash
pip install -e ".[all]"
```

## Dataset path errors: "data_root not found"

**Symptom:** `FileNotFoundError` or `AssertionError` about missing data root.

**Solution:** BaoIAD looks for datasets under `BAOIAD_DATA_ROOT`. Set it before running:

```bash
export BAOIAD_DATA_ROOT=/path/to/data
python tools/train.py configs/patchcore/patchcore_wrn50_256_mvtec_strict.py
```

Or use the provided env script:

```bash
source tools/env.sh
```

You can also override the data root per-run:

```bash
python tools/train.py <config> --cfg-options \
    train_dataloader.dataset.data_root=/path/to/data
```

## Dataset errors: masks not found

**Symptom:** Errors about missing ground-truth mask files during evaluation.

**Solution:** Verify your dataset directory structure matches what BaoIAD expects. Each category should have `good/` (normal images) and defective category folders with both images and corresponding mask PNGs in a `ground_truth/` or mask subdirectory. Check the specific dataset config for the expected layout.

## Memory bank not built errors

**Symptom:** `RuntimeError` or `KeyError` about missing memory bank when running feature-memory methods (PatchCore, PaDiM, DFM, DFKDE).

**Solution:** These methods require a training phase to build the memory bank before testing. Run training first:

```bash
python tools/train.py configs/patchcore/patchcore_wrn50_256_mvtec_strict.py --work-dir runs/patchcore
python tools/test.py configs/patchcore/patchcore_wrn50_256_mvtec_strict.py runs/patchcore/best.pth
```

## No visualization output

**Symptom:** Training or testing completes but no images are saved.

**Solution:** The visualization hook is **disabled by default** in [`configs/_base_/default_runtime.py`](../../configs/_base_/default_runtime.py). Enable it via config override:

```bash
python tools/test.py <config> <checkpoint> \
    --cfg-options default_hooks.visualization.enable=True
```

Or modify the config file directly:

```python
default_hooks = dict(
    visualization=dict(type='ADVisualizationHook', enable=True),
)
```

## Read the Docs build vs runtime environment confusion

**Symptom:** Docs build fails on Read the Docs due to missing dependencies like `torch` or `mmcv`.

**Solution:** The RTD build environment does not install the full `[all]` extras. Make sure `.readthedocs.yaml` is configured to install only the documentation dependencies, not the training packages. The docs configuration should use `pip install -e .` without extras or with a minimal `docs` extra that does not pull in torch/mmcv.

## CUDA out of memory

**Symptom:** `RuntimeError: CUDA out of memory` during training or inference.

**Solution:**

1. **Reduce batch size** in the config:
   ```bash
   python tools/train.py <config> --cfg-options \
       train_dataloader.batch_size=4
   ```

2. **Use a smaller input resolution** if the method allows it.

3. **Methods that are particularly memory-intensive** include vision-language methods (MuSc, AnomalyCLIP) due to large backbone models, and normalizing flow methods at high resolution. Consider using CPU-only mode or reducing `num_workers` for data loading.

4. For PatchCore and other memory-bank methods, the memory bank itself can consume significant GPU memory. Use `--cfg-options` to reduce the coreset sampling ratio.

## Config inheritance errors

**Symptom:** `KeyError` or `ConfigTypeError` when loading a config that uses `_base_`.

**Solution:** BaoIAD configs use MMEngine's config inheritance system. Common issues:

- Make sure you are running from the **repository root** (the directory containing `configs/` and `baoiad/`), not from inside a subdirectory.
- Check that all `_base_` paths are relative to the `configs/` directory.
- Run `python -c "from mmengine.config import Config; Config.fromfile('configs/patchcore/patchcore_wrn50_256_mvtec_strict.py')"` to validate a config without launching training.
