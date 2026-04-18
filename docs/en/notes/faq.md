# FAQ

## Installation

### Q: How do I install MMCV with CUDA support?

A: Install from the pre-built wheel index matching your CUDA and PyTorch versions:

```bash
pip install mmcv -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
```

Replace `cu118` and `torch2.0` with your CUDA toolkit and PyTorch versions.

### Q: FrEIA import fails when using flow methods.

A: Install the flow optional dependency:

```bash
pip install -e ".[flow]"
```

This installs `FrEIA>=0.2`, which is required by CSFlow, FastFlow, CFlow, UFlow, DifferNet, PyramidFlow, and AST.

### Q: open_clip import fails when using VL methods.

A: Install the VL optional dependency:

```bash
pip install -e ".[vl]"
```

## Training

### Q: Memory bank methods (PatchCore, SPADE) seem to train very fast -- is that normal?

A: Yes. Memory bank methods only perform feature extraction during "training." The actual memory bank is built by `MemoryBankHook` after the training loop. Training may complete in seconds, but memory bank construction can take longer.

### Q: How do I train on a specific category?

A: Override the category list and set `multi_class=False`:

```bash
python tools/train.py <config> --work-dir runs/bottle \
    --cfg-options train_dataloader.dataset.cls_names="['bottle']" train_dataloader.dataset.multi_class=False
```

### Q: Training runs out of memory. What can I do?

A: Several options:
- Reduce batch size: `--cfg-options train_dataloader.batch_size=8`
- Use gradient accumulation: `--cfg-options optim_wrapper.accumulative_counts=4`
- Reduce image size: `--cfg-options train_dataloader.dataset.pipeline.1.size=224`
- Force CPU (not recommended): `--cpu`

### Q: DRAEM/DeSTSeg training fails with DTD path error.

A: These methods require the DTD (Describable Textures Dataset) for anomaly synthesis. Set `dtd_path='auto'` in the model config or provide an explicit path:

```python
model = dict(
    type='DRAEM',
    dtd_path='/path/to/dtd',  # or 'auto' for automatic download
)
```

## Evaluation

### Q: What is the difference between AUPRO and AUPIMO?

A: AUPRO computes per-region overlap, which is sensitive to large connected anomaly regions. AUPIMO computes per-image mean overlap, which is more robust to images with many small regions. AUPIMO is generally preferred for datasets with diverse anomaly sizes.

### Q: Pixel AUROC is very high but the anomaly maps look poor. Why?

A: Pixel AUROC can be misleadingly high when anomalous pixels are sparse (the vast majority of pixels are normal, inflating the true negative rate). Use AUPRO or AUPIMO for a more informative pixel-level evaluation.

### Q: How do I compute FPR@95TPR?

A: `image_fpr@95tpr` is included in the default metrics computed by `AnomalyDetectionMetric`. No additional configuration is needed.

## Alignment

### Q: My PatchCore results don't match the original paper. What should I check?

A: Common alignment issues:
1. **Backbone `out_indices`**: PatchCore uses `(2, 3)` for WRN-50-2, corresponding to layer2+layer3 in timm's 0-indexed scheme
2. **Coreset ratio**: Default is 0.1 (10% of features)
3. **Number of neighbors**: Default is 9 (matching anomalib)
4. **Image scoring**: Should use the maximum of the post-processed anomaly map
5. **Seed**: Set `randomness.seed=42` for reproducibility

### Q: PaDiM results vary between runs. Why?

A: PaDiM uses random projection for dimensionality reduction, making it sensitive to the random seed. Ensure `randomness.seed=42` is set consistently.

### Q: RD image-level AUROC is lower than expected. What should I check?

A: Check the cosine loss implementation. Some reference implementations use spatial cosine (per-pixel) while others use flattened cosine. BaoIAD's strict configs align with the original paper's implementation.

## Methods

### Q: Which methods support multi-class training?

A: UniAD, ViTAD, InvAD, and MambaAD support training a single model on all categories (`multi_class=True`). Most other methods train one model per category.

### Q: Which methods support zero-shot inference?

A: Vision-language methods (WinCLIP, AnomalyCLIP, AnoVL, AnomalyDINO) support zero-shot inference without any training data. AdaCLIP and AACLIP require minimal training.

### Q: How do I use SAA/SAA+?

A: SAA+ requires `groundingdino` and `segment_anything`:

```bash
pip install -e ".[saa]"
```

These are optional dependencies and may require manual installation from source.
