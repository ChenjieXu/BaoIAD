# Frequently Asked Questions

## What is BaoIAD?

BaoIAD is an industrial anomaly detection (IAD) benchmark toolbox built on MMEngine. Its repository inventory contains **37 method integrations** across **9 families**, with configuration entry points for multiple datasets. Runtime prerequisites and validation depth vary by method; consult the method-status manifest instead of assuming uniform reproducibility.

The method inventory lives in [`baoiad/method_inventory.py`](../../../baoiad/method_inventory.py). The 9 method families are:

| Family | Methods |
|--------|---------|
| Feature-memory / density | PatchCore, PaDiM, DFM, DFKDE |
| Knowledge distillation | RD, RD++, AST, EfficientAD, DeSTSeg |
| Normalizing flow | FastFlow, CFlow, DifferNet, U-Flow, PyramidFlow |
| Reconstruction / ViT | Dinomaly, ViTAD, MemSeg, UniAD, GANomaly |
| Self-supervised synthesis | DRAEM, GLASS, DSR, CutPaste, NSA |
| Discriminative | SimpleNet, SuperSimpleNet, CFA |
| Vision-language / foundation | WinCLIP, AnomalyCLIP, MuSc, AACLIP, AnoVL, AdaCLIP, SAA+ |
| Few-shot / registration | AnomalyDINO, RegAD |
| Hybrid / unified | UniNet |

## What is the difference between strict and unified configs?

BaoIAD provides per-dataset configuration entry points. The `_strict` suffix (for example, `patchcore_wrn50_256_mvtec_strict.py`) identifies a reference-oriented configuration; it is a naming convention, not proof that every upstream hyperparameter, runtime path, or published result is reproduced. Unified configs standardize selected repository settings, but do not by themselves establish cross-method comparability.

Use reference-oriented configs when investigating method-specific settings, and inspect the corresponding provenance record and limitations before interpreting results. Use unified configs only when the standardized choices are suitable for the comparison you intend to make.

## How many methods are included?

**37 methods**. See the full list in [`baoiad/method_inventory.py`](../../../baoiad/method_inventory.py). Each entry has a slug, display name, family, config paths, and links to its config README and provenance record.

## What is alignment evidence?

Each method has a provenance and reproducibility record in [`docs/alignment/`](../../alignment/) that documents the public source, implementation differences, runtime state, and known limitations.

- **Public source**: the paper and implementation source recorded by the release inventory
- **Implementation differences**: the repository adapters or deviations known at release time
- **Runtime state**: optional dependencies, external artifacts, or network behavior
- **Limitations**: the method-specific boundary for historical or partial validation

See [`docs/alignment/README.md`](../../alignment/README.md) for the full table of methods and links.

## What does `--methods all` mean?

```bash
python tools/benchmark.py --data_root data/mvtec_ad --methods all
```

This selects all 37 method slugs from `baoiad/method_inventory.py` and runs each one against the requested dataset categories. You can also pass specific slugs:

```bash
python tools/benchmark.py --data_root data/mvtec_ad --methods patchcore rd efficientad --categories all
```

## How do I run on a single category?

Use `--cfg-options` to override the category, or use a specific config file that targets one category:

```bash
python tools/train.py configs/patchcore/patchcore_wrn50_256_mvtec_strict.py \
    --cfg-options train_dataloader.dataset.categories='[bottle]'
```

## How do I choose between MVTec and VisA configs?

Each method directory under `configs/` contains separate configs for MVTec AD (`_mvtec_strict.py`) and VisA (`_visa.py`). Choose based on your target dataset:

- **MVTec AD** configs: intended to follow method-specific reference settings; consult the method-status manifest and record for validation limits
- **VisA** configs: repository-standardized settings; suitability for a cross-method comparison still depends on method prerequisites and protocol choices

Point your data root to the appropriate dataset via `BAOIAD_DATA_ROOT` or `--cfg-options`.

## How do I run a benchmark workflow?

1. Install the core package and the extras required by the selected methods. The `[all]` extra covers common optional families but does not guarantee every external artifact or platform-specific dependency:
   ```bash
   pip install -e ".[all]"
   ```

2. Set the data root:
   ```bash
   export BAOIAD_DATA_ROOT=/path/to/data
   ```

3. Run the benchmark for specific methods:
   ```bash
   python tools/benchmark.py --data_root data/mvtec_ad --methods patchcore rd --categories all
   ```

4. Or run all 37 methods:
   ```bash
   python tools/benchmark.py --data_root data/mvtec_ad --methods all --categories all
   ```

Results will be saved under the `runs/` directory by default.

## How do I run the speed benchmark?

Use `tools/benchmark_speed.py` (see [Speed Benchmark](../user_guides/speed_benchmark.md) for details):

```bash
python tools/benchmark_speed.py --methods patchcore --data-root data/mvtec_ad --output results/speed_patchcore.json
```

This measures inference latency and throughput for each method.

## Can I use BaoIAD for custom datasets?

Yes. BaoIAD supports custom datasets through the MMEngine dataset registry. See [Add a Custom Dataset](../advanced_guides/add_custom_dataset.md) for step-by-step instructions. In short:

1. Create a dataset class registered under `baoiad.datasets`
2. Write a config that points to your data directory
3. Run training and evaluation as usual

## What metrics does BaoIAD compute?

The [`AnomalyDetectionMetric`](../../../baoiad/evaluation/ad_metric.py) class computes the following:

**Image-level:**
- `image_auroc` — Area Under the ROC Curve
- `image_f1max` — Maximum F1 score over all thresholds
- `image_ap` — Average Precision
- `image_ece` — Expected Calibration Error
- `image_fpr@95tpr` — False Positive Rate at 95% True Positive Rate

**Pixel-level:**
- `pixel_auroc` — Area Under the ROC Curve
- `pixel_f1max` — Maximum F1 score over all thresholds
- `pixel_ap` — Average Precision
- `aupro` — Area Under the Per-Region Overlap curve
- `aupimo` — Area Under the Per-Image Mean Overlap curve
- `pixel_ece` — Expected Calibration Error

All metrics are computed per-class and then averaged across classes.
