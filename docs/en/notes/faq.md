# Frequently Asked Questions

## What is BaoIAD?

BaoIAD is a unified industrial anomaly detection (IAD) benchmark built on MMEngine. It provides **37 methods** across **9 families**, evaluated on multiple datasets (MVTec AD, VisA, and others) with a standardized training, testing, and evaluation pipeline. Every method ships with strict-aligned configs, frozen reference evidence, and reproducible benchmark scripts.

The method inventory lives in [`baoiad/method_inventory.py`](../../baoiad/method_inventory.py). The 9 method families are:

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

BaoIAD provides per-dataset configs for each method. For MVTec AD, configs are marked `_strict` (e.g., `patchcore_wrn50_256_mvtec_strict.py`), meaning they replicate the original paper's hyperparameters exactly. VisA configs use the BaoIAD unified settings (same backbone, resolution, and training schedule) for fair cross-method comparison.

Use strict configs when you need to reproduce original paper numbers. Use unified configs for fair cross-method benchmarking.

## How many methods are included?

**37 methods**. See the full list in [`baoiad/method_inventory.py`](../../baoiad/method_inventory.py). Each entry has a slug, display name, family, config paths, and links to its config README and alignment evidence.

## What is alignment evidence?

Each method has an alignment record in [`docs/alignment/`](../alignment/) that documents how the BaoIAD implementation matches the original paper. These records include:

- **Reference freeze**: the commit hash and source used for migration
- **Code-path parity check**: verified that forward/backward paths match the original
- **Behavior probes**: numerical spot-checks on intermediate outputs
- **Benchmark stop-line**: the point at which alignment was declared

See [`docs/alignment/README.md`](../alignment/README.md) for the full table of methods and links.

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

- **MVTec AD** configs: tuned for the original paper's settings (strict alignment)
- **VisA** configs: unified settings for fair comparison across methods

Point your data root to the appropriate dataset via `BAOIAD_DATA_ROOT` or `--cfg-options`.

## How do I reproduce benchmark results from the paper?

1. Install with all optional dependencies:
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

The [`AnomalyDetectionMetric`](../../baoiad/evaluation/ad_metric.py) class computes the following:

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
