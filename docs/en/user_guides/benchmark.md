# Benchmark

BaoIAD provides `tools/benchmark.py` to train and evaluate methods across categories in a single sweep. It orchestrates per-method subprocess runs, parses metrics from the output, and writes a structured JSON results file.

The method inventory is sourced from [`baoiad/method_inventory.py`](../../baoiad/method_inventory.py) — `--methods all` selects those 37 repo-local method slugs, not the contents of the `configs/` directory.

## Quick Start

```bash
# Benchmark PatchCore and RD on all MVTec AD categories
python tools/benchmark.py \
    --data_root data/mvtec_ad \
    --categories all \
    --methods patchcore rd

# Benchmark all 37 methods on MVTec AD
CUDA_VISIBLE_DEVICES=0 python tools/benchmark.py \
    --data_root data/mvtec_ad \
    --categories all \
    --methods all

# Benchmark a single method on one category
python tools/benchmark.py \
    --data_root data/mvtec_ad \
    --categories bottle \
    --methods patchcore
```

## CLI Reference

```
python tools/benchmark.py [OPTIONS]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--data_root` | str (required) | — | Dataset root directory (e.g. `data/mvtec_ad`). Must contain per-category subdirectories with `train/` and `test/` splits. |
| `--categories` | list[str] | `['bottle']` | Category names to evaluate, or `all`. When `all`, auto-detects MVTec AD vs VisA categories based on which subdirectories exist. |
| `--methods` | list[str] | `['patchcore', 'rd']` | Method slugs to evaluate, or `all`. When `all`, loads all 37 method slugs from the inventory. If unset but `--config` is provided, derives the method name from the config path. |
| `--device` | str | `cuda` | Device for training/evaluation. Use `cpu` for CPU-only runs. For GPU selection, set `CUDA_VISIBLE_DEVICES` externally. |
| `--epochs` | int | `None` | Override `max_epochs` in the config. Ignored for iteration-based methods. When omitted, the config's own epoch count is used. |
| `--batch_size` | int | `None` | Override batch size for train, test, and val dataloaders. When omitted, the config's own value is used. |
| `--timeout` | int | `3600` | Per-run timeout in seconds. If a single method+category run exceeds this, the process is killed and the result is recorded as a timeout error. |
| `--config` | str | `None` | Override the config path. Useful for running a specific config file rather than auto-discovering one for the method. |
| `--output` | str | `results/benchmark.json` | Path to the output JSON file. Created if it does not exist. Results are saved incrementally after each method+category run. |
| `--work-dir-root` | str | `runs/benchmark/` | Root directory under which per-method, per-category work directories are created (e.g. `runs/benchmark/patchcore/bottle/`). |
| `--cfg-options` | list[str] | `None` | Extra config overrides forwarded to `tools/train.py`. Format: `key=value`. These are appended after all benchmark-injected overrides. |

## How It Works

1. **Method resolution**: For each method slug, `benchmark.py` locates a config file under `configs/<method>/`. It prefers configs with `mvtec` and `strict` in the filename, with a per-method priority list for disambiguation.
2. **Category loop**: For each method, the runner iterates over the requested categories. Multi-class methods (flagged by `benchmark_multi_class=True` in their config) run once across all categories.
3. **Subprocess execution**: Each method+category pair is launched as a subprocess via `tools/train.py` (or a method-specific training script). The benchmark runner injects cfg-options to set `data_root`, category, worker count, and checkpoint hooks.
4. **Metric parsing**: The runner parses metrics (AUROC, F1-max, AP, AUPRO, etc.) from the subprocess stdout/stderr. Configs can declare a `benchmark_result_selector` to control which snapshot is used (e.g. `mode='best'` for best-epoch selection).
5. **Incremental save**: Results are written to the output JSON after each run, so partial results survive crashes.

## Category Auto-Detection

When `--categories all` is passed, the script checks whether any VisA-specific category names (e.g. `candle`, `cashew`, `pcb1`) have a `train/` subdirectory under `--data_root`. If so, it uses the full VisA category list; otherwise it defaults to the 15 MVTec AD categories.

## Output Format

The output JSON is a nested dictionary:

```json
{
  "patchcore": {
    "bottle": {
      "image_auroc": 0.9872,
      "pixel_auroc": 0.9765,
      "image_f1max": 0.9521,
      "aupro": 0.9234,
      "time": 12.3
    },
    "_average": {
      "image_auroc": 0.9512,
      "pixel_auroc": 0.9345,
      "num_categories": 15
    }
  }
}
```

Each method contains per-category metric dictionaries plus an `_average` entry with mean metrics across categories. Failed runs contain an `error` key instead of metrics.

## Config-Level Overrides

Configs can declare benchmark-specific directives that alter how the benchmark runner handles them:

| Config Key | Effect |
|------------|--------|
| `benchmark_multi_class=True` | Run once with all categories instead of per-category |
| `benchmark_keep_train_data_root=True` | Preserve the config's train `data_root` (for methods that train on auxiliary data) |
| `benchmark_keep_dataloader_workers=True` | Preserve the config's worker count instead of clamping to 0 |
| `benchmark_preserve_checkpoint_hooks=True` | Keep checkpoint saving enabled |
| `benchmark_eval_only=True` | Run via `tools/test.py` instead of training |
| `benchmark_test_after_train=True` | Run `tools/test.py` with the trained checkpoint after training completes |
| `benchmark_resume_existing=True` | Pass `--resume` to the training subprocess |
| `benchmark_rescale_epoch_schedulers=True` | Rescale scheduler milestones when epoch count is overridden |
| `benchmark_disable_compile=True` | Set `TORCH_COMPILE_DISABLE=1` for the subprocess |
| `benchmark_timeout=7200` | Declare a minimum per-run timeout (CLI timeout is used if larger) |
| `benchmark_result_selector=dict(mode='best', metric='image_auroc')` | Select which training snapshot to report |
| `benchmark_train_script='tools/train_ast.py'` | Use a method-specific training script |
| `benchmark_categories=['bottle','cable',...]` | Restrict which categories this config supports |
| `benchmark_summary_categories=[...]` | Subset of categories used for official average computation |
| `benchmark_checkpoint_source='best'` | Use best checkpoint instead of last for post-training evaluation |

## Examples

### Benchmark all methods on MVTec AD

```bash
python tools/benchmark.py \
    --data_root data/mvtec_ad \
    --categories all \
    --methods all \
    --timeout 7200 \
    --output results/mvtec_all.json
```

### Benchmark specific methods on VisA

```bash
python tools/benchmark.py \
    --data_root data/visa \
    --categories all \
    --methods patchcore rd efficientad \
    --output results/visa_subset.json
```

### Custom config with epoch override

```bash
python tools/benchmark.py \
    --data_root data/mvtec_ad \
    --categories all \
    --config configs/patchcore/patchcore_wrn50_256_mvtec_strict.py \
    --epochs 1 \
    --output results/patchcore_smoke.json
```

### CPU-only smoke test

```bash
python tools/benchmark.py \
    --data_root data/mvtec_ad \
    --categories bottle \
    --methods rd \
    --device cpu \
    --epochs 1
```
