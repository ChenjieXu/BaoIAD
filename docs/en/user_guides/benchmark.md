# Benchmark

BaoIAD provides a benchmarking tool that automates running experiments across methods and categories, collecting results into a structured JSON summary.

## Basic Usage

```bash
python tools/benchmark.py \
    --data_root data/mvtec_ad \
    --methods patchcore rd \
    --categories all \
    --output runs/benchmark_results.json
```

This will:

1. Auto-detect the config file for each method
2. Spawn one subprocess per (method, category) pair
3. Parse `ad/<metric>: <value>` from stdout/stderr
4. Write a JSON summary to the output path

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | `data/mvtec_ad` | Path to dataset root |
| `--methods` | Required | Space-separated list of method names |
| `--categories` | `all` | Categories to benchmark (`all` or comma-separated list) |
| `--config` | Auto-detected | Override config path for a single method |
| `--output` | `runs/benchmark.json` | JSON output path |
| `--timeout` | `3600` | Timeout per (method, category) run in seconds |
| `--epochs` | None | Override number of training epochs |
| `--batch_size` | None | Override batch size |
| `--gpus` | `0` | Comma-separated GPU IDs to use |

## Config Auto-Detection

The benchmark tool automatically finds config files by convention:

1. Looks in `configs/<method>/` for files matching `<method>_*.py`
2. Prefers configs containing `256` and `mvtec` in the filename
3. Excludes `unified` and `.bak` variants

Override with `--config` for explicit control:

```bash
python tools/benchmark.py \
    --data_root data/mvtec_ad \
    --methods patchcore \
    --config configs/patchcore/patchcore_wrn50_256_mvtec_strict.py \
    --categories bottle cable \
    --timeout 7200
```

## Override Epochs and Batch Size

```bash
python tools/benchmark.py \
    --data_root data/mvtec_ad \
    --methods rd \
    --categories all \
    --epochs 50 \
    --batch_size 16 \
    --timeout 10800
```

## Multi-GPU Benchmarking

Use `tools/run_benchmark.py` for multi-GPU support:

```bash
# Run benchmarks distributed across GPUs 0,1,2,3
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/run_benchmark.py \
    --methods patchcore rd padim stfpm \
    --data_root data/mvtec_ad \
    --output runs/multi_gpu_benchmark.json
```

## JSON Output Format

The benchmark output is a JSON file with this structure:

```json
{
    "benchmark_info": {
        "timestamp": "2025-01-15T10:30:00",
        "data_root": "data/mvtec_ad",
        "methods": ["patchcore", "rd"],
        "categories": ["bottle", "cable", ...]
    },
    "results": {
        "patchcore": {
            "bottle": {
                "image_auroc": 0.9923,
                "pixel_auroc": 0.9812,
                "image_f1max": 0.9750,
                ...
            },
            ...
        },
        "rd": {
            ...
        }
    },
    "summary": {
        "patchcore": {
            "mean_image_auroc": 0.9523,
            "mean_pixel_auroc": 0.9781,
            ...
        },
        ...
    }
}
```

## Alignment Results

Alignment results (reproducing published benchmarks) are stored in `runs/alignment/`. Per-method alignment notes are documented in `docs/alignment/`.

## Benchmarking Tips

- **Timeout**: Set generous timeouts for methods with long training (e.g., DRAEM, DSR). Default is 3600s (1 hour) per run.
- **Memory bank methods**: PatchCore, SPADE, PaDiM are fast (feature extraction only), while reconstruction methods (DRAEM, DeSTSeg) are slower.
- **Seed sensitivity**: Some methods (PaDiM, DFKDE) are sensitive to random seed. The default seed is 42.
- **GPU selection**: Use `CUDA_VISIBLE_DEVICES` or `--gpus` to control GPU allocation.
