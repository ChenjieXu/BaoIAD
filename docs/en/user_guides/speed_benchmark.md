# Speed Benchmark

BaoIAD provides `tools/benchmark_speed.py` to measure per-image inference latency for one or more methods using real MVTec AD test images. For each method it loads the model, picks one normal and one anomalous image per MVTec category, runs a warmup phase, then times repeated forward passes.

## Quick Start

```bash
# Benchmark PatchCore and RD on GPU 0
python tools/benchmark_speed.py \
    --methods patchcore,rd \
    --gpu 0 \
    --data-root data/mvtec_ad \
    --output results/speed_patchcore_rd.json

# Benchmark several methods
python tools/benchmark_speed.py \
    --methods patchcore,rd,padim,efficientad,simplenet \
    --gpu 0 \
    --data-root data/mvtec_ad \
    --output results/speed_subset.json
```

## CLI Reference

```
python tools/benchmark_speed.py [OPTIONS]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--methods` | str (required) | — | Comma-separated method slugs (e.g. `patchcore,rd,efficientad`). No `all` shortcut — list the methods explicitly. |
| `--gpu` | int | `0` | CUDA device index for inference. |
| `--data-root` | str | `data/mvtec_ad` | Path to the MVTec AD dataset root. Must contain per-category subdirectories with `test/good/` and at least one defect subdirectory. |
| `--output` | str (required) | — | Path to write the JSON results file. Parent directories are created automatically. |
| `--warmup` | int | `10` | Number of warmup forward passes per image before timing. Ensures GPU kernels are compiled and caches are warm. |
| `--runs` | int | `100` | Number of timed forward passes per image. Latency is the mean across these runs. |

## How It Works

1. **Config discovery**: For each method slug, the script looks under `configs/<method>/` for a config matching `*mvtec_strict*.py` or `*mvtec*.py`.
2. **Model construction**: The model is built via `MODELS.build(cfg.model)` and moved to the specified GPU. Memory-bank models receive dummy bank data so that forward passes work without a training phase.
3. **Image collection**: For each of the 15 MVTec AD categories, one normal image (`test/good/000.png`) and one anomalous image (first defect's `000.png`) are loaded, resized to the config's `img_size`, and normalized. Vision-language methods use CLIP normalization; others use ImageNet normalization.
4. **Warmup + timing**: Each image gets `--warmup` untimed forward passes followed by `--runs` timed passes. GPU synchronization (`torch.cuda.synchronize`) brackets each pass for accurate measurement.
5. **Metrics**: Per-image latency (ms), standard deviation across images, and frames per second (FPS = 1000 / avg_ms) are computed.

## Output Format

The output JSON is a list of per-method result objects:

```json
[
  {
    "method": "patchcore",
    "img_size": 256,
    "avg_ms_per_img": 12.34,
    "std_ms": 2.10,
    "fps": 81.0,
    "n_images": 30,
    "warmup": 10,
    "runs": 100,
    "forward_mode": "predict"
  }
]
```

| Field | Description |
|-------|-------------|
| `method` | Method slug |
| `img_size` | Image resolution used for inference |
| `avg_ms_per_img` | Mean latency per image in milliseconds |
| `std_ms` | Standard deviation of per-image latency |
| `fps` | Frames per second (1000 / avg_ms_per_img) |
| `n_images` | Number of test images used (up to 30: 2 per category × 15 categories) |
| `warmup` | Warmup passes per image |
| `runs` | Timed passes per image |
| `forward_mode` | Forward mode used (`predict`, `tensor`, or `bare`) |

The script also prints a summary table sorted by FPS (descending) when finished.

## Hardware and Reproducibility

- **GPU required**: The script uses `torch.cuda.synchronize` for accurate timing and does not support CPU measurement.
- **Deterministic results**: Latency varies with GPU model, driver, CUDA version, and thermal state. For reproducible comparisons, run on the same hardware under similar load conditions.
- **Batch size**: The speed benchmark uses batch size 1 (one image per forward pass), matching common deployment scenarios.
- **Memory bank methods**: Methods that require a pre-computed memory bank (PatchCore, PaDiM, DFM, DFKDE, RegAD, AnomalyDINO) receive randomly initialized dummy banks. This measures inference-only latency, not the full training+inference pipeline.
- **Vision-language methods**: The benchmark detects VL model types and applies CLIP normalization. Pretrained weights may use a local cache; pass `--offline` when the required assets are already present and network access must be disabled.

## Examples

### Single method speed test

```bash
python tools/benchmark_speed.py \
    --methods patchcore \
    --gpu 0 \
    --data-root data/mvtec_ad \
    --output results/speed_patchcore.json
```

### Compare multiple methods with extended timing

```bash
python tools/benchmark_speed.py \
    --methods patchcore,rd,padim,cflow,fastflow \
    --gpu 0 \
    --data-root data/mvtec_ad \
    --warmup 20 \
    --runs 200 \
    --output results/speed_comparison.json
```

### Use a specific GPU on a multi-GPU machine

```bash
CUDA_VISIBLE_DEVICES=2 python tools/benchmark_speed.py \
    --methods efficientad,rdpp \
    --gpu 0 \
    --data-root data/mvtec_ad \
    --output results/speed_kd.json
```
