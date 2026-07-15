# CLI Reference

Complete reference for all scripts in the `tools/` and `scripts/` directories.

## Core Scripts

### tools/train.py

Train an anomaly detector.

```bash
python tools/train.py <config> [--work-dir DIR] [--resume] [--cpu] [--cfg-options KEY=VAL ...]
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `config` | positional | required | Training config file path. |
| `--work-dir` | `str` | config value | Working directory for checkpoints and logs. |
| `--resume` | flag | `False` | Resume from `<work_dir>/last_checkpoint`. |
| `--cpu` | flag | `False` | Force CPU-only execution. Disables CUDA and MPS. |
| `--cfg-options` | `DictAction` | none | Override config options as `key=value` pairs. |

See [Training and Testing](train_test.md) for detailed usage.

### tools/test.py

Test an anomaly detector.

```bash
python tools/test.py <config> [checkpoint] [--work-dir DIR] [--cpu] [--cfg-options KEY=VAL ...]
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `config` | positional | required | Test config file path. |
| `checkpoint` | positional | `None` | Optional checkpoint file path. Overrides `load_from` in config. |
| `--work-dir` | `str` | config value | Working directory for test results. |
| `--cpu` | flag | `False` | Force CPU-only execution. |
| `--cfg-options` | `DictAction` | none | Override config options as `key=value` pairs. |

See [Training and Testing](train_test.md) for detailed usage.

### tools/benchmark.py

Unified benchmark runner. Trains and evaluates methods across categories.

```bash
python tools/benchmark.py --data_root <path> --methods <names> --categories <cats> [options]
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--data_root` | `str` | required | Dataset root directory. |
| `--categories` | `nargs='+'` | `['bottle']` | Categories to test, or `all` for all MVTec AD / VisA categories. |
| `--methods` | `nargs='+'` | `['patchcore', 'rd']` | Methods to benchmark, or `all` for all 37 repo methods. |
| `--device` | `str` | `cuda` | Compute device (`cuda` or `cpu`). |
| `--epochs` | `int` | `None` | Override `max_epochs`. Ignored for iter-based methods. |
| `--batch_size` | `int` | `None` | Override batch size. Uses config value if not set. |
| `--timeout` | `int` | `3600` | Timeout per run in seconds. |
| `--config` | `str` | `None` | Explicit config path (use with a single method). |
| `--output` | `str` | `results/benchmark.json` | Output JSON file path. |
| `--work-dir-root` | `str` | `runs/benchmark` | Root for per-method per-category work directories. |
| `--cfg-options` | `nargs='+'` | `None` | Extra config options forwarded to the training script. |

**Examples**:

```bash
# Run PatchCore and RD on all MVTec AD categories
python tools/benchmark.py --data_root data/mvtec_ad --categories all --methods patchcore rd

# Run all 37 methods on all categories
python tools/benchmark.py --data_root data/mvtec_ad --categories all --methods all

# Override timeout for heavy methods
python tools/benchmark.py --data_root data/mvtec_ad --categories all --methods efficientad --timeout 7200

# CPU-only run
python tools/benchmark.py --data_root data/mvtec_ad --methods patchcore --device cpu

# Custom output location
python tools/benchmark.py --data_root data/mvtec_ad --methods all --output results/my_benchmark.json
```

### tools/benchmark_speed.py

Benchmark inference speed (latency and FPS) for trained models using real MVTec AD images.

```bash
python tools/benchmark_speed.py --methods <names> --output <path> [options]
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--methods` | `str` | required | Comma-separated method names (e.g. `patchcore,padim,rd`). |
| `--gpu` | `int` | `0` | GPU device index. |
| `--data-root` | `str` | `data/mvtec_ad` | Dataset root for loading test images. |
| `--output` | `str` | required | Output JSON file path. |
| `--warmup` | `int` | `10` | Number of warmup iterations per image before timing. |
| `--runs` | `int` | `100` | Number of timed forward passes per image. |

**Example**:

```bash
python tools/benchmark_speed.py --methods patchcore,padim,rd --gpu 0 --output results/speed.json
```

Output includes per-method average latency (ms/img), standard deviation, and FPS.

## Special Training Scripts

### tools/train_ast.py

Two-stage AST (Asymmetric Student-Teacher) training.

```bash
python tools/train_ast.py <config> [--work-dir DIR] [--resume] [--cpu] [--cfg-options KEY=VAL ...]
```

Arguments are identical to `tools/train.py`. The script runs teacher training first, then student training with the teacher checkpoint. See [Training and Testing](train_test.md#tools-train_ast-py----two-stage-ast-training) for details.

### tools/train_regad_strict.py

RegAD few-shot training with the official protocol.

```bash
python tools/train_regad_strict.py <config> [--work-dir DIR] [--resume] [--cpu] [--cfg-options KEY=VAL ...]
```

Arguments are identical to `tools/train.py`. Uses a custom training loop (not MMEngine Runner) for strict alignment with the official RegAD code. See [Training and Testing](train_test.md#tools-train_regad_strict-py----regad-few-shot-protocol) for details.

### tools/train_vitad_exact_order.py

ViTAD replay with a user-supplied, verified per-epoch sample order.

```bash
python tools/train_vitad_exact_order.py <config> --order-file ORDER.json [--work-dir DIR] [--resume] [--cpu] [--cfg-options KEY=VAL ...]
```

BaoIAD does not generate or distribute the official order artifact. The script exits with an error when the verified JSON is absent, then replays the supplied file through `PerEpochOrderSampler`. See [Training and Testing](train_test.md#tools-train_vitad_exact_order-py----vitad-verified-sample-order-replay) for details.

## Utility Scripts

### tools/check_method_inventory.py

Validates the repo-local method inventory and documentation layout. Checks that all 37 methods have valid configs, READMEs, alignment records, and required README sections.

```bash
python tools/check_method_inventory.py
```

No arguments. Exits with code 0 on success, 1 on failure. Outputs `PASS` or `FAIL` with detailed error messages.

### tools/env.sh

Environment configuration script. Source before running BaoIAD tools:

```bash
source tools/env.sh
```

Sets the following environment variables:

| Variable | Default | Description |
|---|---|---|
| `BAOIAD_DATA_ROOT` | `<repo>/data` | Root directory for datasets. |
| `HF_HOME` | `~/.cache/huggingface` | HuggingFace cache directory. |
| `TORCH_HOME` | `~/.cache/torch` | PyTorch model cache directory. |
| `BAOIAD_USE_MIRROR` | unset | Set to `1` to use HuggingFace mirror. |
| `BAOIAD_CACHE_DIR` | unset | Custom cache directory for model weights. |

Override variables by setting them before sourcing:

```bash
export BAOIAD_DATA_ROOT=/data/datasets
source tools/env.sh
```

### tools/smoke_test_gpu.sh

Quick validation that all 37 method configs exist and are discoverable.

```bash
bash tools/smoke_test_gpu.sh [method1 method2 ...]
```

Without arguments, tests all 37 methods. With arguments, tests only the named methods. Outputs `ok` for each method that has a discoverable config, exits on first failure.

### tools/smoke_test_remaining.sh

Validates configs for the remaining methods not yet tested, using the same inventory-based discovery.

```bash
bash tools/smoke_test_remaining.sh
```

No arguments. Checks that each method has a discoverable config file. Writes a summary to `runs/smoke_test_summary.txt`.
