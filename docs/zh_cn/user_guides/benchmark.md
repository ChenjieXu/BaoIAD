# 基准测试

## 基本用法

```bash
python tools/benchmark.py \
    --data_root data/mvtec_ad \
    --methods patchcore rd \
    --categories all \
    --output runs/benchmark_results.json
```

## 命令行参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--data_root` | `data/mvtec_ad` | 数据集根目录路径 |
| `--methods` | 必需 | 空格分隔的方法名列表 |
| `--categories` | `all` | 要基准测试的类别 |
| `--config` | 自动检测 | 覆盖配置路径 |
| `--output` | `runs/benchmark.json` | JSON 输出路径 |
| `--timeout` | `3600` | 每次（方法, 类别）运行的超时秒数 |

## 多 GPU 基准测试

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/run_benchmark.py \
    --methods patchcore rd padim stfpm \
    --data_root data/mvtec_ad \
    --output runs/multi_gpu_benchmark.json
```
