# 配置系统

BaoIAD 使用 MMEngine 的配置系统，支持继承机制，允许简洁的方法定义复用共享基础配置。

## 配置继承

配置可通过 `_base_` 字段继承一个或多个基础配置：

```python
# configs/patchcore/patchcore_wrn50_256_mvtec_strict.py
_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/mvtec_ad.py',
    '../_base_/schedules/schedule_100e.py',
]
```

## 运行时配置覆盖

使用 `--cfg-options` 从命令行覆盖任何配置字段：

```bash
python tools/train.py <config> --work-dir runs/test \
    --cfg-options \
    train_dataloader.batch_size=16 \
    train_dataloader.dataset.cls_names="['bottle']" \
    train_dataloader.dataset.multi_class=False
```

## Strict 与 Unified 配置

- **`*_strict.py`**：与原始论文实现对齐的参考配置
- **`*_unified.py`**：使用标准化 WRN-50-2 骨干网络的配置，用于公平跨方法比较
