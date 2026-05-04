# 训练与测试

## 训练

### 基本训练

```bash
python tools/train.py configs/patchcore/patchcore_wrn50_256_mvtec_strict.py \
    --work-dir runs/patchcore
```

### 单类别训练

```bash
python tools/train.py configs/patchcore/patchcore_wrn50_256_mvtec_strict.py \
    --work-dir runs/patchcore_bottle \
    --cfg-options \
    train_dataloader.dataset.cls_names="['bottle']" \
    train_dataloader.dataset.multi_class=False
```

## 测试

```bash
python tools/test.py configs/patchcore/patchcore_wrn50_256_mvtec_strict.py \
    runs/patchcore/best.pth
```

## 记忆库方法

PatchCore、PaDiM 等方法使用两阶段工作流：

1. **阶段 1**：训练（对记忆库方法而言，这通常只是特征提取）
2. **阶段 2**：通过 `MemoryBankHook` 构建记忆库

`MemoryBankHook` 在训练完成后、验证之前自动调用 `model.build_memory_bank()` 或 `model.fit()`。
