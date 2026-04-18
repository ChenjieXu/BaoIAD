# 记忆库生命周期

## MemoryBankHook

`MemoryBankHook` 在 `configs/_base_/default_runtime.py` 中声明，**始终处于活动状态**。它在训练完成后、验证之前自动调用相应方法。

### 钩子生命周期

```
训练循环
    │
    ├── epoch 1 ... epoch N（loss 模式）
    │
    ▼
训练完成后（首次验证前）
    │
    ├── model.build_memory_bank()    ← MemoryBankHook 调用
    │       或
    ├── model.fit()                  ← 备选（未定义 build_memory_bank 时）
    │
    ▼
验证 / 测试（predict 模式）
```

## 使用记忆库的方法

| 方法 | 重写方法 | 描述 |
|------|----------|------|
| PatchCore | `build_memory_bank()` | 提取特征，coreset 子采样，构建 FAISS 索引 |
| SPADE | `build_memory_bank()` | 存储所有正常特征及其像素位置 |
| PaDiM | `build_memory_bank()` | 对每个位置的特征拟合多元高斯分布 |
| DFM | `build_memory_bank()` | 对特征应用 PCA 变换 |
| DFKDE | `build_memory_bank()` | 对特征拟合核密度估计 |
| RegAD | `fit()` | 注册支持集特征用于少样本匹配 |
| GraphCore | `build_memory_bank()` | 构建图结构记忆库 |
