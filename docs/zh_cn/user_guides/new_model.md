# 添加新检测器

## 步骤

1. 在 `baoiad/models/detectors/` 中创建检测器文件，扩展合适的 BaseADModel 子类
2. 使用 `@MODELS.register_module()` 装饰器注册
3. 在 `baoiad/models/detectors/__init__.py` 中添加导入
4. 在 `configs/my_method/` 中创建配置文件，继承 `_base_` 配置
5. 在 `tests/test_models/test_detectors/` 中添加测试
6. 在应用或测试入口显式调用 `baoiad.register_all_modules()`；Registry 也会在 `get/build` 时懒加载对应模块

## 选择基类

| 基类 | 适用场景 |
|------|----------|
| `MemoryBankADModel` | 特征匹配（kNN、coreset） |
| `KnowledgeDistillationADModel` | 师生网络差异 |
| `FlowBasedADModel` | 特征上的归一化流 |
| `ReconstructionADModel` | 自编码器/重建 |
| `VisionLanguageADModel` | 基于 CLIP 的零/少样本 |
| `DiscriminatorADModel` | 特征判别 |
| `BaseADModel` | 以上均不适用 |
