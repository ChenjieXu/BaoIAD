# 自定义损失函数

## 内置损失

| 损失 | 注册名 | 常用方法 |
|------|--------|---------|
| CosineLoss | `CosineLoss` | RD、STFPM、EfficientAD |
| MSELoss | `MSELoss` | 特征匹配、重建 |
| L1Loss | `L1Loss` | 重建 |
| BCELoss | `BCELoss` | 分类方法 |
| CrossEntropyLoss | `CrossEntropyLoss` | 分类方法 |
| FocalLoss | `FocalLoss` | DRAEM |
| DiceLoss | `DiceLoss` | DSR、DeSTSeg |
| SSIMLoss | `SSIMLoss` | DRAEM、DSR |

## 添加自定义损失

1. 在 `baoiad/models/losses/` 中创建损失文件
2. 使用 `@MODELS.register_module()` 装饰器注册
3. 在 `baoiad/models/losses/__init__.py` 中添加导入
4. 在配置中使用 `dict(type='MyLoss', ...)` 引用
