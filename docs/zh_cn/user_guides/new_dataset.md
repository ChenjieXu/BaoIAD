# 添加新数据集

1. 在 `baoiad/datasets/` 中创建数据集类，扩展 `BaseADDataset`
2. 使用 `@DATASETS.register_module()` 装饰器注册
3. 在 `baoiad/datasets/__init__.py` 中添加导入
4. 在 `configs/_base_/datasets/` 中创建基础数据集配置
5. 创建继承新数据集配置的方法特定配置

## 期望的目录结构

```
data/my_dataset/
├── cat_a/
│   ├── train/
│   │   └── good/
│   └── test/
│       ├── good/
│       └── defect_type_1/
├── cat_b/
└── ...
```
