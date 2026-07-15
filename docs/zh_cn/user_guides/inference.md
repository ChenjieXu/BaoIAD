# 推理

## 使用 tools/test.py

```bash
python tools/test.py configs/patchcore/patchcore_wrn50_256_mvtec_strict.py \
    runs/patchcore/best.pth
```

## 编程式推理

```python
from baoiad import register_all_modules
from mmengine.config import Config
from mmengine.runner import Runner

# 应用入口建议显式注册。Registry.get/build 仍支持懒加载，
# 因此未显式调用该函数的旧编程式用法仍可按需解析组件。
register_all_modules()

cfg = Config.fromfile('configs/patchcore/patchcore_wrn50_256_mvtec_strict.py')
runner = Runner.from_cfg(cfg)
runner.load_checkpoint('runs/patchcore/best.pth')
runner.test()
```

## 解读结果

| 字段 | 类型 | 描述 |
|------|------|------|
| `pred_score` | float | 预测异常分数（越高越异常） |
| `pred_anomaly_map` | Tensor (1, H, W) | 预测像素级异常图 |
