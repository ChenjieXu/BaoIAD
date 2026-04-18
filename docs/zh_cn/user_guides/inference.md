# 推理

## 使用 tools/test.py

```bash
python tools/test.py configs/patchcore/patchcore_wrn50_256_mvtec_strict.py \
    runs/patchcore/best.pth
```

## 编程式推理

```python
import baoiad  # 必须首先导入以触发注册器
from mmengine.config import Config
from mmengine.runner import Runner

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
