# 数据流

## ADDataSample 字段

| 字段 | 类型 | 设置者 | 描述 |
|------|------|--------|------|
| `gt_label` | int | 数据集 | 真值标签（0=正常，1=异常） |
| `gt_mask` | Tensor (H, W) | 数据集 | 真值像素掩码 |
| `cls_name` | str | 数据集 | 产品类别名 |
| `img_path` | str | 数据集 | 输入图像路径 |
| `defect_type` | str | 数据集 | 缺陷类型名 |
| `pred_score` | float | 检测器 predict() | 预测图像级异常分数 |
| `pred_anomaly_map` | Tensor (1, H, W) | 检测器 predict() | 预测像素级异常图 |

## 标准数据管线

```
LoadImage → ResizeAD → NormalizeAD → PackADInputs
```

## build_predict_results

使用 `baoiad.models.predict_utils` 中的 `build_predict_results()` 统一构建预测输出：

```python
from baoiad.models.predict_utils import build_predict_results

def predict(self, batch_inputs, data_samples):
    feats = self.extract_feat(batch_inputs)
    anomaly_map = self.compute_anomaly_map(feats)
    return build_predict_results(data_samples=data_samples, anomaly_map=anomaly_map)
```
