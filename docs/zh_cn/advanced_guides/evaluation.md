# 评估

## AnomalyDetectionMetric

`AnomalyDetectionMetric` 计算 12 项指标：

### 图像级指标

| 指标 | 键 | 描述 |
|------|-----|------|
| AUROC | `image_auroc` | ROC 曲线下面积 |
| F1-max | `image_f1max` | 所有阈值下的最大 F1 分数 |
| 平均精度 | `image_ap` | 精确率-召回率曲线下面积 |
| ECE | `image_ece` | 期望校准误差 |
| FPR@95TPR | `image_fpr@95tpr` | 95% 真阳性率下的假阳性率 |

### 像素级指标

| 指标 | 键 | 描述 |
|------|-----|------|
| AUROC | `pixel_auroc` | ROC 曲线下面积（逐像素） |
| F1-max | `pixel_f1max` | 所有阈值下的最大 F1 分数 |
| 平均精度 | `pixel_ap` | 精确率-召回率曲线下面积 |
| AUPRO | `aupro` | 每区域重叠曲线下面积 |
| AUPIMO | `aupimo` | 每图像平均重叠曲线下面积 |
| ECE | `pixel_ece` | 期望校准误差（逐像素） |

### 指标日志

- **平均值**：`ad/<metric>: <value>`（跨类别平均）
- **每类别**：`ad/<category>/<metric>: <value>`
