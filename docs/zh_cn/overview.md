# 概述

BaoIAD 是一个基于 [MMEngine](https://github.com/open-mmlab/mmengine)（OpenMMLab 风格）构建的**工业异常检测（IAD）** 统一基准测试框架。它在单一的配置驱动接口下集成了 **50 多种异常检测方法**，支持在 MVTec AD、VisA、BTech 等多个数据集上进行公平比较。

## 主要特性

- **50+ 方法**：PatchCore、RD、Dinomaly、AnomalyCLIP、SimpleNet、DRAEM 等
- **统一接口**：配置驱动，MMEngine 风格的训练与评估
- **公平比较**：标准化骨干网络、一致的评估指标、可控随机种子
- **OpenMMLab 生态**：注册器、配置继承、模块化组件
- **全面的指标**：图像级（AUROC、F1-max、AP、ECE、FPR@95TPR）和像素级（AUROC、F1-max、AP、AUPRO、AUPIMO、ECE）
- **多数据集支持**：MVTec AD、VisA、BTech、MVTec 3D AD、MVTec LOCO、MPDD、MVTec AD 2、Kolektor、VAD、RealIAD

## 架构

BaoIAD 遵循 OpenMMLab 生态的 **backbone -> neck -> head** 管线模式：

```
输入图像
    |
    v
+----------+     +-------+     +------+
|  骨干网络  | --> |  颈部  | --> |  头部  |
+----------+     +-------+     +------+
    |                              |
    | (冻结)                        |
    v                              v
  特征                         损失 / 预测
```

### 核心组件

- **BaseADModel**：基类，支持 3 模式前向分发（`loss`、`predict`、`tensor`）
- **6 个专用子类**：MemoryBankADModel、KnowledgeDistillationADModel、FlowBasedADModel、ReconstructionADModel、VisionLanguageADModel、DiscriminatorADModel
- **ADDataSample**：承载真值与预测字段的数据结构
- **AnomalyDetectionMetric**：统一度量，计算 12 项图像/像素级指标
- **MemoryBankHook**：训练后自动构建记忆库的生命周期钩子

### 方法分类

| 类别 | 方法 | 基类 |
|------|------|------|
| 记忆库 | PatchCore、SPADE、PaDiM、DFM、DFKDE、RegAD、GraphCore | `MemoryBankADModel` |
| 知识蒸馏 | RD、RD++、STFPM、EfficientAD、Dinomaly | `KnowledgeDistillationADModel` |
| 归一化流 | CSFlow、FastFlow、CFlow、UFlow、DifferNet、PyramidFlow | `FlowBasedADModel` |
| 重建 | DRAEM、MemSeg、DeSTSeg、MemAE、FRE、GANomaly、DSR | `ReconstructionADModel` |
| 视觉语言 | WinCLIP、AnomalyCLIP、AnoVL、MuSc、AdaCLIP、AACLIP、AnomalyDINO | `VisionLanguageADModel` |
| 判别器 | SimpleNet、SuperSimpleNet、CFA | `DiscriminatorADModel` |
| 其他 | InvAD、ViTAD、UniAD、MambaAD、NSA、ResAD、CutPaste、GLASS、AST、UniNet、UniVAD | `BaseADModel` |
