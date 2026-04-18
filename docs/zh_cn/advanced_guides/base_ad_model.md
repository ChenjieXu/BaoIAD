# BaseADModel 架构

`BaseADModel` 是 BaoIAD 中所有异常检测器的核心基类。它遵循 MMEngine 的骨干网络-颈部-头部分解约定，并实现了 3 模式前向分发。

## 类层次结构

```
BaseADModel
├── MemoryBankADModel
│   └── PatchCore、SPADE、PaDiM、DFM、DFKDE、RegAD、GraphCore
├── KnowledgeDistillationADModel
│   └── RD、RD++、STFPM、EfficientAD、Dinomaly
├── FlowBasedADModel
│   └── CSFlow、FastFlow、CFlow、UFlow、DifferNet、PyramidFlow
├── ReconstructionADModel
│   └── DRAEM、MemSeg、DeSTSeg、MemAE、FRE、GANomaly、DSR
├── VisionLanguageADModel
│   └── WinCLIP、AnomalyCLIP、AnoVL、MuSc、AdaCLIP、AACLIP、AnomalyDINO
├── DiscriminatorADModel
│   └── SimpleNet、SuperSimpleNet、CFA
└── BaseADModel（直接继承）
    └── InvAD、ViTAD、UniAD、MambaAD、NSA、ResAD、CutPaste、GLASS、AST、PNI 等
```

## 三模式前向分发

| 模式 | 调用方 | 需要实现的方法 |
|------|--------|---------------|
| `loss` | 训练循环 | `loss()` 返回 `Dict[str, Tensor]` |
| `predict` | 测试/验证循环 | `predict()` 返回 `List[ADDataSample]` |
| `tensor` | 特征提取 | `_forward()` 返回 `Tensor` 或 `Tuple[Tensor]` |

## 冻结行为

`BaseADModel.train()` 被重写以保持骨干网络处于评估模式，确保批归一化和 dropout 层始终使用推理统计量，这对一致的特征提取至关重要。
