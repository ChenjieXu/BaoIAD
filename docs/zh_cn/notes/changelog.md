# 更新日志

## v0.1.0 (2025)

BaoIAD 首次发布。

### 方法（50+）

- **记忆库**：PatchCore、SPADE、PaDiM、DFM、DFKDE、RegAD、GraphCore
- **知识蒸馏**：RD、RD++、STFPM、EfficientAD、Dinomaly
- **归一化流**：CSFlow、FastFlow、CFlow、UFlow、DifferNet、PyramidFlow
- **重建**：DRAEM、MemSeg、DeSTSeg、MemAE、FRE、GANomaly、DSR
- **视觉语言**：WinCLIP、AnomalyCLIP、AnoVL、MuSc、AdaCLIP、AACLIP、AnomalyDINO
- **判别器**：SimpleNet、SuperSimpleNet、CFA
- **其他**：InvAD、ViTAD、UniAD、MambaAD、NSA、ResAD、CutPaste、GLASS、AST、PNI、RealNet、ComposeAD、UniNet、UniVAD、SAA+

### 数据集

MVTec AD、VisA、BTech、MVTec 3D AD、MVTec LOCO、MPDD、MVTec AD 2、Kolektor、VAD、RealIAD

### 特性

- 基于 MMEngine 的统一框架，支持配置继承
- 6 个专用 BaseADModel 子类
- 全面评估：12 项图像/像素级指标，包括 AUPRO 和 AUPIMO
- MemoryBankHook 自动管理记忆库生命周期
- 基准测试工具，支持每类别子进程执行和 JSON 输出
- 多 GPU 基准测试支持
