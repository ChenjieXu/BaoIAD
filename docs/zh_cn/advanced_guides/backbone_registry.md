# 骨干网络注册器

## 可用骨干网络

| 类 | 用途 |
|---|------|
| `TIMMBackbone` | 大多数检测器；包装 timm 模型 |
| `OpenCLIPBackbone` | 视觉语言方法 |
| `DINOv2Backbone` | Dinomaly |
| `ViTEncoderBackbone` / `DistilledVisionTransformerBackbone` | ViTAD |
| `EfficientNetFeatureExtractor` / `GenericFeatureExtractor` | AST |
| `CSFlowFeatureExtractor` | CSFlow 专用 |
| `MuScCLIPBackbone` | MuSc（多层 CLIP + 位置嵌入缩放） |
| `GraphCoreViGBackbone` | GraphCore |
| `RawBackbone` | 遗留/透传；新实现请优先使用 TIMMBackbone |

## 特征层索引

`out_indices` 参数控制提取哪些特征层，使用 0 索引。不同方法需要不同的层：

| 模型 | 常用 `out_indices` | 使用方法 |
|------|-------------------|---------|
| `wide_resnet50_2` | `(2, 3)` | PatchCore、RD、SPADE、SimpleNet |
| `wide_resnet50_2` | `(1, 2, 3)` | PaDiM、CFA、STFPM |
| `resnet18` | `(1, 2, 3)` | FastFlow、STFPM |
| `efficientnet_b4` | `(1, 2, 3, 4)` | UniAD、MambaAD |
