---
type: concept
aliases: [TesserACT, TesserAct, Tesseract]
---

# TesserACT

## 定义
基于 CogVideoX 的 4D 具身世界模型，通过联合生成 RGB、深度和表面法向量来实现时空重建和动作预测，是当前视频世界模型用于机器人操纵的代表性工作。

## 核心要点
1. 在 CogVideoX 基础上微调，同时预测 RGB 视频、深度图和表面法向量（多输出头）
2. 显式 4D 监督方式：在输出空间添加几何量预测，需要大量深度和法向量标注
3. 逆动力学模块未开源，外部方法（如 GEM-4D 的 AIDS）需要自行处理其生成的视频
4. 限制：修改了输出空间，约束了预训练视频骨干的表达能力

## 代表工作
- [[GEM-4D]]: 将 TesserACT 作为主要对比基线，通过特征蒸馏（而非输出空间修改）在几何一致性和操作成功率上全面超越

参见 [[TesserAct]] 获取更多信息。

## 相关概念
- [[CogVideoX]]
- [[视频世界模型]]
- [[Diffusion Transformer]]
- [[几何基础模型]]
