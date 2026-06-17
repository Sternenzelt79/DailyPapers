---
type: concept
aliases: [DPG-Bench, Dense Prompt Graph Benchmark]
---

# DPG-Bench

## 定义
用于评估文本到图像生成模型对密集、复杂提示词理解和生成能力的 benchmark，特别关注多属性绑定和细粒度指令跟随。

## 核心要点
1. 提示词包含多个属性绑定（颜色、形状、数量、空间关系等）
2. 评估模型对复杂语义约束的准确率
3. 常与 [[GenEval]] 配合使用评估 T2I 模型

## 代表工作
- [[i1]]: 在 DPG-Bench 上验证完全开源 T2I 配方的效果

## 相关概念
- [[GenEval]]
- [[FLUX]]
- [[DiT]]
