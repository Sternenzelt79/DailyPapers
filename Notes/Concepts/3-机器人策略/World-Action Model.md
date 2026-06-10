---
type: concept
aliases: [WAM, World Action Model, 世界行动模型]
---

# World-Action Model

## 定义
将未来视频预测与机器人动作生成联合建模的框架，通过预测环境的未来状态来增强策略决策能力。

## 核心要点
1. 联合建模 $p(\mathbf{z}^v, a_{1:H} \mid o, l, s)$，未来视频潜变量作为动作生成的结构化条件
2. 与纯 VLA 方法相比，引入视觉世界模型先验，增强长时序规划和物理直觉
3. 代价是额外的视频生成计算开销，通常导致大参数量（5-8B）和高推理延迟

## 代表工作
- [[UWM]]: 5B WAM，统一世界模型方法
- [[GigaWorld-Policy]]: 5B，RoboTwin 2.0 基准
- [[Motus]]: 8B，目前仿真 SOTA
- [[Efficient-WAM]]: 1B，高效 WAM，32× 提速

## 相关概念
- [[Flow Matching]]
- [[Video Diffusion Model]]
- [[Action Chunking]]
- [[Diffusion Model]]
