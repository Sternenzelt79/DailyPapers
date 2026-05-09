---
concept: MolmoAct2-Think
category: VLA
tags: [vla, world-model, reasoning, adaptive-depth]
created: 2026-05-09
---

# MolmoAct2-Think

## 定义

[[MolmoAct2]] 的"会思考"版本：在动作生成前先通过 [[Adaptive Depth Reasoning|自适应深度推理]] 维护场景的 10×10 深度 token 网格，作为隐式 [[World Model|世界模型]] 引导动作专家。

## 核心要点

1. **深度 token 作为世界模型表示**：100 个 token，从 128 大小的 [[VQ Codebook|VQ 码本]] 取出。
2. **时间局部性**：32×32 RGB patch 余弦相似度 < 0.996 才重预测对应深度 token。
3. **门控注入**：静止区域 KV 通过 sigmoid gate 弱化。
4. **三种输出风格**等概率采样训练：depth-only / action-only / depth-and-action。
5. **结果**：LIBERO 平均 98.1%（vs MolmoAct2 97.2%、π0.5 96.9%），长程任务 +3 点。

## 代表工作

- [[MolmoAct2]]: 主论文。

## 相关概念

- [[Adaptive Depth Reasoning]]
- [[World Model]]
- [[Depth Token]]
- [[VQ Codebook]]
- [[Chain-of-Thought]]
