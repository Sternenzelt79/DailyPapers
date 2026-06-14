---
type: concept
aliases: [世界专家, 世界预测器, World Model Expert]
---

# World Expert（世界专家）

## 定义

World Expert 是 WLA 框架中的视觉预测子网络，以物理动态表示为条件预测未来 VAE 潜变量（即未来视觉状态），在训练时通过梯度影响 backbone 学习更好的动态表示，推理时可完全移除以降低延迟。

## 数学形式

$$
\mathbf{o}_{t+n} = f_{wm}(\mathbf{h}_{t},\, \mathbf{o}_{t})
$$

## 核心要点

1. 基于 SANA-600M（含 VAE 共 900M 参数）
2. 预测 VAE 潜变量而非像素，降低建模复杂度
3. 训练期影响 backbone 参数优化（隐式世界建模），推理期可关闭
4. 测试时扩展（TTS）模式下重新启用，用于对想象轨迹打分

## 代表工作

- [[WLA]]: WLA 三专家架构之一，与 Action Expert 共享物理动态表示 $\mathbf{h}_t$

## 相关概念

- [[扩散世界模型]]
- [[Meta-Query]]
- [[价值模型]]
