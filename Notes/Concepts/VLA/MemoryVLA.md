---
type: concept
aliases: [MemoryVLA, 记忆增强 VLA]
---

# MemoryVLA

## 定义
MemoryVLA：为 VLA（Vision-Language-Action）模型引入显式记忆模块，使机器人能在 long-horizon、partially observable 任务中利用历史观测和动作信息进行决策。

## 数学形式
$$a_t = \pi_\theta(o_t, \mathcal{M}_t), \quad \mathcal{M}_t = \text{Memory}(o_{0:t-1}, a_{0:t-1})$$

## 核心要点
1. 解决标准 VLA 仅基于当前帧决策、缺乏记忆的根本局限
2. 记忆模块可以是 key-value memory、episodic memory 或压缩的隐状态
3. [[RoboMemArena]] benchmark 专门用于评估此类方法

## 代表工作
- MemoryVLA 相关工作（具体 arXiv 待查）
- [[RoboMemArena]]：针对记忆能力的 benchmark

## 相关概念
- [[VLA]]
- [[PrediMem]]
- [[Hierarchical Memory]]
