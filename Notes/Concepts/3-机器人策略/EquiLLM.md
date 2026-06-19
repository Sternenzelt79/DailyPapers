---
type: concept
aliases: [Equivariant LLM]
---

# EquiLLM

## 定义
为大型语言模型（LLM）引入等变性（通常是旋转等变）的方法，使得空间变换下模型输出保持几何一致性，主要用于 robotics 动作生成场景。

## 核心要点
1. 与 [[EquiVLA]] 类似，但针对 LLM backbone 本身的等变化
2. 通过修改 attention 机制或嵌入层实现几何一致性
3. 在 VLA 系统中作为等变方法的对比基线

## 代表工作
- [[EquiVLA]]: 相关框架，使用 EquiPerceptor + EquiActor 实现近似 SO(2) 等变

## 相关概念
- [[EquiVLA]]
- [[VLA]]
