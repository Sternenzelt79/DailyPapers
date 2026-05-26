---
type: concept
aliases: [EBF, Embodied Forcing, 具身强迫]
---

# EBF（Embodied Forcing）

## 定义
X-DiffVLA 中提出的具身形态编码方法：把机器人的形态信息（DoF、关节配置、运动学参数）编码为 soft prompt，注入 VLM 骨干，使模型无需具身特定 fine-tuning 即可适应不同机器人。

## 数学形式
$$h_\text{emb} = \text{Proj}(\text{Embed}(\text{morphology})), \quad \mathbf{x}_\text{input} = [\mathbf{x}_\text{lang}, h_\text{emb}, \mathbf{x}_\text{vis}]$$

把形态 embedding 作为 soft prefix token 拼接到语言和视觉 token 序列前。

## 核心要点
1. **无需具身 fine-tuning**：只改变输入 prompt，不修改模型权重
2. **形态编码**：DoF 数量、关节类型、运动学树结构均编码进 soft prompt
3. **跨具身泛化基础**：[[X-DiffVLA]] 的核心组件之一，配合 [[MCTD]] 实现跨具身迁移

## 代表工作
- [[X-DiffVLA]]：Li et al. 2026，提出 EBF 的原始工作

## 相关概念
- [[X-DiffVLA]]
- [[MCTD]]
- [[VLA]]
- [[Soft Prompt]]
