---
concept: Vision-Language-Action Model
aliases: [VLA, Vision Language Action]
category: VLA
tags: [vla, robotics, foundation-model]
created: 2026-05-09
---

# Vision-Language-Action Model (VLA)

## 定义

VLA 是一类把 **视觉观测 + 语言指令 → 机器人动作** 端到端建模的基础模型。通常在大规模多本体（multi-embodiment）数据上预训练，再在目标本体上微调。

## 典型架构

- **Backbone**: 预训练 VLM（视觉编码器 + LLM）。
- **Action head**: 离散 token / 回归 / [[Diffusion Model]] / [[Flow Matching]]。
- **输出**: 单步动作 or [[Action Chunking|动作块]] $a_{t:t+H}$。

## 三个时代

| 时代 | 代表 | 特点 |
|------|------|------|
| 离散 token | [[RT-2]], [[OpenVLA]] | 动作离散化为 LLM token |
| 连续回归 | [[Octo]] | 直接回归 |
| 生成式 | [[Pi0]], [[Pi05]], [[RLDX-1]] | Flow Matching / Diffusion |

## 代表工作

- [[OpenVLA]]: 7B，OXE 训练。
- [[Pi0]] / [[Pi05]]: Flow matching 流派的代表。
- [[GR00T-N1.6]]: NVIDIA 人形 VLA。
- [[RLDX-1]]: 多流架构，运动/记忆/物理三模块。
- [[MolmoAct2]]: 完全开源多本体 VLA，Molmo2-ER + OpenFAST + Flow Matching。

## 相关概念

- [[Action Chunking]]
- [[Open-X-Embodiment]]
- [[Flow Matching]]
- [[Multi-Stream Action Transformer]]
