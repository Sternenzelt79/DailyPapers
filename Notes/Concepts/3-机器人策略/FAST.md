---
type: concept
aliases: [Frequency-based Action Sequence Tokenizer, VLA动作分词器]
---

# FAST

## 定义
FAST（Frequency-based Action Sequence Tokenizer）是专为 VLA 设计的连续动作序列分词器，将连续动作轨迹离散化为 token 序列，使 VLA 能用自回归方式预测动作。

## 核心要点
1. 基于频域分析（DCT/小波变换）将连续动作序列压缩为紧凑 token 表示
2. 保留动作序列的时频结构，优于简单量化方法
3. 支持变长动作块（与固定 chunk 的 [[Action Chunking]] 互补）
4. 与 [[DiT]] 架构结合可加速 VLA 推理

## 代表工作
- π0-FAST: 将 FAST 集成到 π0 VLA 框架的首个系统化工作
- [[LabVLA]]: 使用 FAST token 预训练作为两阶段训练的第一阶段，赋予 VLM 动作语义理解

## 相关概念
- [[Action Chunking]]
- [[VLA]]
- [[Pi0-FAST]]
