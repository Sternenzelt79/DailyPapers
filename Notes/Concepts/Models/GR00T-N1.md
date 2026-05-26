---
type: concept
aliases: [GR00T N1, GR00T-N1, GROOT N1]
---

# GR00T-N1

## 定义
NVIDIA 发布的通用人形机器人基础模型，采用双系统架构（慢速推理 VLM + 快速动作扩散），支持多任务机器人控制。

## 核心要点
1. **双系统架构**: System 2（慢速 VLM 推理）驱动 System 1（快速扩散动作执行），兼顾泛化与实时性
2. **人形机器人专用**: 针对全身控制（双臂、腿部）设计，支持高自由度具身
3. **对比**: X-DiffVLA 在 RoboCasa 跨具身任务中成功率（64.5%）显著超越 GR00T-N1（39.5%）

## 代表工作
- GR00T N1: NVIDIA, 2025，通用人形机器人 VLA 基础模型

## 相关概念
- [[VLA]]
- [[Diffusion Policy]]
- [[X-DiffVLA]]
