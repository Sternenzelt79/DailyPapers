---
type: concept
aliases: [UniVLA]
---

# UniVLA

## 定义
统一多具身 VLA 模型：通过统一的语言动作接口，单一模型支持多种机器人形态（dual-arm、mobile manipulator 等）执行多样化操作任务。

## 核心要点
1. **跨具身统一**：单一模型权重覆盖多种 embodiment，不需要具身特定 fine-tuning
2. **语言动作接口**：把不同机器人的控制信号统一为语言描述格式
3. **被多篇 VLA 工作比较**：[[QuoVLA]]、[[X-DiffVLA]] 均将 UniVLA 作为 baseline 之一

## 代表工作
- Li et al.《UniVLA: Learning to Act Anywhere with Task-centric Token Projection》

## 相关概念
- [[VLA]]
- [[X-DiffVLA]]
- [[OpenVLA]]
