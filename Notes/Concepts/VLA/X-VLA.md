---
type: concept
aliases: [X-VLA, XVLA, Cross-Embodied VLA]
---

# X-VLA

## 定义
跨具身视觉语言动作模型（Cross-Embodied VLA）：利用 Soft Prompt 将不同机器人形态信息编码为可学习 prefix token，在统一 VLM 骨干下实现多具身策略学习，是 X-DiffVLA 的重要前置工作。

## 核心要点
1. **Soft Prompt 具身编码**: 每种具身对应一组可学习 prefix token，注入 VLM 骨干，无需修改主干网络权重
2. **统一动作空间**: 异维度机器人通过零填充对齐到统一最大维度空间
3. **基线地位**: 在 RoboCasa（47.6%）和 Isaac Gym 实验中，X-DiffVLA 在其基础上分别提升约 17% 和 13.8%

## 代表工作
- X-VLA 原始工作（先于 X-DiffVLA 提出，具体年份/作者未在论文中明确）
- [[X-DiffVLA]]: 在 X-VLA 基础上引入 EBF 和 MPTD，进一步提升跨具身性能

## 相关概念
- [[VLA]]
- [[Soft Prompt]]
- [[X-DiffVLA]]
- [[EBF]]
