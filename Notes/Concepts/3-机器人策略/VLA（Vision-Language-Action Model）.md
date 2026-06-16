---
type: concept
aliases: [VLA, Vision-Language-Action Model, 视觉-语言-动作模型]
---

# VLA（Vision-Language-Action Model）

## 定义

视觉-语言-动作模型（VLA）是将视觉感知、语言理解与机器人动作预测统一在单一模型中的多模态机器人策略，通常以大型预训练视觉-语言模型（VLM）为主干，经微调后直接输出机器人控制指令。

## 核心要点

1. **统一架构**: 将机器人感知（图像）、语义理解（语言指令）和控制（动作）整合为端到端学习
2. **主干复用**: 利用 VLM 的预训练表示，通过少量机器人数据微调即可获得良好的泛化能力
3. **动作表示**: 通常使用 [[Action Chunking]] 预测动作序列，配合扩散或流匹配生成连续控制量
4. **在线 RL 微调**: 可在真实机器人环境中通过 RL 进一步优化，见 HABC、RIPT-VLA 等工作
5. **代表模型**: π0、π0.5、OpenVLA、RT-2、GR00T 等

## 代表工作

- [[π0.5]]: 基于流匹配的 VLA，HABC 的基础模型
- [[HABC]]: 通过层次优势加权实现 VLA 的在线 RL 微调
- [[RT-2]]: 早期将 LLM 迁移至机器人动作预测的工作
- [[OpenVLA-OFT]]: OpenVLA 的在线微调版本

## 相关概念

- [[Flow Matching]]
- [[Action Chunking]]
- [[强化学习]]
- [[π0.5]]
