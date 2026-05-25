---
type: concept
aliases: [RDT, Robotics Diffusion Transformer, RDT-1B]
---

# RDT

## 定义
Robotics Diffusion Transformer，基于扩散模型的机器人通用策略网络，以 Transformer 为骨架，支持多模态指令（文本、图像）条件下的动作生成；RDT-1B 是 10 亿参数版本，在双臂操作任务上表现突出。

## 核心要点
1. 基于 [[扩散Transformer]]，将扩散过程用于动作生成
2. 支持语言 + 图像多模态条件输入
3. 在大规模机器人演示数据上预训练，具有较强的跨任务泛化
4. 通过 flow matching 或 DDPM 解码动作序列
5. 常被作为 VLA fine-tuning 研究中的竞争基线

## 代表工作
- [[Agentic-VLA]]: 将 RDT 作为对比 baseline，在在线适应任务上对比 GRPO 微调效果

## 相关概念
- [[Diffusion Policy]]
- [[扩散Transformer]]
- [[Action Chunking]]
