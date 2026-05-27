---
type: concept
aliases: [RT-1 Dataset, Robotics Transformer Dataset]
---

# RT-1

## 定义
RT-1 是 Google DeepMind 发布的大规模机器人操作数据集，配套 Robotics Transformer 1 模型，包含 130K+ 真实机器人演示，涵盖桌面抓取、放置等多种任务。

## 核心要点
1. 130,000+ 演示数据，13 种机器人任务
2. 配套 RT-1 Transformer 策略模型
3. 是 Open X-Embodiment 生态的基础数据集之一

## 数学形式
RT-1 使用 TokenLearner 将图像压缩为 8 个 token，再输入 Transformer 预测离散化动作。

## 代表工作
- [[RT-2]]: RT-1 的继任工作，引入视觉-语言预训练
- [[FineVLA]]: 作为 10 个源数据集之一，贡献 5,232 条轨迹

## 相关概念
- [[RT-2]]
- [[OpenVLA-OFT]]
