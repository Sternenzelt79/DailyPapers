---
type: concept
aliases: [HOIDiff, 扩散HOI生成]
---

# HOIDiff

## 定义

基于扩散模型的人-物交互生成方法，通过条件扩散过程生成时序一致的人体与物体交互运动序列。

## 核心要点

1. 采用扩散模型生成 HOI 序列，具备多样性
2. 未针对机器人形态约束设计，追踪成功率低
3. 在 GRAIL 对比中：SR 仅 15.8%，Contact Distance 0.012

## 代表工作

- [[GRAIL]]: 作为 HOI 生成基线，GRAIL SR 88.9% vs HOIDiff SR 15.8%

## 相关概念

- [[HOI]]
- [[扩散模型]]
- [[CHOIS]]
- [[DAViD]]
