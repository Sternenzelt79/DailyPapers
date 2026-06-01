---
type: concept
aliases: [RAVEN, Reinforcement Learning for Autoregressive Video Generation]
---

# RAVEN

## 定义
基于 [[CausVid]] 框架的在线强化学习对齐方法，通过 Training-time Test（TTT）和 CM-GRPO 奖励优化，解决因果 AR 视频扩散模型训练-推理阶段历史帧分布偏移（distribution gap）问题。

## 核心要点
1. CausVid 蒸馏训练时历史帧来自 GT，推理时来自模型自身输出，存在分布偏移
2. RAVEN 引入在线 RL 让模型在推理条件下进行自我修正
3. 使用 CM-GRPO（Consistency Model Group Relative Policy Optimization）作为奖励优化框架

## 代表工作
- [[RAVEN]]: 本概念对应论文

## 相关概念
- [[CausVid]]: RAVEN 的基础框架
- [[DMD]]: 扩散模型蒸馏
- [[CM-GRPO]]: 使用的 RL 优化方法
