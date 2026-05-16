---
type: concept
aliases: [UniPi, Learning Universal Policies via Text-Guided Video Generation]
---

# UniPi

## 定义
UniPi：通过文本引导的视频生成来学习通用策略——将 planning 问题转化为视频生成问题，生成的视频帧序列即为执行计划。

## 数学形式
$$\pi(a | o, g) \approx p_\theta(\text{video} | o, g) \rightarrow \text{extract actions}$$

## 核心要点
1. 用文本条件视频扩散模型生成"规划视频"，从视频中提取动作
2. 无需显式动作标注，以视频预测作为 implicit world model
3. 开创了"video generation as planning"范式
4. 局限：视频生成慢，动作提取需要额外的逆动力学模型

## 代表工作
- Du et al. 2023，Learning Universal Policies via Text-Guided Video Generation
- 后续：[[GigaWorld]]、[[World Model]] 等扩展了这一范式

## 相关概念
- [[World Model]]
- [[VLA]]
- [[Diffusion Model]]
