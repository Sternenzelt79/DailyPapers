---
type: concept
aliases: [Video PreTraining, VPT, IDM pretraining]
---

# VPT

## 定义
OpenAI 提出的视频预训练方法：先训练 Inverse Dynamics Model（IDM）从视频预测动作，再用 IDM 标注大量无标签游戏视频，最后用标注数据预训练基础 policy。

## 数学形式
$$a_t = \text{IDM}(o_t, o_{t+1})$$

IDM 从相邻帧预测执行的动作，用少量标注数据训练后泛化到海量无标签视频。

## 核心要点
1. 解决大规模游戏视频无动作标注的问题
2. IDM 标注 → 预训练 → 下游微调（如 [[BASALT]]）
3. 展示了视频数据作为 policy 预训练来源的可行性
4. 在 Minecraft 中训练，产生强大的通用基础 policy
5. 在 world model 研究中作为 baseline policy 使用（见 [[PROWL]]）

## 代表工作
- Baker et al. (2022): Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos

## 相关概念
- [[BASALT]]
- [[Reinforcement Learning]]
- [[Inverse Dynamics Model]]
