---
type: concept
category: Training
tags: [reinforcement-learning, vla]
created: 2026-05-09
---

# RECAP

VLA 后训练阶段使用的强化学习框架，核心是用 [[VLM Critic]]（让 VLM 直接输出整数 token 当 value）替代传统回归头 critic，在数据采集 + 失败模式精修的迭代中改进策略。

## 代表工作

- [[RLDX-1]]: post-training 阶段用 RECAP + 文本式 VLM critic 提升复杂任务成功率。
